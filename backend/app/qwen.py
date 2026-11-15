"""Asynchronous OpenAI-compatible HTTP client with validated model output."""
import asyncio
import json
import logging
import time

import httpx
from pydantic import ValidationError

from .config import Settings
from .catalog import NAMES
from .errors import BackendError
from .prompts import messages
from .schemas import Draft, GenerateRequest, GenerateResponse, ModelResponse
from .titles import repair_title

logger = logging.getLogger(__name__)
SCHEMA_FIELDS = {'drafts', 'categoryId', 'title', 'text'}


def validation_issues(error: ValidationError):
    """Pydantic metadata only; never log input, message or exception string."""
    issues = []
    for item in error.errors(include_input=False, include_context=False, include_url=False)[:8]:
        location = '.'.join(str(part) if isinstance(part, int) else
                            part if isinstance(part, str) and part in SCHEMA_FIELDS else '<other>'
                            for part in item['loc'])
        issues.append(f'{location}:{item["type"]}')
    return issues


def parse_drafts(content: str, request_id: str):
    """Repair title formatting only; keep strict validation for other fields."""
    body = json.loads(content)
    if isinstance(body, dict) and isinstance(body.get('drafts'), list):
        for draft in body['drafts']:
            if not isinstance(draft, dict) or not isinstance(draft.get('title'), str):
                continue
            category_id = draft.get('categoryId')
            if type(category_id) is not int or category_id not in NAMES or not draft['title'].strip():
                continue
            repaired = repair_title(draft['title'], category_id)
            if repaired != draft['title']:
                draft['title'] = repaired
                logger.info('qwen_title_repaired request_id=%s category_id=%d', request_id, category_id)
    return ModelResponse.model_validate(body)


class QwenClient:
    def __init__(self, client: httpx.AsyncClient, settings: Settings):
        self.client = client
        self.settings = settings
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    async def ready(self):
        try:
            response = await self.client.get('models', timeout=5)
            response.raise_for_status()
            body = response.json()
            data = body.get('data') if isinstance(body, dict) else None
            return isinstance(data, list) and any(isinstance(row, dict) and row.get('id') == self.settings.qwen_model for row in data)
        except (httpx.HTTPError, ValueError, KeyError, TypeError):
            return False

    async def output_budget(self, chat_messages, category_count):
        if not self.settings.tokenize_before_generate:
            return self.settings.max_tokens
        # /tokenize is a vLLM endpoint outside /v1. Preserve a possible proxy
        # prefix before /v1, and never send credentials to another origin.
        base = self.client.base_url
        path = base.path.rstrip('/')
        prefix = path[:-3] if path.endswith('/v1') else path
        response = await self.client.post(base.copy_with(path=prefix + '/tokenize'), json={
            'model': self.settings.qwen_model, 'messages': chat_messages, 'add_generation_prompt': True,
        })
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict): raise ValueError('Expected token budget')
        count, limit = body.get('count'), body.get('max_model_len')
        if type(count) is not int or type(limit) is not int or count < 0 or limit <= 0:
            raise ValueError('Invalid token budget')
        available = limit - count - 256
        minimum = max(512, 256 * category_count)
        if available < minimum:
            raise BackendError(422, 'Исходный текст вместе с примерами не помещается в контекст Qwen. Сократите описание или выберите меньше категорий.')
        return min(self.settings.max_tokens, available)

    async def generate(self, request: GenerateRequest, *, request_id: str = '-'):
        started = time.monotonic()
        category_ids = [c.id for c in request.categories]
        logger.info('qwen_start request_id=%s categories=%s description_chars=%d',
                    request_id, category_ids, len(request.description))
        payload = {
            'model': self.settings.qwen_model, 'messages': messages(request),
            'temperature': 0, 'max_tokens': self.settings.max_tokens,
            'stream': False,
        }
        if self.settings.response_format == 'json_object':
            payload['response_format'] = {'type': 'json_object'}
        elif self.settings.response_format == 'json_schema':
            schema = ModelResponse.model_json_schema()
            schema['$defs']['ModelDraft']['properties']['categoryId'] = {
                'type': 'integer', 'enum': [c.id for c in request.categories],
            }
            payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'advert_drafts', 'strict': True, 'schema': schema,
            }}
        try:
            stage = 'token_budget'
            # Includes time waiting for the model concurrency slot.
            async with asyncio.timeout(self.settings.timeout_seconds):
                async with self.semaphore:
                    payload['max_tokens'] = await self.output_budget(payload['messages'], len(request.categories))
                    stage = 'completion_http'
                    response = await self.client.post('chat/completions', json=payload)
                    response.raise_for_status()
                    stage = 'completion_envelope'
                    body = response.json()
                    if not isinstance(body, dict) or not isinstance(body.get('choices'), list) or not body['choices'] or not isinstance(body['choices'][0], dict):
                        raise ValueError('Expected completion choices')
                    completion = body['choices'][0]
                    if completion.get('finish_reason') != 'stop':
                        finish = completion.get('finish_reason')
                        logger.warning('qwen_rejected request_id=%s stage=completion_finish reason=%s',
                                       request_id, finish if finish in ('length', 'content_filter', 'tool_calls') else 'other')
                        raise BackendError(502, 'Qwen не завершил ответ. Проверьте лимит токенов и повторите генерацию.')
                    message = completion.get('message')
                    if not isinstance(message, dict): raise ValueError('Expected completion message')
                    content = message.get('content')
                    if not isinstance(content, str):
                        raise ValueError('Expected text completion')
                    stage = 'model_schema'
                    try:
                        result = parse_drafts(content, request_id)
                    except ValidationError as exc:
                        logger.warning('qwen_rejected request_id=%s stage=model_schema issues=%s output_chars=%d',
                                       request_id, validation_issues(exc), len(content))
                        raise BackendError(502, 'Qwen вернул некорректный JSON или структуру черновиков.') from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            logger.warning('qwen_failed request_id=%s stage=%s reason=timeout', request_id, stage)
            raise BackendError(504, 'Qwen не ответил за отведённое время. Повторите попытку.') from exc
        except httpx.HTTPStatusError as exc:
            logger.warning('qwen_failed request_id=%s stage=%s reason=http_status status=%d',
                           request_id, stage, exc.response.status_code)
            if exc.response.status_code in (429, 503):
                raise BackendError(503, 'Qwen занят или временно недоступен. Повторите попытку позже.') from exc
            raise BackendError(502, 'Qwen отклонил запрос. Проверьте имя модели и настройки API.') from exc
        except httpx.RequestError as exc:
            logger.warning('qwen_failed request_id=%s stage=%s reason=network error_type=%s',
                           request_id, stage, type(exc).__name__)
            raise BackendError(502, 'Не удалось подключиться к Qwen. Проверьте API и SSH-туннель.') from exc
        except (ValueError, KeyError, IndexError, TypeError, ValidationError) as exc:
            logger.warning('qwen_failed request_id=%s stage=%s reason=invalid_envelope error_type=%s',
                           request_id, stage, type(exc).__name__)
            raise BackendError(502, 'Qwen вернул некорректный JSON или структуру черновиков.') from exc

        expected = [c.id for c in request.categories]
        actual = [draft.categoryId for draft in result.drafts]
        if len(actual) != len(set(actual)) or set(actual) != set(expected):
            logger.warning('qwen_rejected request_id=%s stage=category_match expected=%s actual=%s',
                           request_id, expected, actual)
            raise BackendError(502, 'Ответ Qwen не соответствует выбранным категориям.')
        by_id = {draft.categoryId: draft for draft in result.drafts}
        drafts = [Draft(categoryId=category, title=by_id[category].title,
                        description=by_id[category].text) for category in expected]
        logger.info('qwen_success request_id=%s categories=%s elapsed_ms=%d',
                    request_id, expected, int((time.monotonic() - started) * 1000))
        return GenerateResponse(drafts=drafts)
