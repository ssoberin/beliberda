"""Asynchronous OpenAI-compatible HTTP client with validated model output."""
import asyncio
import logging
import re
import time

import httpx
from pydantic import ValidationError

from .config import Settings
from .catalog import NAMES
from .errors import BackendError
from .prompts import messages
from .schemas import Draft, GenerateRequest, GenerateResponse, GenerationWarning, ModelResponse

logger = logging.getLogger(__name__)
SCHEMA_FIELDS = {'drafts', 'categoryId', 'supported', 'title', 'text', 'evidence'}


def validation_issues(error: ValidationError):
    """Pydantic metadata only; never log input, message or exception string."""
    issues = []
    for item in error.errors(include_input=False, include_context=False, include_url=False)[:8]:
        location = '.'.join(str(part) if isinstance(part, int) else
                            part if isinstance(part, str) and part in SCHEMA_FIELDS else '<other>'
                            for part in item['loc'])
        issues.append(f'{location}:{item["type"]}')
    return issues


def grounded(draft, description: str):
    # llama.cpp may change capitalization or line breaks in an otherwise
    # verbatim citation. Require the same words and punctuation in the source;
    # never treat a paraphrase or an invented operation as evidence.
    def found(quote):
        parts = quote.split()
        if not parts:
            return False
        return re.search(r'\s+'.join(map(re.escape, parts)), description, re.IGNORECASE) is not None

    if any(not found(quote) for quote in draft.evidence):
        raise BackendError(502, 'Qwen вернул черновик без точных подтверждающих цитат. Повторите генерацию.')
    output = draft.title + '\n' + draft.text
    # Conservative checks, not a complete semantic hallucination detector.
    numbers = lambda text: {x.replace(',', '.') for x in re.findall(r'\d+(?:[.,]\d+)?', text)}
    if not numbers(output) <= numbers(description):
        raise BackendError(502, 'Qwen добавил числовые сведения, отсутствующие в исходном тексте.')
    for pattern in (r'\bгаранти\w*', r'\bскидк\w*', r'\bбесплатн\w*', r'\b\d+\s*лет\s*опыта'):
        if re.search(pattern, output, re.I) and not re.search(pattern, description, re.I):
            raise BackendError(502, 'Qwen добавил неподтверждённые рекламные условия.')
    if re.search(r'\b(?:исходн\w* объявлен\w*|из объявления)\b', output, re.I):
        raise BackendError(502, 'Qwen вернул служебное пояснение вместо самостоятельного объявления.')


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

    async def generate(self, request: GenerateRequest, *, request_id: str = '-', allow_retry: bool = True):
        started = time.monotonic()
        category_ids = [c.id for c in request.categories]
        logger.info('qwen_start request_id=%s attempt=%s categories=%s description_chars=%d',
                    request_id, 'initial' if allow_retry else 'focused', category_ids, len(request.description))
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
                        result = ModelResponse.model_validate_json(content)
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
        drafts, warnings = [], []
        def placeholder(category):
            return Draft(categoryId=category, title=NAMES[category],
                         description=f'{NAMES[category]}. [Уточните, какие работы вы предлагаете, и условия заказа.]')

        for category in expected:
            draft = by_id[category]
            if not draft.supported:
                warnings.append(GenerationWarning(
                    categoryId=category,
                    message='Черновик создан по выбранной категории. В исходнике недостаточно сведений об услуге — уточните текст перед публикацией.',
                ))
                if draft.title is None and draft.text is None:
                    # Older/refusing model outputs must not silently remove a
                    # selected category. Return an explicit editable placeholder.
                    drafts.append(placeholder(category))
                    continue
            try:
                grounded(draft, request.description)
            except BackendError as exc:
                # Do not log the quote, draft, or source text. Only the check type.
                logger.warning('qwen_rejected request_id=%s stage=grounding category_id=%d reason=%s',
                               request_id, category, {
                                   'Qwen вернул черновик без точных подтверждающих цитат. Повторите генерацию.': 'evidence',
                                   'Qwen добавил числовые сведения, отсутствующие в исходном тексте.': 'numbers',
                                   'Qwen добавил неподтверждённые рекламные условия.': 'claims',
                                   'Qwen вернул служебное пояснение вместо самостоятельного объявления.': 'meta_text',
                               }.get(exc.message, 'other'))
                # A multi-category response may botch a citation even when the
                # service is genuinely offered. Retry ONLY this category once;
                # never expose the failed draft or skip normal validation.
                if allow_retry:
                    remaining = self.settings.timeout_seconds - (time.monotonic() - started)
                    if remaining > 1:
                        focused = request.model_copy(update={
                            'categories': [next(c for c in request.categories if c.id == category)],
                        })
                        try:
                            repaired = await asyncio.wait_for(
                                self.generate(focused, request_id=request_id, allow_retry=False),
                                timeout=remaining,
                            )
                        except (BackendError, TimeoutError) as retry_error:
                            logger.warning('qwen_retry_failed request_id=%s category_id=%d reason=%s',
                                           request_id, category, type(retry_error).__name__)
                        else:
                            logger.info('qwen_retry_complete request_id=%s category_id=%d warnings=%d',
                                        request_id, category, len(repaired.warnings))
                            if warnings and warnings[-1].categoryId == category:
                                warnings.pop()
                            warnings.extend(repaired.warnings)
                            drafts.extend(repaired.drafts)
                            continue
                if warnings and warnings[-1].categoryId == category:
                    warnings.pop()
                warnings.append(GenerationWarning(
                    categoryId=category,
                    message='Модель не подтвердила сведения для этой категории исходным текстом. Черновик заменён заготовкой; уточните услугу вручную.',
                ))
                drafts.append(placeholder(category))
                continue
            drafts.append(Draft(categoryId=category, title=draft.title.strip(), description=draft.text.strip()))
        logger.info('qwen_success request_id=%s categories=%s warnings=%d elapsed_ms=%d',
                    request_id, expected, len(warnings), int((time.monotonic() - started) * 1000))
        return GenerateResponse(drafts=drafts, warnings=warnings)
