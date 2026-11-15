"""Unified single/batch input and incremental creation pipeline."""
import asyncio
import csv
from dataclasses import dataclass
import io
import json
import logging
import time

from pydantic import Field, ValidationError

from .classifier import classify
from .errors import BackendError
from .observability import event, failure
from .schemas import Category, ClassificationRequest, GenerateRequest

logger = logging.getLogger(__name__)
MAX_BODY_BYTES = 5 * 1024 * 1024
MAX_ITEMS = 300


class CreateItem(ClassificationRequest):
    testcase: str | None = Field(default=None, max_length=200)
    categories: list[Category] | None = Field(default=None, min_length=1, max_length=6)


@dataclass
class InvalidCreateItem:
    error: BackendError
    testcase: str | None = None


def csv_delimiter(source):
    for delimiter in (',', ';'):
        try:
            header = next(csv.reader(io.StringIO(source, newline=''), delimiter=delimiter, strict=True), [])
        except csv.Error:
            continue
        if 'description' in header:
            return delimiter
    return None


def invalid_item(index, value, message):
    testcase = value.get('testcase') if isinstance(value, dict) else None
    return InvalidCreateItem(
        BackendError(422, f'Объявление {index + 1}: {message}'),
        testcase if isinstance(testcase, str) and len(testcase) <= 200 else None,
    )


def parse_creation(body: bytes, content_type: str, generate_text: bool | None = None,
                   *, default_title='', allow_invalid_records=False):
    """Parse and normalize records on the backend, including uploaded input."""
    if len(body) > MAX_BODY_BYTES:
        raise BackendError(413, 'Максимальный размер запроса — 5 МБ.')
    try:
        source = body.decode('utf-8-sig')
        if not isinstance(default_title, str) or len(default_title.strip()) > 100:
            raise ValueError('Название объявления должно быть строкой не длиннее 100 символов.')
        default_title = default_title.strip()
        kind = content_type.split(';')[0].strip().lower()
        if kind in ('auto', 'text/plain'):
            if source.lstrip().startswith(('{', '[')):
                kind = 'application/json'
            elif csv_delimiter(source):
                kind = 'text/csv'
            else:
                kind = 'text/plain'
        if kind in ('text/csv', 'application/csv'):
            delimiter = csv_delimiter(source) or ','
            reader = csv.DictReader(io.StringIO(source, newline=''), delimiter=delimiter, strict=True)
            if reader.fieldnames and len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise ValueError('В CSV повторяются названия столбцов.')
            if not reader.fieldnames or 'description' not in reader.fieldnames:
                raise ValueError('CSV должен содержать столбец description.')
            values = []
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    if allow_invalid_records:
                        values.append(invalid_item(len(values), row, 'число полей CSV не совпадает с заголовком.'))
                        if len(values) > MAX_ITEMS:
                            raise ValueError(f'В одном запросе допускается до {MAX_ITEMS} объявлений.')
                        continue
                    raise ValueError('Число полей CSV не совпадает с заголовком.')
                value = {key: value for key, value in row.items() if value != ''}
                values.append(value)
                if len(values) > MAX_ITEMS:
                    raise ValueError(f'В одном запросе допускается до {MAX_ITEMS} объявлений.')
            batch = True
            mode = True
        elif kind == 'application/json':
            payload = json.loads(source)
            mode = True
            if isinstance(payload, dict) and 'items' in payload:
                if set(payload) - {'items', 'generateText'}:
                    raise ValueError('Неизвестные поля пакета JSON.')
                mode = payload.get('generateText', True)
                values, batch = payload['items'], True
            elif isinstance(payload, list):
                values, batch = payload, True
            elif isinstance(payload, dict):
                mode = payload.pop('generateText', True)
                values, batch = [payload], False
            else:
                raise ValueError('Ожидается объект объявления, массив или объект с items.')
        elif kind == 'text/plain':
            values, batch, mode = [{'description': source}], False, True
        else:
            raise BackendError(415, 'Используйте JSON, CSV или обычный текст в UTF-8.')
        if not isinstance(mode, bool):
            raise ValueError('generateText должен быть boolean.')
        if not isinstance(values, list) or not 1 <= len(values) <= MAX_ITEMS:
            raise ValueError(f'Передайте от 1 до {MAX_ITEMS} объявлений.')
        items = []
        for index, value in enumerate(values):
            if isinstance(value, InvalidCreateItem):
                items.append(value)
                continue
            try:
                if isinstance(value, dict):
                    value = dict(value)
                    if kind in ('text/csv', 'application/csv') and 'mcId' in value:
                        value['mcId'] = int(value['mcId'])
                    title = value.get('title', '')
                    if default_title and isinstance(title, str) and not title.strip():
                        value['title'] = default_title
                item = CreateItem.model_validate(value)
                if item.categories and len({c.id for c in item.categories}) != len(item.categories):
                    raise ValueError('Категории не должны повторяться.')
                items.append(item)
            except (ValidationError, ValueError) as exc:
                if allow_invalid_records:
                    items.append(invalid_item(index, value, 'проверьте description, title, mcId и категории.'))
                    continue
                raise ValueError(f'Объявление {index + 1}: проверьте description, mcId и остальные поля.') from exc
        return items, batch, mode if generate_text is None else generate_text
    except (ValueError, UnicodeError, csv.Error) as exc:
        message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, (json.JSONDecodeError, UnicodeError)) else 'Некорректный JSON/CSV. Используйте UTF-8 и проверьте кавычки.'
        raise BackendError(422, message) from exc


async def creation_events(app, items, generate_text, request_id, *, batch=False):
    """Bound item workers; run synchronous C_syntax in actual backend threads.

    Qwen HTTP calls stay asynchronous and have a shared concurrency limit.
    Normally one model call writes all categories of an item to avoid mixing
    cross-category details. On a technical model failure, retry categories
    independently to preserve any valid sibling drafts.
    """
    started = time.monotonic()
    event(logger, 'batch_start', items=len(items), generate_text=generate_text)
    queue = asyncio.Queue(maxsize=32)
    pending = iter(enumerate(items))
    completed = 0
    failed = 0
    classified = {}

    async def process(index, item):
        if isinstance(item, InvalidCreateItem):
            raise item.error
        identity = {'index': index, **({'testcase': item.testcase} if item.testcase is not None else {})}
        item_started = time.monotonic()
        event(logger, 'creation_item_start', index=index, source_id=item.mcId, manual_categories=item.categories is not None)
        categories = item.categories
        if categories is None:
            model = app.state.classifier
            if model is None:
                raise BackendError(503, 'Классификатор C_syntax недоступен.')
            values = await app.state.inference.run(classify, model, item.description, item.mcId)
            categories = [Category.model_validate(value) for value in values]
        event(logger, 'creation_categories_ready', index=index, categories=[c.id for c in categories])
        classified[index] = [c.id for c in categories]
        await queue.put({'type': 'categories', **identity, 'categories': [c.model_dump() for c in categories]})
        parts = {}
        errors = []

        async def call_qwen(selected):
            event(logger, 'creation_generation_start', index=index, categories=[c.id for c in selected])
            tunnel = app.state.tunnel
            if tunnel is not None and not tunnel.alive:
                raise BackendError(503, 'SSH-туннель отключён и автоматически переподключается.')
            async with app.state.creation_slots:
                return await app.state.qwen.generate(GenerateRequest(
                    title=item.title, description=item.description, mcId=item.mcId,
                    categories=selected,
                ), request_id=request_id)

        async def emit_draft(category, draft):
            values = [{'mcId': category.id, 'title': draft.title, 'text': draft.description}]
            parts[category.id] = values
            event(logger, 'creation_draft_ready', index=index, category_id=category.id)
            await queue.put({'type': 'drafts', **identity, 'drafts': values, 'warnings': []})

        async def create_category(category):
            try:
                if not generate_text:
                    values = [{'mcId': category.id, 'title': None, 'text': None}]
                    parts[category.id] = values
                    await queue.put({'type': 'drafts', **identity, 'drafts': values, 'warnings': []})
                else:
                    result = await call_qwen([category])
                    await emit_draft(category, result.drafts[0])
            except BackendError as exc:
                failure(logger, 'creation_category_failed', exc, index=index, category_id=category.id)
                errors.append({'mcId': category.id, 'message': exc.message, 'code': exc.code, 'requestId': request_id})
                await queue.put({'type': 'category_error', **identity, 'mcId': category.id, 'message': exc.message, 'code': exc.code, 'requestId': request_id})

        if generate_text and len(categories) > 1:
            try:
                result = await call_qwen(categories)
                for category, draft in zip(categories, result.drafts, strict=True):
                    await emit_draft(category, draft)
            except BackendError as exc:
                failure(logger, 'creation_batch_generation_failed', exc, index=index)
                logger.warning('creation_batch_retry request_id=%s index=%d status=%d',
                               request_id, index, exc.status_code)
                tasks = [asyncio.create_task(create_category(category)) for category in categories]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        else:
            tasks = [asyncio.create_task(create_category(category)) for category in categories]
            try:
                if tasks:
                    await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
        result = {**identity, 'microcats':[c.id for c in categories],
                  'drafts': [draft for c in categories for draft in parts.get(c.id, [])]}
        if errors:
            result['errors'] = errors
        event(logger, 'creation_item_complete', index=index, drafts=len(result['drafts']), errors=len(errors), elapsed_ms=int((time.monotonic()-item_started)*1000))
        return result

    async def worker():
        nonlocal completed, failed
        for index, item in pending:
            identity = {'index': index, **({'testcase': item.testcase} if item.testcase is not None else {})}
            try:
                result = await process(index, item)
                if result.get('errors'):
                    failed += 1
            except Exception as exc:
                failed += 1
                failure(logger, 'creation_item_failed', exc, index=index)
                result = {**identity, 'microcats':classified.get(index), 'drafts': [], 'error': {'message': exc.message if isinstance(exc, BackendError) else 'Не удалось обработать объявление.', 'code': exc.code if isinstance(exc, BackendError) else 'internal_error', 'requestId': request_id}}
            completed += 1
            await queue.put({'type': 'item_done', **result, 'completed': completed, 'total': len(items)})

    workers = [asyncio.create_task(worker()) for _ in range(min(len(items), app.state.creation_workers))]
    async def finish():
        await asyncio.gather(*workers)
        event(logger, 'batch_complete', completed=completed, failed=failed, elapsed_ms=int((time.monotonic()-started)*1000))
        await queue.put({'type': 'done', 'total': len(items), 'completed': completed, 'failed': failed})
    finisher = asyncio.create_task(finish())
    try:
        yield {'type': 'start', 'total': len(items), 'generateText': generate_text, 'batch': batch}
        while True:
            try:
                payload_event = await asyncio.wait_for(queue.get(), timeout=15)
            except TimeoutError:
                yield {'type': 'heartbeat'}
                continue
            yield payload_event
            if payload_event['type'] == 'done':
                break
    finally:
        if not finisher.done():
            event(logger, 'batch_cancelled', completed=completed, total=len(items))
        finisher.cancel()
        for worker_task in workers:
            worker_task.cancel()
        await asyncio.gather(finisher, *workers, return_exceptions=True)
