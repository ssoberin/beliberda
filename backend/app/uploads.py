"""Receive original files; parsing and record normalization belong to backend."""
from pathlib import Path

from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.formparsers import MultiPartException
from starlette.requests import Request

from .creation import MAX_BODY_BYTES, parse_creation
from .errors import BackendError

MAX_REQUEST_BYTES = MAX_BODY_BYTES + 64 * 1024  # Multipart headers and title.
FILE_TYPES = {'.json': 'application/json', '.csv': 'text/csv', '.txt': 'text/plain'}


async def read_creation_request(request: Request, generate_text=None):
    content_type = request.headers.get('content-type', '')
    if content_type.partition(';')[0].strip().lower() != 'multipart/form-data':
        chunks, size = [], 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise BackendError(413, 'Максимальный размер запроса — 5 МБ.')
            chunks.append(chunk)
        parsed = await run_in_threadpool(parse_creation, b''.join(chunks), content_type, generate_text)
        return *parsed, size

    size = 0

    async def limited_receive():
        nonlocal size
        message = await request.receive()
        if message['type'] == 'http.request':
            size += len(message.get('body', b''))
            if size > MAX_REQUEST_BYTES:
                # Starlette closes temporary files when multipart parsing fails.
                raise MultiPartException('Максимальный размер файла — 5 МБ.')
        return message

    bounded = Request(request.scope, receive=limited_receive)
    try:
        async with bounded.form(max_files=1, max_fields=2, max_part_size=MAX_BODY_BYTES) as form:
            if set(form) - {'file', 'text', 'title'} or any(len(form.getlist(key)) != 1 for key in form):
                raise BackendError(422, 'Допустимы поля file или text и необязательное title, без повторов.')
            if ('file' in form) == ('text' in form):
                raise BackendError(422, 'Передайте один файл или текст.')
            title = form.get('title', '')
            if not isinstance(title, str):
                raise BackendError(422, 'Название объявления должно быть строкой.')
            if 'file' in form:
                upload = form['file']
                if not isinstance(upload, UploadFile):
                    raise BackendError(422, 'Поле file должно содержать файл.')
                kind = FILE_TYPES.get(Path(upload.filename or '').suffix.lower())
                if kind is None:
                    raise BackendError(415, 'Выберите файл .json, .csv или .txt.')
                body = await upload.read(MAX_BODY_BYTES + 1)
            else:
                text = form['text']
                if not isinstance(text, str):
                    raise BackendError(422, 'Поле text должно содержать текст.')
                body, kind = text.encode('utf-8'), 'text/plain'
            parsed = await run_in_threadpool(
                parse_creation, body, kind, generate_text,
                default_title=title, allow_invalid_records=True,
            )
            return *parsed, len(body)
    except HTTPException as error:
        if size > MAX_REQUEST_BYTES or (isinstance(error.detail, str) and error.detail.startswith('Part exceeded maximum size')):
            raise BackendError(413, 'Максимальный размер файла — 5 МБ.') from error
        if error.status_code == 400:
            raise BackendError(422, f'Некорректная загрузка файла: {error.detail}') from error
        raise
