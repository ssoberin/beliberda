"""Application logs with request correlation and tracebacks without payload dumps."""
from contextvars import ContextVar
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import time
import traceback
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx
from pydantic import ValidationError
from starlette.responses import JSONResponse

from .errors import BackendError

request_id_context = ContextVar('backend_request_id', default='-')
LOG = logging.getLogger('backend.http')


class LogFormatter(logging.Formatter):
    def format(self, record):
        record.correlation_id = request_id_context.get()
        return super().format(record)

    def formatTime(self, record, datefmt=None):
        return datetime.fromtimestamp(record.created, ZoneInfo('Europe/Moscow')).isoformat(timespec='milliseconds')


def configure_logging():
    package = logging.getLogger('backend')
    level = os.getenv('BACKEND_LOG_LEVEL', 'INFO').upper()
    package.setLevel(getattr(logging, level, logging.INFO))
    package.propagate = False
    formatter = LogFormatter('%(asctime)s %(levelname)-7s %(name)s request_id=%(correlation_id)s %(message)s')
    if not any(getattr(h, '_backend_console', False) for h in package.handlers):
        handler = logging.StreamHandler()
        handler._backend_console = True
        handler.setFormatter(formatter)
        package.addHandler(handler)
    filename = os.getenv('BACKEND_LOG_FILE', str(Path(__file__).resolve().parents[1] / 'logs/backend.log'))
    if filename and not any(getattr(h, '_backend_file', False) for h in package.handlers):
        path = Path(filename).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.close(fd)
        handler = RotatingFileHandler(path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
        handler._backend_file = True
        handler.setFormatter(formatter)
        package.addHandler(handler)


def event(logger, name, *, level=logging.INFO, **fields):
    fields.setdefault('request_id', request_id_context.get())
    values = ' '.join(f'{key}={json.dumps(value, ensure_ascii=False, default=str)}' for key, value in fields.items())
    logger.log(level, '%s %s', name, values)


def _error_message(error):
    if isinstance(error, ValidationError):
        return json.dumps(error.errors(include_input=False, include_context=False, include_url=False), ensure_ascii=False)
    if isinstance(error, httpx.HTTPStatusError):
        return f'HTTP {error.response.status_code}: {error.request.method} {error.request.url.path}'
    text = str(error)
    # Never put configured credentials in diagnostics, including exception text.
    for key, value in os.environ.items():
        if len(value) >= 8 and any(part in key.upper() for part in ('API_KEY', 'TOKEN', 'PASSWORD', 'SECRET')):
            text = text.replace(value, '[REDACTED]')
    return text[:2000]


def failure(logger, name, error, **fields):
    """Keep actual stack frames and cause types; omit locals/HTTP bodies/inputs."""
    chain, seen = [], set()
    current = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        frames = traceback.format_list(traceback.extract_tb(current.__traceback__))
        chain.append('Traceback (most recent call last):\n' + ''.join(frames)
                     + f'{type(current).__name__}: {_error_message(current)}\n')
        current = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
    event(logger, name, level=logging.ERROR, error_type=type(error).__name__, error=_error_message(error), **fields)
    logger.error('%s', '\nCaused by:\n'.join(chain))


class RequestLoggingMiddleware:
    """Measure the full response stream, not just time to response headers."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        request_id = uuid4().hex
        scope.setdefault('state', {})['request_id'] = request_id
        token = request_id_context.set(request_id)
        started = time.monotonic()
        status, sent_bytes, response_started, response_finished = 500, 0, False, False
        outcome = 'complete'
        path = scope.get('path', '')
        if path not in ('/api/create', '/api/categories', '/api/generate', '/health', '/ready', '/docs', '/openapi.json'):
            path = '<other>'
        event(LOG, 'http_request_start', method=scope.get('method'), path=path)

        async def logged_send(message):
            nonlocal status, sent_bytes, response_started, response_finished
            if message['type'] == 'http.response.start':
                response_started = True
                status = message['status']
                message = {**message, 'headers': [
                    (key, value) for key, value in message.get('headers', []) if key.lower() != b'x-request-id'
                ] + [(b'x-request-id', request_id.encode('ascii'))]}
            elif message['type'] == 'http.response.body':
                sent_bytes += len(message.get('body', b''))
                response_finished = not message.get('more_body', False)
            await send(message)

        try:
            await self.app(scope, receive, logged_send)
        except BaseException as error:
            import asyncio
            if isinstance(error, asyncio.CancelledError):
                outcome = 'cancelled'
                event(LOG, 'http_request_cancelled', method=scope.get('method'), path=path)
                raise
            if not isinstance(error, Exception):
                raise
            outcome = 'error'
            failure(LOG, 'http_request_failed', error, method=scope.get('method'), path=path, status=status)
            if not response_started:
                await JSONResponse(status_code=500, content={
                    'message': 'Внутренняя ошибка сервера.', 'code': 'internal_error',
                    'requestId': request_id,
                })(scope, receive, logged_send)
            elif not response_finished:
                # A broken NDJSON stream stays incomplete; never emit fake success.
                try:
                    await logged_send({'type': 'http.response.body', 'body': b'', 'more_body': False})
                except (OSError, RuntimeError):
                    pass
        finally:
            event(LOG, 'http_request_finish', method=scope.get('method'), path=path, status=status,
                  outcome=outcome, response_bytes=sent_bytes, elapsed_ms=int((time.monotonic() - started) * 1000))
            request_id_context.reset(token)
