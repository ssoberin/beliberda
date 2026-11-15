from contextlib import AsyncExitStack, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
import httpx

from .classifier import classify, load_classifier
from .config import Settings
from .creation import MAX_BODY_BYTES, creation_events, parse_creation
from .errors import BackendError
from .qwen import QwenClient
from .schemas import ClassificationRequest, ClassificationResponse, GenerateRequest, GenerateResponse
from .ssh_tunnel import SSHTunnel

# Uvicorn configures its own loggers, not the application package. Keep
# diagnostics on a dedicated handler without enabling INFO logs for httpx or
# other libraries that might expose URLs or headers.
package_logger = logging.getLogger('backend')
if not package_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s %(name)s %(message)s'))
    package_logger.addHandler(handler)
package_logger.setLevel(logging.INFO)
package_logger.propagate = False
logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            app.state.classifier = await run_in_threadpool(load_classifier, settings.classifier_model_path)
            tunnel = await stack.enter_async_context(SSHTunnel(settings)) if settings.ssh_target else None
            headers = {'Authorization': f'Bearer {settings.qwen_api_key}'} if settings.qwen_api_key else {}
            client = await stack.enter_async_context(httpx.AsyncClient(
                base_url=settings.qwen_base_url.rstrip('/') + '/',
                headers=headers, timeout=httpx.Timeout(settings.timeout_seconds, connect=5),
                transport=transport, trust_env=False, follow_redirects=False,
            ))
            app.state.qwen = QwenClient(client, settings)
            app.state.tunnel = tunnel
            app.state.creation_workers = settings.max_concurrent_requests
            app.state.creation_slots = asyncio.Semaphore(settings.max_concurrent_requests)
            with ThreadPoolExecutor(max_workers=settings.max_concurrent_requests, thread_name_prefix='creation') as threads:
                app.state.creation_threads = threads
                yield

    app = FastAPI(title='Avito Qwen Draft Backend', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'], allow_credentials=False)

    @app.middleware('http')
    async def request_id(request: Request, call_next):
        # Generated server-side: never trust a client-supplied correlation ID.
        request.state.request_id = uuid4().hex
        started = time.monotonic()
        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error('api_unhandled request_id=%s path=%s error_type=%s',
                         request.state.request_id,
                         request.url.path if request.url.path in ('/api/create', '/api/generate', '/api/categories') else '<other>',
                         type(exc).__name__)
            raise
        response.headers['X-Request-ID'] = request.state.request_id
        if request.url.path in ('/api/create', '/api/generate', '/api/categories'):
            logger.info('api_complete request_id=%s path=%s status=%s elapsed_ms=%d',
                        request.state.request_id, request.url.path, response.status_code,
                        int((time.monotonic() - started) * 1000))
        return response

    @app.exception_handler(BackendError)
    async def backend_error(request, error):
        request_id = request.state.request_id
        return JSONResponse(status_code=error.status_code, content={
            'message': f'{error.message} (код запроса: {request_id})',
            'requestId': request_id,
        })

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, error):
        return JSONResponse(status_code=422, content={
            'message': 'Проверьте описание и выбранные категории.',
            'errors': [{'field': '.'.join(map(str, item['loc'])), 'message': item['msg']} for item in error.errors()],
        })

    @app.get('/health')
    async def health():
        return {'status': 'ok'}

    @app.get('/ready')
    async def ready(request: Request):
        category_generation = request.app.state.classifier is not None
        tunnel = request.app.state.tunnel
        if tunnel is not None and not tunnel.alive:
            return JSONResponse(status_code=503, content={
                'status': 'not_ready', 'component': 'ssh',
                'categoryGeneration': category_generation,
                'message': 'SSH-туннель отключён. Backend автоматически восстанавливает соединение.',
                'detail': tunnel.last_error,
            })
        if not await request.app.state.qwen.ready():
            return JSONResponse(status_code=503, content={'status': 'not_ready', 'component': 'qwen', 'categoryGeneration': category_generation, 'message': 'Соединение с API Qwen недоступно либо нужная модель не загружена.'})
        return {'status': 'ready', 'model': settings.qwen_model, 'categoryGeneration': category_generation}

    @app.post('/api/create')
    async def create(request: Request, generateText: bool | None = None):
        chunks = []
        size = 0
        async for chunk in request.stream():
            size += len(chunk)
            if size > MAX_BODY_BYTES:
                raise BackendError(413, 'Максимальный размер запроса — 5 МБ.')
            chunks.append(chunk)
        items, batch, generate_text = await run_in_threadpool(
            parse_creation, b''.join(chunks), request.headers.get('content-type', ''), generateText,
        )
        events = creation_events(request.app, items, generate_text, request.state.request_id)
        if 'application/x-ndjson' in request.headers.get('accept', ''):
            async def stream():
                try:
                    async for event in events:
                        yield json.dumps(event, ensure_ascii=False) + '\n'
                finally:
                    await events.aclose()
            return StreamingResponse(stream(), media_type='application/x-ndjson', headers={
                'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no',
            })
        results = []
        async for event in events:
            if event['type'] == 'item_done':
                results.append({k: v for k, v in event.items() if k not in ('type', 'completed', 'total')})
        results.sort(key=lambda item: item['index'])
        if batch:
            return {'results': results}
        result = {k: v for k, v in results[0].items() if k not in ('index', 'testcase')}
        if 'error' in result:
            return JSONResponse(status_code=503, content={**result, 'message': result['error']['message']})
        return result

    @app.post('/api/categories', response_model=ClassificationResponse)
    async def categories(payload: ClassificationRequest, request: Request):
        model = request.app.state.classifier
        if model is None:
            raise BackendError(503, 'TF-IDF не настроена: укажите проверенный CLASSIFIER_MODEL_PATH.')
        return {'categories': await run_in_threadpool(classify, model, payload.description, payload.mcId)}

    @app.post('/api/generate', response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request):
        tunnel = request.app.state.tunnel
        if tunnel is not None and not tunnel.alive:
            raise BackendError(503, 'SSH-туннель отключён и автоматически переподключается. Повторите генерацию после восстановления подключения.')
        return await request.app.state.qwen.generate(payload, request_id=request.state.request_id)

    return app


app = create_app()
