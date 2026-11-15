from contextlib import AsyncExitStack, asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import asyncio
import json
import logging
import time

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException
import httpx

from .classifier import classify, load_classifier
from .config import Settings
from .creation import creation_events
from .errors import BackendError
from .inference import InferenceQueue
from .observability import configure_logging, event, failure, RequestLoggingMiddleware
from .qwen import QwenClient
from .schemas import ClassificationRequest, ClassificationResponse, GenerateRequest, GenerateResponse
from .ssh_tunnel import SSHTunnel
from .uploads import read_creation_request

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None):
    settings = settings or Settings.from_env()
    configure_logging()

    @asynccontextmanager
    async def lifespan(app):
        started = time.monotonic()
        event(logger, 'startup_start', classifier='C_syntax', qwen_model=settings.qwen_model,
              ssh_enabled=bool(settings.ssh_target), max_concurrent=settings.max_concurrent_requests)
        async with AsyncExitStack() as stack:
            executor = stack.enter_context(ThreadPoolExecutor(max_workers=1, thread_name_prefix='classifier'))
            loop = asyncio.get_running_loop()
            try:
                app.state.classifier = await loop.run_in_executor(executor, load_classifier, settings.classifier_model_path)
            except Exception as error:
                failure(logger, 'startup_model_failed', error)
                raise
            pipeline = app.state.classifier.get('pipeline') if isinstance(app.state.classifier, dict) else None
            if pipeline is not None and callable(getattr(pipeline, 'warmup', None)):
                event(logger, 'model_warmup_start', model='C_syntax')
                warmup_started = time.monotonic()
                try:
                    await loop.run_in_executor(executor, pipeline.warmup)
                except Exception as error:
                    failure(logger, 'model_warmup_failed', error)
                    raise
                event(logger, 'model_warmup_complete', elapsed_ms=int((time.monotonic()-warmup_started)*1000))
            app.state.inference = await stack.enter_async_context(InferenceQueue(executor))
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
            event(logger, 'startup_complete', category_generation=app.state.classifier is not None,
                  elapsed_ms=int((time.monotonic()-started)*1000))
            try:
                yield
            finally:
                event(logger, 'shutdown_start')
        event(logger, 'shutdown_complete')

    app = FastAPI(title='Avito Qwen Draft Backend', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'], allow_credentials=False)

    app.add_middleware(RequestLoggingMiddleware)

    @app.exception_handler(BackendError)
    async def backend_error(request, error):
        request_id = request.state.request_id
        failure(logger, 'api_error', error, status=error.status_code, code=error.code)
        return JSONResponse(status_code=error.status_code, content={
            'message': error.message,
            'code': error.code,
            'requestId': request_id,
        })

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        event(logger, 'api_validation_failed', level=logging.WARNING, status=422,
              issues=[{'location': item['loc'][0], 'type': item['type']} for item in error.errors()])
        return JSONResponse(status_code=422, content={
            'message': 'Проверьте описание и выбранные категории.',
            'code': 'validation_error', 'requestId': request.state.request_id,
            'errors': [{'field': '.'.join(map(str, item['loc'])), 'message': item['msg']} for item in error.errors()],
        })

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        message = error.detail if isinstance(error.detail, str) else 'Ошибка HTTP-запроса.'
        code = {401: 'unauthorized', 403: 'forbidden', 404: 'not_found', 405: 'method_not_allowed'}.get(error.status_code, 'http_error')
        event(logger, 'http_error', level=logging.WARNING, status=error.status_code, code=code)
        return JSONResponse(status_code=error.status_code, headers=error.headers, content={
            'message': message, 'code': code, 'requestId': request.state.request_id,
        })

    @app.exception_handler(Exception)
    async def unexpected_error(request, error):
        return JSONResponse(status_code=500, content={
            'message': 'Внутренняя ошибка сервера.', 'code': 'internal_error',
            'requestId': request.state.request_id,
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
        items, batch, generate_text, size = await read_creation_request(request, generateText)
        event(logger, 'creation_input_parsed', content_type=request.headers.get('content-type', '').partition(';')[0],
              bytes=size, items=len(items), batch=batch, generate_text=generate_text)
        events = creation_events(request.app, items, generate_text, request.state.request_id, batch=batch)
        if 'application/x-ndjson' in request.headers.get('accept', ''):
            async def stream():
                try:
                    async for payload_event in events:
                        event(logger, 'stream_event', event_type=payload_event['type'], index=payload_event.get('index'))
                        yield json.dumps(payload_event, ensure_ascii=False) + '\n'
                finally:
                    await events.aclose()
            return StreamingResponse(stream(), media_type='application/x-ndjson', headers={
                'Cache-Control': 'no-cache, no-transform', 'X-Accel-Buffering': 'no',
            })
        results = []
        async for payload_event in events:
            if payload_event['type'] == 'item_done':
                results.append({k: v for k, v in payload_event.items() if k not in ('type', 'completed', 'total')})
        results.sort(key=lambda item: item['index'])
        if batch:
            return {'results': results}
        result = {k: v for k, v in results[0].items() if k not in ('index', 'testcase')}
        if 'error' in result:
            return JSONResponse(status_code=503, content={**result, 'message': result['error']['message'], 'requestId': request.state.request_id})
        return result

    @app.post('/api/categories', response_model=ClassificationResponse)
    async def categories(payload: ClassificationRequest, request: Request):
        model = request.app.state.classifier
        if model is None:
            raise BackendError(503, 'Классификатор C_syntax недоступен.')
        return {'categories': await request.app.state.inference.run(classify, model, payload.description, payload.mcId)}

    @app.post('/api/generate', response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request):
        tunnel = request.app.state.tunnel
        if tunnel is not None and not tunnel.alive:
            raise BackendError(503, 'SSH-туннель отключён и автоматически переподключается. Повторите генерацию после восстановления подключения.')
        return await request.app.state.qwen.generate(payload, request_id=request.state.request_id)

    return app


app = create_app()
