from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

from .config import Settings
from .errors import BackendError
from .qwen import QwenClient
from .schemas import GenerateRequest, GenerateResponse
from .ssh_tunnel import SSHTunnel


def create_app(settings: Settings | None = None, *, transport: httpx.AsyncBaseTransport | None = None):
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app):
        async with AsyncExitStack() as stack:
            tunnel = await stack.enter_async_context(SSHTunnel(settings)) if settings.ssh_target else None
            headers = {'Authorization': f'Bearer {settings.qwen_api_key}'} if settings.qwen_api_key else {}
            client = await stack.enter_async_context(httpx.AsyncClient(
                base_url=settings.qwen_base_url.rstrip('/') + '/',
                headers=headers, timeout=httpx.Timeout(settings.timeout_seconds, connect=5),
                transport=transport, trust_env=False, follow_redirects=False,
            ))
            app.state.qwen = QwenClient(client, settings)
            app.state.tunnel = tunnel
            yield

    app = FastAPI(title='Avito Qwen Draft Backend', version='0.1.0', lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins),
                       allow_methods=['GET', 'POST'], allow_headers=['Content-Type'], allow_credentials=False)

    @app.exception_handler(BackendError)
    async def backend_error(_request, error):
        return JSONResponse(status_code=error.status_code, content={'message': error.message})

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
        tunnel = request.app.state.tunnel
        if tunnel is not None and not tunnel.alive:
            return JSONResponse(status_code=503, content={
                'status': 'not_ready', 'component': 'ssh',
                'message': 'SSH-туннель отключён. Backend автоматически восстанавливает соединение.',
                'detail': tunnel.last_error,
            })
        if not await request.app.state.qwen.ready():
            return JSONResponse(status_code=503, content={'status': 'not_ready', 'component': 'qwen', 'message': 'Соединение с API Qwen недоступно либо нужная модель не загружена.'})
        return {'status': 'ready', 'model': settings.qwen_model}

    @app.post('/api/generate', response_model=GenerateResponse)
    async def generate(payload: GenerateRequest, request: Request):
        tunnel = request.app.state.tunnel
        if tunnel is not None and not tunnel.alive:
            raise BackendError(503, 'SSH-туннель отключён и автоматически переподключается. Повторите генерацию после восстановления подключения.')
        return await request.app.state.qwen.generate(payload)

    return app


app = create_app()
