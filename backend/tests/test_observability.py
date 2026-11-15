import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import logging
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.errors import BackendError
from backend.app.inference import InferenceQueue
from backend.app.main import create_app
from backend.app.observability import event, request_id_context


@contextmanager
def api():
    with patch('backend.app.main.load_classifier', return_value=None):
        with TestClient(create_app(Settings(), transport=httpx.MockTransport(lambda _: httpx.Response(503)))) as client:
            yield client


@contextmanager
def capture(caplog):
    logger = logging.getLogger('backend')
    logger.addHandler(caplog.handler)
    caplog.set_level(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(caplog.handler)


def test_clean_backend_error_and_validation_keep_correlation_in_separate_field(caplog):
    with capture(caplog), api() as client:
        @client.app.get('/expected-error')
        async def expected():
            raise BackendError(503, 'Сервер модели недоступен.')
        result = client.get('/expected-error', headers={'X-Request-ID': 'untrusted-id'})
        assert result.status_code == 503
        assert result.json()['message'] == 'Сервер модели недоступен.'
        assert result.json()['code'] == 'service_unavailable'
        assert result.json()['requestId'] == result.headers['X-Request-ID'] != 'untrusted-id'
        assert result.json()['requestId'] in caplog.text
        assert 'api_error' in caplog.text and 'Traceback' in caplog.text
        bad = client.post('/api/categories', json={'description': ' '})
        assert bad.status_code == 422
        assert bad.json()['requestId'] == bad.headers['X-Request-ID']
        assert bad.json()['code'] == 'validation_error'
        assert 'api_validation_failed' in caplog.text


def test_unhandled_exception_returns_json_and_logs_stack(caplog):
    with capture(caplog), api() as client:
        @client.app.get('/broken')
        async def broken():
            raise RuntimeError('model process is broken')
        result = client.get('/broken')
        assert result.status_code == 500
        assert result.headers['content-type'].startswith('application/json')
        assert result.json()['code'] == 'internal_error'
        assert result.json()['requestId'] == result.headers['X-Request-ID']
        assert 'RuntimeError: model process is broken' in caplog.text
        assert 'http_request_failed' in caplog.text
        assert 'http_request_finish' in caplog.text


def test_request_id_is_propagated_into_resident_inference_thread(caplog):
    async def run():
        logger = logging.getLogger('backend.test')
        def predict():
            event(logger, 'test_prediction')
            return request_id_context.get()
        with ThreadPoolExecutor(max_workers=1) as executor:
            async with InferenceQueue(executor) as worker:
                async def request(value):
                    token = request_id_context.set(value)
                    try:
                        return await worker.run(predict)
                    finally:
                        request_id_context.reset(token)
                return await asyncio.gather(request('request-one'), request('request-two'))
    with capture(caplog):
        assert asyncio.run(run()) == ['request-one', 'request-two']
    assert 'request-one' in caplog.text and 'request-two' in caplog.text
    assert 'inference_enqueued' in caplog.text and 'inference_complete' in caplog.text


def test_stream_errors_contain_request_id_without_losing_final_event(caplog):
    with capture(caplog), api() as client:
        result = client.post('/api/create', json={'description': 'Устанавливаем сантехнику.'}, headers={'Accept': 'application/x-ndjson'})
        events = [json.loads(line) for line in result.text.splitlines()]
        failed = next(event for event in events if event['type'] == 'item_done')
        assert failed['error']['code'] == 'service_unavailable'
        assert failed['error']['requestId'] == result.headers['X-Request-ID']
        assert events[-1]['type'] == 'done' and events[-1]['failed'] == 1
        assert 'creation_item_failed' in caplog.text and 'batch_complete' in caplog.text


def test_routing_errors_also_use_clean_json():
    with api() as client:
        missing = client.get('/does-not-exist')
        assert missing.status_code == 404
        assert missing.json()['code'] == 'not_found'
        assert missing.json()['requestId'] == missing.headers['X-Request-ID']
        wrong_method = client.get('/api/categories')
        assert wrong_method.status_code == 405
        assert wrong_method.json()['code'] == 'method_not_allowed'
        assert 'POST' in wrong_method.headers['Allow']
