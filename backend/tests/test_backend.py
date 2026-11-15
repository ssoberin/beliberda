import asyncio
import json
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.ssh_tunnel import SSHTunnel, ssh_command

SOURCE = 'Укладываем плитку и меняем смесители. Работы выполняем только по записи.'
REQUEST = {'title': 'Услуги мастера', 'description': SOURCE,
           'categories': [{'id': 103, 'name': 'Плитка'}, {'id': 104, 'name': 'Сантехника'}]}


def model_draft(category=103, **changes):
    value = {'categoryId': category, 'supported': True, 'title': 'Плиточные работы',
             'text': 'Укладываем плитку. Работаем только по записи.', 'evidence': ['Укладываем плитку', 'только по записи']}
    if category == 104:
        value.update(title='Замена смесителей', text='Меняем смесители. Работаем только по записи.',
                     evidence=['меняем смесители', 'только по записи'])
    return {**value, **changes}


def completion(drafts=None, **changes):
    return {'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({
        'drafts': drafts if drafts is not None else [model_draft(), model_draft(104)],
    }, ensure_ascii=False)}, **changes}]}


def client_for(handler, **settings):
    settings.setdefault('tokenize_before_generate', False)
    return TestClient(create_app(Settings(**settings), transport=httpx.MockTransport(handler)))


def test_frontend_contract_prompt_and_one_batch_request():
    calls = []
    def remote(request):
        calls.append(request)
        assert str(request.url) == 'http://127.0.0.1:8000/v1/chat/completions'
        assert request.headers['authorization'] == 'Bearer test-secret'
        payload = json.loads(request.content)
        assert payload['model'] == 'test-qwen'
        assert payload['response_format'] == {'type': 'json_object'}
        assert payload['temperature'] == 0
        assert 'Не выдумывай' in payload['messages'][0]['content']
        assert len(payload['messages']) == 8
        assert [m['role'] for m in payload['messages']] == ['system', 'user', 'assistant', 'user', 'assistant', 'user', 'assistant', 'user']
        user = json.loads(payload['messages'][-1]['content'])
        assert user['description'] == SOURCE
        assert [c['id'] for c in user['requested_categories']] == [103, 104]
        assert user['requested_categories'][0]['name'] == 'Плиточные работы'
        for category in user['requested_categories']:
            assert category['description'] and category['examples'] and category['boundaries']
        return httpx.Response(200, json=completion())
    with client_for(remote, qwen_model='test-qwen', qwen_api_key='test-secret') as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 200
        assert response.json()['drafts'] == [
            {'categoryId': 103, 'title': 'Плиточные работы', 'description': 'Укладываем плитку. Работаем только по записи.'},
            {'categoryId': 104, 'title': 'Замена смесителей', 'description': 'Меняем смесители. Работаем только по записи.'},
        ]
        assert response.json()['warnings'] == []
        assert len(calls) == 1


def test_evidence_accepts_only_case_and_whitespace_differences():
    source = 'Штукатурим стены, укладываем плитку.\nРаботаем в Казани.'
    request = {'description': source, 'categories': [{'id': 103}]}
    example = model_draft(evidence=['Укладываем плитку', 'Работаем  в Казани.'],
                          title='Укладка плитки', text='Укладываем плитку.')
    with client_for(lambda r: httpx.Response(200, json=completion([example]))) as client:
        response = client.post('/api/generate', json=request)
        assert response.status_code == 200, response.text
        assert response.json()['drafts'][0]['categoryId'] == 103
    for invalid in ['укладываем керамогранит', 'укладываем  плитку бесплатно', 'укладываем плитку, меняем трубы']:
        example['evidence'] = [invalid]
        with client_for(lambda r: httpx.Response(200, json=completion([example]))) as client:
            assert client.post('/api/generate', json=request).status_code == 502


@pytest.mark.parametrize('title,text', [(None, None), ('Сантехнические работы', 'Сантехнические работы. [Уточните виды работ.]')])
def test_every_selected_category_has_a_draft_even_without_source_details(title, text):
    unsupported = model_draft(104, supported=False, title=title, text=text, evidence=[])
    with client_for(lambda r: httpx.Response(200, json=completion([model_draft(), unsupported]))) as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 200
        assert [d['categoryId'] for d in response.json()['drafts']] == [103, 104]
        assert all(d['title'] and d['description'] for d in response.json()['drafts'])
        assert response.json()['warnings'][0]['categoryId'] == 104
        assert response.json()['warnings'][0]['reason'] == 'needs_details'


@pytest.mark.parametrize('update', [
    {'description': '   '}, {'description': 'x' * 12001}, {'categories': []},
    {'categories': [{'id': 999}]}, {'categories': [{'id': 103}, {'id': 103}]},
    {'unexpected': True},
])
def test_invalid_requests_do_not_call_remote(update):
    def unexpected(_): raise AssertionError('Invalid input must not call Qwen')
    with client_for(unexpected) as client:
        response = client.post('/api/generate', json={**REQUEST, **update})
        assert response.status_code == 422
        assert response.json()['message']


@pytest.mark.parametrize('drafts', [
    [model_draft(evidence=['Выдуманная цитата']), model_draft(104)],
    [model_draft(text='Укладываем плитку за 999 рублей.'), model_draft(104)],
    [model_draft(text='Гарантия на плитку.'), model_draft(104)],
    [model_draft(), model_draft()],
    [model_draft()],
    [model_draft(), model_draft(105)],
    [model_draft(title='x' * 51), model_draft(104)],
])
def test_untrusted_model_output_is_rejected(drafts):
    with client_for(lambda r: httpx.Response(200, json=completion(drafts))) as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 502
        assert response.json()['message']


@pytest.mark.parametrize('body', [
    {'choices': []}, {'unexpected': True},
    {'choices': [None]}, {'choices': [{'finish_reason': 'stop', 'message': None}]},
    {'choices': [{'finish_reason': 'length', 'message': {'content': '{}'}}]},
    {'choices': [{'finish_reason': 'stop', 'message': {'content': '```json\n{}\n```'}}]},
])
def test_bad_completion_does_not_become_success(body):
    with client_for(lambda r: httpx.Response(200, json=body)) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 502


@pytest.mark.parametrize('status,expected', [(401, 502), (429, 503), (500, 502), (503, 503)])
def test_remote_errors_are_sanitized(status, expected):
    with client_for(lambda r: httpx.Response(status, text='private remote detail: test-secret')) as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == expected
        assert 'test-secret' not in response.text
        assert 'private remote' not in response.text


def test_timeout_and_connection_failure():
    def timeout(request): raise httpx.ReadTimeout('private timeout', request=request)
    def unavailable(request): raise httpx.ConnectError('private address', request=request)
    with client_for(timeout) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 504
    with client_for(unavailable) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 502


def test_deadline_includes_waiting_for_model():
    async def slow(_):
        await asyncio.sleep(.05)
        return httpx.Response(200, json=completion())
    with client_for(slow, timeout_seconds=.01) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 504


def test_liveness_and_model_readiness_are_different():
    with client_for(lambda r: httpx.Response(200, json={'data': [{'id': 'avito-qwen'}]})) as client:
        assert client.get('/health').json() == {'status': 'ok'}
        assert client.get('/ready').status_code == 200
    with client_for(lambda r: httpx.Response(200, json={'data': [{'id': 'other-model'}]})) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/ready').status_code == 503


@pytest.mark.parametrize('mode', ['json_schema', 'none'])
def test_response_format_modes(mode):
    def remote(request):
        body = json.loads(request.content)
        if mode == 'none': assert 'response_format' not in body
        else:
            schema = body['response_format']['json_schema']['schema']
            assert schema['$defs']['ModelDraft']['properties']['categoryId']['enum'] == [103, 104]
        return httpx.Response(200, json=completion())
    with client_for(remote, response_format=mode) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 200


def test_ssh_arguments_and_configuration_do_not_execute_shell():
    settings = Settings(ssh_target='qwen-box', qwen_base_url='http://127.0.0.1:18000/v1')
    command = ssh_command(settings)
    assert command[-1] == 'qwen-box'
    assert '127.0.0.1:18000:127.0.0.1:8000' in command
    assert 'StrictHostKeyChecking=yes' in command
    assert 'BatchMode=yes' in command
    assert '-p' not in command  # Preserve the alias Port setting.
    with pytest.raises(ValueError): Settings(ssh_target='-oProxyCommand=bad')
    with pytest.raises(ValueError): Settings(ssh_target='user@host; touch /tmp/bad')
    assert 'secret' not in repr(Settings(qwen_api_key='secret'))


def test_tunnel_process_cleanup():
    async def run():
        tunnel = SSHTunnel(Settings())
        class Process:
            returncode = None
            terminated = False
            def terminate(self): self.terminated = True
            async def wait(self): self.returncode = 0
        process = Process()
        tunnel.process = process
        await tunnel.stop()
        assert process.terminated
        assert not tunnel.alive
    asyncio.run(run())


def test_startup_failure_does_not_leave_a_tunnel():
    async def run():
        settings = Settings(ssh_target='qwen-box', qwen_base_url='http://127.0.0.1:18000/v1')
        tunnel = SSHTunnel(settings)
        with patch('backend.app.ssh_tunnel.socket.socket') as probe, \
             patch('backend.app.ssh_tunnel.asyncio.create_subprocess_exec', new=AsyncMock(side_effect=FileNotFoundError)), \
             patch.object(tunnel, 'stop', new=AsyncMock()) as stop:
            with pytest.raises(FileNotFoundError): await tunnel.__aenter__()
            stop.assert_awaited_once()
    asyncio.run(run())


def test_source_category_does_not_filter_explicit_selection():
    with client_for(lambda r: httpx.Response(200, json=completion())) as client:
        response = client.post('/api/generate', json={**REQUEST, 'mcId': 103})
        assert response.status_code == 200
        assert [d['categoryId'] for d in response.json()['drafts']] == [103, 104]


def test_few_shots_are_valid_and_preserve_relevant_conditions():
    from backend.app.prompts import FEW_SHOTS
    from backend.app.schemas import ModelResponse
    for i in range(0, len(FEW_SHOTS), 2):
        request = json.loads(FEW_SHOTS[i]['content'])
        result = ModelResponse.model_validate_json(FEW_SHOTS[i + 1]['content'])
        assert [d.categoryId for d in result.drafts] == [c['id'] for c in request['requested_categories']]
        assert all(d.title and d.text for d in result.drafts)
        for draft in result.drafts:
            assert all(quote in request['description'] for quote in draft.evidence)
    first = json.loads(FEW_SHOTS[1]['content'])
    assert 'только вместе' in first['drafts'][1]['text']
    assert all(not d['supported'] for d in json.loads(FEW_SHOTS[3]['content'])['drafts'])


@pytest.mark.parametrize('count,expected_status,expected_budget', [(6500, 200, 1436), (7800, 422, None)])
def test_actual_token_budget_includes_all_few_shots(count, expected_status, expected_budget):
    calls = []
    def remote(request):
        calls.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            assert len(body['messages']) == 8
            return httpx.Response(200, json={'count': count, 'max_model_len': 8192})
        assert body['max_tokens'] == expected_budget
        return httpx.Response(200, json=completion())
    with client_for(remote, tokenize_before_generate=True) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == expected_status
    assert calls == (['/tokenize', '/v1/chat/completions'] if expected_status == 200 else ['/tokenize'])


def test_tunnel_reconnects_after_process_exit_and_cleans_up():
    async def run():
        tunnel = SSHTunnel(Settings())
        tunnel.RETRY_DELAY = .001
        connected_again = asyncio.Event()
        processes = []
        class Process:
            def __init__(self):
                self.returncode = None
                self.exited = asyncio.Event()
            async def wait(self):
                await self.exited.wait()
                return self.returncode
            def terminate(self):
                self.returncode = 0
                self.exited.set()
        async def connect():
            process = Process()
            processes.append(process)
            tunnel.process = process
            tunnel._connected = True
            if len(processes) == 2: connected_again.set()
        with patch.object(tunnel, '_connect', new=AsyncMock(side_effect=connect)) as connection:
            async with tunnel:
                assert tunnel.alive
                processes[0].returncode = 255
                processes[0].exited.set()
                await asyncio.wait_for(connected_again.wait(), timeout=1)
                assert tunnel.alive
                assert connection.await_count == 2
            assert not tunnel.alive
            assert processes[1].returncode == 0
            await asyncio.sleep(.01)
            assert connection.await_count == 2
    asyncio.run(run())


def test_tunnel_can_start_degraded_and_recover_when_network_returns():
    async def run():
        tunnel = SSHTunnel(Settings())
        tunnel.RETRY_DELAY = .001
        recovered = asyncio.Event()
        class Process:
            returncode = None
            def __init__(self): self.exited = asyncio.Event()
            async def wait(self): await self.exited.wait(); return self.returncode
            def terminate(self): self.returncode = 0; self.exited.set()
        attempts = 0
        async def connect():
            nonlocal attempts
            attempts += 1
            if attempts == 1: raise RuntimeError('Сервер не отвечает на SSH-порту (таймаут).')
            tunnel.process = Process()
            tunnel._connected = True
            tunnel.last_error = None
            recovered.set()
        with patch.object(tunnel, '_connect', new=AsyncMock(side_effect=connect)):
            async with tunnel:
                assert not tunnel.alive
                assert 'таймаут' in tunnel.last_error
                await asyncio.wait_for(recovered.wait(), timeout=1)
                assert tunnel.alive
                assert tunnel.last_error is None
            assert not tunnel.alive
    asyncio.run(run())


def test_broken_tunnel_is_reported_before_calling_qwen():
    class OfflineTunnel:
        alive = False
        last_error = 'Сервер не отвечает на SSH-порту (таймаут).'
        async def __aenter__(self): return self
        async def __aexit__(self, *_): pass
    def unexpected(_): raise AssertionError('A broken tunnel must not call Qwen')
    settings = Settings(ssh_target='qwen-box', qwen_base_url='http://127.0.0.1:18000/v1')
    with patch('backend.app.main.SSHTunnel', return_value=OfflineTunnel()):
        with TestClient(create_app(settings, transport=httpx.MockTransport(unexpected))) as client:
            assert client.get('/health').status_code == 200
            readiness = client.get('/ready')
            assert readiness.status_code == 503
            assert readiness.json()['component'] == 'ssh'
            assert 'таймаут' in readiness.json()['detail']
            response = client.post('/api/generate', json=REQUEST)
            assert response.status_code == 503
            assert 'переподключается' in response.json()['message']


def test_retry_keeps_last_diagnostic_until_successful_connection():
    async def run():
        tunnel = SSHTunnel(Settings())
        tunnel.last_error = 'Сервер не отвечает на SSH-порту (таймаут).'
        with patch('backend.app.ssh_tunnel.socket.socket') as probe:
            probe.return_value.__enter__.return_value.bind.side_effect = OSError('busy')
            with pytest.raises(RuntimeError): await tunnel._connect()
        assert 'таймаут' in tunnel.last_error
    asyncio.run(run())
