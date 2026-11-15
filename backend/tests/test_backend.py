import asyncio
import json
import logging
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.main import create_app
from backend.app.prompts import CONDITION_EXAMPLES, FEW_SHOTS
from backend.app.ssh_tunnel import SSHTunnel, ssh_command
from backend.app.titles import repair_title, valid_title

SOURCE = 'Укладываем плитку и меняем смесители. Работы выполняем только по записи.'
REQUEST = {'title': 'Услуги мастера', 'description': SOURCE,
           'categories': [{'id': 103, 'name': 'Плитка'}, {'id': 104, 'name': 'Сантехника'}]}


def model_draft(category=103, **changes):
    value = {'categoryId': category, 'title': 'Плиточные работы',
             'text': 'Укладываем плитку. Работаем только по записи.'}
    if category == 104:
        value.update(title='Замена смесителей', text='Меняем смесители. Работаем только по записи.')
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
        assert payload['response_format']['type'] == 'json_schema'
        assert payload['response_format']['json_schema']['strict'] is True
        assert payload['temperature'] == 0
        assert 'Не добавляй неподтверждённые операции' in payload['messages'][0]['content']
        assert 'Не оценивай supported' in payload['messages'][0]['content']
        assert len(payload['messages']) == len(FEW_SHOTS) + 2
        assert [m['role'] for m in payload['messages']] == ['system'] + ['user', 'assistant'] * (len(FEW_SHOTS) // 2) + ['user']
        assert 'верни ровно 2 элементов drafts' in payload['messages'][0]['content']
        user = json.loads(payload['messages'][-1]['content'])
        assert user['description'] == SOURCE
        assert [c['id'] for c in user['requested_categories']] == [103, 104]
        assert user['requested_categories'][0] == {'id': 103, 'name': 'Плиточные работы'}
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


def test_diagnostics_correlate_json_schema_error_without_logging_private_text(caplog):
    private = 'PRIVATE_AD_TEXT_987654'
    bad = {'choices': [{'finish_reason': 'stop', 'message': {
        'content': json.dumps({'drafts': [
            {**model_draft(), 'title': private * 4}, model_draft(104),
        ]})}}]}
    package_logger = logging.getLogger('backend')
    package_logger.addHandler(caplog.handler)
    try:
        with client_for(lambda r: httpx.Response(200, json=bad)) as client:
            response = client.post('/api/generate', json={**REQUEST, 'description': SOURCE + private})
    finally:
        package_logger.removeHandler(caplog.handler)
    assert response.status_code == 200
    request_id = response.headers['X-Request-ID']
    assert len(response.json()['drafts'][0]['title']) <= 50
    assert response.json()['drafts'][0]['description'] == model_draft()['text']
    assert response.json()['drafts'][1]['title'] == model_draft(104)['title']
    assert any('qwen_title_repaired' in record.message
               and request_id in record.message for record in caplog.records)
    assert private not in caplog.text
    assert SOURCE not in caplog.text


def test_diagnostics_for_invalid_json_do_not_log_model_output(caplog):
    private = 'PRIVATE_MODEL_OUTPUT_123456'
    package_logger = logging.getLogger('backend')
    package_logger.addHandler(caplog.handler)
    try:
        with client_for(lambda r: httpx.Response(200, json={
                'choices': [{'finish_reason': 'stop', 'message': {'content': private + '{'}}],
        })) as client:
            response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 502
        assert 'stage=model_schema' in caplog.text
    finally:
        package_logger.removeHandler(caplog.handler)
    assert private not in caplog.text


def test_draft_is_not_blocked_by_obsolete_evidence_or_supported_fields():
    # Old model servers may emit incidental metadata. It is not a decision
    # signal, and the backend must not replace selected drafts with placeholders.
    outdated = model_draft(supported=False, evidence=['несовпадающая цитата'])
    with client_for(lambda r: httpx.Response(200, json=completion([outdated, model_draft(104)]))) as client:
        response = client.post('/api/generate', json=REQUEST)
    assert response.status_code == 200
    assert [d['categoryId'] for d in response.json()['drafts']] == [103, 104]
    assert response.json()['warnings'] == []


def test_generic_service_gets_a_real_draft_without_invented_operations():
    generic = model_draft(104, title='Сантехнические работы',
                          text='Выполняем сантехнические работы в составе комплексного ремонта.')
    with client_for(lambda r: httpx.Response(200, json=completion([model_draft(), generic]))) as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 200
        assert [d['categoryId'] for d in response.json()['drafts']] == [103, 104]
        assert response.json()['drafts'][1]['description'] == generic['text']
        assert response.json()['warnings'] == []


@pytest.mark.parametrize('update', [
    {'description': '   '}, {'description': 'x' * 100001}, {'categories': []},
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
    [model_draft(), model_draft()],
    [model_draft()],
    [model_draft(), model_draft(105)],
])
def test_untrusted_model_output_is_rejected(drafts):
    with client_for(lambda r: httpx.Response(200, json=completion(drafts))) as client:
        response = client.post('/api/generate', json=REQUEST)
        assert response.status_code == 502
        assert response.json()['message']


@pytest.mark.parametrize('invalid', [model_draft(title=' '), model_draft(text=' '), model_draft(text=None)])
def test_empty_or_non_text_drafts_are_structurally_invalid(invalid):
    with client_for(lambda r: httpx.Response(200, json=completion([invalid, model_draft(104)]))) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 502


def test_oversized_title_preserves_other_drafts_and_full_text():
    long_title = 'Реконструкция деревянных домов и замена старых венцов под ключ'
    valid = model_draft(104)
    with client_for(lambda r: httpx.Response(200, json=completion([
            model_draft(title=long_title), valid,
    ]))) as client:
        response = client.post('/api/generate', json=REQUEST)
    assert response.status_code == 200
    drafts = response.json()['drafts']
    assert drafts[0]['title'] == 'Плиточные работы'
    assert drafts[0]['description'] == model_draft()['text']
    assert drafts[1] == {'categoryId': 104, 'title': valid['title'], 'description': valid['text']}
    assert response.json()['warnings'] == []


@pytest.mark.parametrize('title,expected', [
    ('Замена инженерных сетей (водоснабжение,', 'Замена инженерных сетей'),
    ('Укладка плитки,', 'Плиточные работы'),
    ('Плиточные работы и', 'Плиточные работы'),
    ('Плиточные работы «под ключ', 'Плиточные работы'),
    ('Плиточные работы (санузел)', 'Плиточные работы (санузел)'),
    ('Монтаж плитки и сантехники в санузле под ключ с договором', 'Плиточные работы'),
])
def test_title_repair_avoids_dangling_fragments(title, expected):
    assert repair_title(title, 103) == expected
    assert valid_title(expected)


def test_title_repair_does_not_rewrite_description_or_category():
    first = model_draft(title='Замена инженерных сетей (водоснабжение,')
    with client_for(lambda r: httpx.Response(200, json=completion([first, model_draft(104)]))) as client:
        response = client.post('/api/generate', json=REQUEST)
    assert response.status_code == 200
    assert response.json()['drafts'][0] == {
        'categoryId': 103, 'title': 'Замена инженерных сетей', 'description': first['text'],
    }


def test_few_shot_separates_electricity_and_painting():
    from backend.app.prompts import FEW_SHOTS
    user = json.loads(FEW_SHOTS[-2]['content'])
    drafts = json.loads(FEW_SHOTS[-1]['content'])['drafts']
    assert [c['id'] for c in user['requested_categories']] == [105, 106]
    assert 'покраск' not in drafts[0]['text'].lower()
    assert 'розетк' not in drafts[1]['text'].lower()


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


@pytest.mark.parametrize('mode', ['json_schema', 'json_object', 'none'])
def test_response_format_modes(mode):
    def remote(request):
        body = json.loads(request.content)
        if mode == 'none': assert 'response_format' not in body
        elif mode == 'json_object': assert body['response_format'] == {'type': 'json_object'}
        else:
            schema = body['response_format']['json_schema']['schema']
            assert schema['$defs']['ModelDraft']['properties']['categoryId']['enum'] == [103, 104]
        return httpx.Response(200, json=completion())
    with client_for(remote, response_format=mode) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 200


@pytest.mark.parametrize('count', range(1, 7))
def test_strict_schema_matches_requested_count_and_ids(count):
    ids = [104, 105, 103, 106, 102, 101][:count]
    def remote(request):
        body = json.loads(request.content)
        contract = body['response_format']['json_schema']
        assert contract['strict'] is True
        schema = contract['schema']
        assert schema['required'] == ['drafts']
        assert schema['additionalProperties'] is False
        array = schema['properties']['drafts']
        assert array['minItems'] == array['maxItems'] == count
        draft = schema['$defs']['ModelDraft']
        assert draft['additionalProperties'] is False
        assert set(draft['required']) == {'categoryId', 'title', 'text'}
        assert draft['properties']['categoryId'] == {'type': 'integer', 'enum': ids}
        assert draft['properties']['title']['maxLength'] == 50
        assert draft['properties']['text']['maxLength'] == 2000
        # Model order may differ; the backend restores the requested order.
        return httpx.Response(200, json=completion([model_draft(i) for i in reversed(ids)]))
    with client_for(remote) as client:
        result = client.post('/api/generate', json={**REQUEST, 'categories': [{'id':i} for i in ids]})
        assert result.status_code == 200
        assert [d['categoryId'] for d in result.json()['drafts']] == ids


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
    from backend.app.schemas import ModelResponse
    counts = set()
    for i in range(0, len(FEW_SHOTS), 2):
        request = json.loads(FEW_SHOTS[i]['content'])
        count = len(request['requested_categories'])
        counts.add(count)
        if count == 0:
            assert json.loads(FEW_SHOTS[i + 1]['content']) == {'drafts': []}
            continue
        result = ModelResponse.model_validate_json(FEW_SHOTS[i + 1]['content'])
        assert [d.categoryId for d in result.drafts] == [c['id'] for c in request['requested_categories']]
        assert all(d.title and d.text for d in result.drafts)
        assert all('Уточните' not in d.text for d in result.drafts)
        assert all(set(d) == {'categoryId', 'title', 'text'} for d in json.loads(FEW_SHOTS[i + 1]['content'])['drafts'])
        assert all(set(c) == {'id', 'name'} for c in request['requested_categories'])
        assert 'Казан' not in FEW_SHOTS[i]['content']
    assert counts == set(range(7))
    first = json.loads(CONDITION_EXAMPLES[1]['content'])
    assert 'только вместе' in first['drafts'][1]['text']
    second = json.loads(CONDITION_EXAMPLES[3]['content'])
    assert all('в составе комплексного ремонта' in d['text'] for d in second['drafts'])


@pytest.mark.parametrize('count,expected_status,expected_budget', [(6500, 200, 1436), (7800, 422, None)])
def test_actual_token_budget_includes_all_few_shots(count, expected_status, expected_budget):
    calls = []
    def remote(request):
        calls.append(request.url.path)
        body = json.loads(request.content)
        if request.url.path == '/tokenize':
            assert len(body['messages']) == len(FEW_SHOTS) + 2
            return httpx.Response(200, json={'count': count, 'max_model_len': 8192})
        assert body['max_tokens'] == expected_budget
        return httpx.Response(200, json=completion())
    with client_for(remote, tokenize_before_generate=True) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == expected_status
    assert calls == (['/tokenize', '/v1/chat/completions'] if expected_status == 200 else ['/tokenize'])


@pytest.mark.parametrize('prefix', ['', '/proxy'])
@pytest.mark.parametrize('token_count,status', [(6500, 200), (7800, 422)])
def test_llama_cpp_budget_uses_rendered_chat_and_actual_slot_context(prefix, token_count, status):
    calls = []
    def remote(request):
        assert request.url.host == '127.0.0.1'
        assert request.headers['authorization'] == 'Bearer test-secret'
        path = request.url.path.removeprefix(prefix)
        calls.append(path)
        if path == '/props':
            return httpx.Response(200, json={'default_generation_settings': {'n_ctx':8192}})
        body = json.loads(request.content)
        if path == '/apply-template':
            assert len(body['messages']) == len(FEW_SHOTS) + 2
            assert body['add_generation_prompt'] is True
            return httpx.Response(200, json={'prompt':'rendered-chat-with-examples'})
        if path == '/tokenize':
            if 'messages' in body:
                return httpx.Response(200, json={'tokens':[]})
            assert body == {'content':'rendered-chat-with-examples', 'add_special':True, 'parse_special':True}
            return httpx.Response(200, json={'tokens':[1] * token_count})
        assert path == '/v1/chat/completions'
        assert body['max_tokens'] == 1436
        return httpx.Response(200, json=completion())
    with client_for(remote, qwen_base_url='http://127.0.0.1:8000' + prefix + '/v1',
                    qwen_api_key='test-secret', tokenize_before_generate=True) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == status
    expected = ['/tokenize', '/apply-template', '/tokenize', '/props']
    assert calls == expected + (['/v1/chat/completions'] if status == 200 else [])


@pytest.mark.parametrize('tokens,limit', [([],8192), ([True],8192), ([1],None), ([1],0), ([1],True)])
def test_invalid_llama_cpp_budget_never_calls_generation(tokens, limit):
    def remote(request):
        path = request.url.path
        if path == '/apply-template':
            return httpx.Response(200, json={'prompt':'rendered'})
        if path == '/props':
            return httpx.Response(200, json={'default_generation_settings':{'n_ctx':limit}})
        if path == '/tokenize':
            return httpx.Response(200, json={'tokens':[] if 'messages' in json.loads(request.content) else tokens})
        raise AssertionError('Invalid token budget must not call generation')
    with client_for(remote, tokenize_before_generate=True) as client:
        assert client.post('/api/generate', json=REQUEST).status_code == 502


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
