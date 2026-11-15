import asyncio
from contextlib import contextmanager
import json
import threading
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.creation import CreateItem, creation_events
from backend.app.main import create_app
from backend.app.schemas import Draft, GenerateResponse

SOURCE = 'Укладываем плитку и меняем смесители.'


@contextmanager
def api(categories=None, generate=None):
    with patch('backend.app.main.load_classifier', return_value=object()), \
         patch('backend.app.creation.classify', return_value=categories if categories is not None else [{'id': 104, 'name': 'Сантехника'}]):
        with TestClient(create_app(Settings(tokenize_before_generate=False), transport=httpx.MockTransport(lambda _: httpx.Response(503)))) as client:
            if generate:
                client.app.state.qwen.generate = generate
            yield client


def response(*categories):
    return GenerateResponse(drafts=[Draft(categoryId=category, title='Услуги', description=SOURCE)
                                    for category in categories])


async def generate(payload, **_):
    return response(*(category.id for category in payload.categories))


def test_single_and_batch_json_contract_and_no_generation():
    with api(generate=generate) as client:
        assert client.post('/api/create', json={'description': SOURCE, 'mcId': 101}).json() == {
            'drafts': [{'mcId': 104, 'title': 'Услуги', 'text': SOURCE}],
        }
        result = client.post('/api/create', json={'items': [
            {'description': SOURCE, 'testcase': 'first'}, {'description': SOURCE, 'mcId': 103},
        ], 'generateText': False}).json()['results']
        assert [item['index'] for item in result] == [0, 1]
        assert result[0]['testcase'] == 'first'
        assert result[1]['drafts'] == [{'mcId': 104, 'title': None, 'text': None}]
        assert client.post('/api/create', json=[{'description': SOURCE}], params={'generateText': 'false'}).json()['results'][0]['drafts'][0]['text'] is None


def test_no_supported_categories_means_empty_drafts_without_qwen():
    with api(categories=[]) as client:
        assert client.post('/api/create', json={'description': 'Собираю шкафы.'}).json() == {'drafts': []}


def test_csv_bom_quotes_newlines_and_testcase_roundtrip():
    body = '\ufefftestcase,description,mcId\r\ncase-001,"Мастер ""Сантехник"".\nМеняем трубы, сифоны.",101\r\ncase-002,Плитка,103\r\n'
    with api() as client:
        result = client.post('/api/create?generateText=false', content=body.encode('utf-8'), headers={'Content-Type': 'text/csv; charset=utf-8'}).json()['results']
        assert [r['testcase'] for r in result] == ['case-001', 'case-002']
        assert all(r['drafts'] == [{'mcId': 104, 'title': None, 'text': None}] for r in result)


@pytest.mark.parametrize('body,kind,status', [
    ('{', 'application/json', 422), ('[]', 'application/json', 422),
    ('{"description":" "}', 'application/json', 422),
    ('{"description":"Услуги","mcId":999}', 'application/json', 422),
    ('{"items":[{"description":"Услуги"}],"generateText":"no"}', 'application/json', 422),
    ('{"description":"Услуги","categories":[{"id":104},{"id":104}]}', 'application/json', 422),
    ('{"description":"Услуги","unknown":true}', 'application/json', 422),
    ('testcase\ncase-001', 'text/csv', 422),
    ('testcase,description\ncase-001,"незакрыто', 'text/csv', 422),
    ('testcase,description\ncase-001,текст,лишнее', 'text/csv', 422),
    ('testcase,description\ncase-001', 'text/csv', 422),
    ('text', 'text/plain', 415),
    (json.dumps([{'description': 'x'}] * 101), 'application/json', 422),
    ('x' * (5 * 1024 * 1024 + 1), 'text/csv', 413),
])
def test_validation_finishes_before_stream_headers(body, kind, status):
    with api() as client:
        result = client.post('/api/create', content=body.encode(), headers={'Content-Type': kind, 'Accept': 'application/x-ndjson'})
        assert result.status_code == status
        assert result.headers['content-type'].startswith('application/json')
        assert result.json()['message']


def test_stream_has_identity_drafts_progress_and_final_event():
    with api(generate=generate) as client:
        result = client.post('/api/create', json=[{'testcase': 'first', 'description': SOURCE}], headers={'Accept': 'application/x-ndjson'})
        assert result.headers['x-accel-buffering'] == 'no'
        events = [json.loads(line) for line in result.text.splitlines()]
        assert [e['type'] for e in events] == ['start', 'categories', 'drafts', 'item_done', 'done']
        assert events[2]['testcase'] == 'first'
        assert events[2]['index'] == 0
        assert events[2]['drafts'][0]['mcId'] == 104
        assert events[-1] == {'type': 'done', 'total': 1, 'completed': 1, 'failed': 0}


def test_fallback_streams_valid_category_before_slow_sibling_and_cancels():
    async def run():
        started = asyncio.Event()
        cancelled = asyncio.Event()
        async def slow(payload, **_):
            if len(payload.categories) > 1:
                from backend.app.errors import BackendError
                raise BackendError(502, 'Ошибка формата.')
            if payload.categories[0].id == 103:
                started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            await started.wait()
            return response(104)
        state = SimpleNamespace(creation_workers=2, creation_slots=asyncio.Semaphore(2), tunnel=None,
                                qwen=SimpleNamespace(generate=slow))
        events = creation_events(SimpleNamespace(state=state), [CreateItem(description=SOURCE, categories=[{'id': 103}, {'id': 104}])], True, 'test')
        assert (await anext(events))['type'] == 'start'
        assert (await anext(events))['type'] == 'categories'
        first = await asyncio.wait_for(anext(events), timeout=1)
        assert first['type'] == 'drafts'
        assert first['drafts'][0]['mcId'] == 104
        await events.aclose()
        assert cancelled.is_set()
    asyncio.run(run())


def test_classification_runs_in_backend_thread_and_batch_continues_after_failure():
    main_thread = threading.get_ident()
    threads = []
    def classify(_model, description, _source):
        threads.append(threading.get_ident())
        if description == 'bad':
            raise ValueError('private model failure')
        return [{'id': 104, 'name': 'Сантехника'}]
    with api() as client, patch('backend.app.creation.classify', side_effect=classify):
        result = client.post('/api/create?generateText=false', json=[{'description': 'bad'}, {'description': 'good'}]).json()['results']
        assert result[0]['drafts'] == []
        assert result[0]['error']['message'] == 'Не удалось обработать объявление.'
        assert len(result[1]['drafts']) == 1
        assert threads and all(t != main_thread for t in threads)
        assert all(t.name.startswith('creation') for t in threading.enumerate() if t.ident in threads)


def test_generation_concurrency_is_bounded_across_batch():
    active = peak = 0
    async def tracked(payload, **_):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(.01)
        active -= 1
        return response(*(category.id for category in payload.categories))
    with api(generate=tracked) as client:
        result = client.post('/api/create', json=[{'description': SOURCE, 'categories': [{'id': 103}, {'id': 104}]}] * 8)
        assert result.status_code == 200
        assert len(result.json()['results']) == 8
    assert peak == 2


def test_real_qwen_adapter_batches_selected_categories_without_support_gate():
    calls = []
    def remote(request):
        payload = json.loads(request.content)
        categories = [c['id'] for c in json.loads(payload['messages'][-1]['content'])['requested_categories']]
        calls.append(categories)
        drafts = [{'categoryId': category, 'title': 'Плиточные работы' if category == 103 else 'Замена смесителей',
                   'text': 'Укладываем плитку.' if category == 103 else 'Меняем смесители.'} for category in categories]
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': json.dumps({'drafts': drafts}, ensure_ascii=False),
        }}]})
    with patch('backend.app.main.load_classifier', return_value=object()), \
         patch('backend.app.creation.classify', return_value=[{'id': 103}, {'id': 104}]):
        with TestClient(create_app(Settings(tokenize_before_generate=False), transport=httpx.MockTransport(remote))) as client:
            result = client.post('/api/create', json={'description': SOURCE}).json()
            assert result['drafts'] == [
                {'mcId': 103, 'title': 'Плиточные работы', 'text': 'Укладываем плитку.'},
                {'mcId': 104, 'title': 'Замена смесителей', 'text': 'Меняем смесители.'},
            ]
            assert calls == [[103, 104]]


def test_one_category_failure_preserves_sibling_draft_and_other_items():
    from backend.app.errors import BackendError
    async def partial(payload, **_):
        if any(c.id == 103 for c in payload.categories):
            raise BackendError(504, 'Qwen не ответил за отведённое время.')
        return response(104)
    with api(generate=partial) as client:
        result = client.post('/api/create', json=[
            {'description': SOURCE, 'categories': [{'id': 103}, {'id': 104}]},
            {'description': SOURCE, 'categories': [{'id': 104}]},
        ]).json()['results']
        assert [d['mcId'] for d in result[0]['drafts']] == [104]
        assert result[0]['errors'][0]['mcId'] == 103
        assert [d['mcId'] for d in result[1]['drafts']] == [104]
