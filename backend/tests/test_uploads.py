"""Original multipart files become isolated classification/Qwen requests."""
from contextlib import contextmanager
import json
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.config import Settings
from backend.app.creation import MAX_BODY_BYTES
from backend.app.main import create_app


@contextmanager
def api(calls):
    def classify(_model, description, source):
        calls['classification'].append((description, source, threading.current_thread().name))
        return [{'id':104, 'name':'Сантехника'}]

    def qwen(request):
        body = json.loads(request.content)
        source = json.loads(body['messages'][-1]['content'])
        calls['qwen'].append(source)
        assert body['response_format']['type'] == 'json_schema'
        if 'generation-error' in source['description']:
            return httpx.Response(503)
        return httpx.Response(200, json={'choices':[{'finish_reason':'stop', 'message':{
            'content':json.dumps({'drafts':[{'categoryId':category['id'], 'title':'Сантехнические работы',
                                          'text':source['description']} for category in source['requested_categories']]}, ensure_ascii=False),
        }}]})

    with patch('backend.app.main.load_classifier', return_value=object()), \
         patch('backend.app.creation.classify', side_effect=classify):
        with TestClient(create_app(Settings(tokenize_before_generate=False), transport=httpx.MockTransport(qwen))) as client:
            yield client


@pytest.fixture
def calls():
    return {'classification':[], 'qwen':[]}


def stream(client, source, filename='input.json', title='Общий заголовок'):
    result = client.post('/api/create', files={'file':(filename, source, 'application/octet-stream')},
                         data={'title':title}, headers={'Accept':'application/x-ndjson'})
    assert result.status_code == 200, result.text
    return [json.loads(line) for line in result.text.splitlines()]


@pytest.mark.parametrize('filename,source', [
    ('input.json', json.dumps([
        {'testcase':'first', 'description':'Первая запись: меняем трубы.', 'mcId':101},
        {'testcase':'second', 'description':'Вторая запись: подключаем раковины.', 'mcId':103, 'title':'Собственный заголовок'},
    ], ensure_ascii=False)),
    ('INPUT.CSV', '\ufefftestcase,description,mcId,title\r\nfirst,Первая запись: меняем трубы.,101,\r\nsecond,Вторая запись: подключаем раковины.,103,Собственный заголовок\r\n'),
    ('input.csv', 'testcase;description;mcId;title\nfirst;Первая запись: меняем трубы.;101;\nsecond;Вторая запись: подключаем раковины.;103;Собственный заголовок'),
])
def test_original_file_becomes_separate_model_messages(filename, source, calls):
    with api(calls) as client:
        events = stream(client, source.encode('utf-8'), filename)
    assert events[0]['total'] == 2 and events[0]['batch'] is True
    assert events[-1] == {'type':'done', 'total':2, 'completed':2, 'failed':0}
    assert len(calls['classification']) == len(calls['qwen']) == 2
    by_text = {item['description']:item for item in calls['qwen']}
    first = by_text['Первая запись: меняем трубы.']
    second = by_text['Вторая запись: подключаем раковины.']
    assert (first['title'], first['source_category']) == ('Общий заголовок',101)
    assert (second['title'], second['source_category']) == ('Собственный заголовок',103)
    assert all(thread.startswith('classifier') for _, _, thread in calls['classification'])
    results = sorted((e for e in events if e['type'] == 'item_done'), key=lambda e:e['index'])
    assert [e['testcase'] for e in results] == ['first','second']
    assert [e['drafts'][0]['text'] for e in results] == ['Первая запись: меняем трубы.', 'Вторая запись: подключаем раковины.']


def test_quoted_csv_newlines_and_delimiters_are_preserved(calls):
    source = '\ufefftestcase,description\r\nfirst,"Название ""Мастер""; трубы, сифоны.\nРаботаем по записи."\r\nsecond,Меняем раковины.\r\n'
    with api(calls) as client:
        events = stream(client, source.encode(), 'input.csv')
    assert events[-1]['failed'] == 0
    assert {item['description'] for item in calls['qwen']} == {
        'Название "Мастер"; трубы, сифоны.\nРаботаем по записи.', 'Меняем раковины.',
    }


@pytest.mark.parametrize('filename,source', [
    ('input.json', json.dumps([{'description':'Первая запись.'}, {'testcase':'bad','description':' '}, {'description':'Третья запись.'}])),
    ('input.json', json.dumps([{'description':'Первая запись.'}, 42, {'description':'Третья запись.'}])),
    ('input.csv', 'testcase,description,mcId\nfirst,Первая запись.,101\nbad,Ошибка.,not-a-number\nlast,Третья запись.,101'),
    ('input.csv', 'testcase,description\nfirst,Первая запись.\nbad,Ошибка.,лишнее поле\nlast,Третья запись.'),
])
def test_bad_record_does_not_stop_other_records(filename, source, calls):
    with api(calls) as client:
        events = stream(client, source.encode(), filename)
    assert events[-1] == {'type':'done','total':3,'completed':3,'failed':1}
    results = {e['index']:e for e in events if e['type'] == 'item_done'}
    assert results[1]['error']['code'] == 'validation_error'
    assert results[1]['drafts'] == []
    assert results[0]['drafts'] and results[2]['drafts']
    assert len(calls['classification']) == len(calls['qwen']) == 2


def test_generation_error_keeps_sibling_record_result(calls):
    with api(calls) as client:
        events = stream(client, b'[{"description":"generation-error"},{"description":"Working record"}]')
    assert events[-1]['failed'] == 1
    results = {e['index']:e for e in events if e['type'] == 'item_done'}
    assert results[0]['errors'][0]['code'] == 'service_unavailable'
    assert results[1]['drafts'][0]['text'] == 'Working record'


@pytest.mark.parametrize('source,total,batch', [
    (' Меняем трубы. ',1,False),
    ('{"description":"Меняем трубы."}',1,False),
    ('[{"description":"Трубы."},{"description":"Раковины."}]',2,True),
    ('{"items":[{"description":"Трубы."}],"generateText":true}',1,True),
    ('description\nТрубы.\nРаковины.',2,True),
    ('description;title\nТрубы.;Собственное',1,True),
])
def test_pasted_input_is_detected_and_normalized_on_backend(source, total, batch, calls):
    with api(calls) as client:
        result = client.post('/api/create', files={'text':(None,source), 'title':(None,'Общее')},
                             headers={'Accept':'application/x-ndjson'})
    assert result.status_code == 200, result.text
    events = [json.loads(line) for line in result.text.splitlines()]
    assert events[0]['total'] == total and events[0]['batch'] is batch
    assert events[-1]['failed'] == 0
    assert len(calls['qwen']) == total
    assert all(item['title'] == ('Собственное' if 'Собственное' in source else 'Общее') for item in calls['qwen'])


@pytest.mark.parametrize('filename,source,status', [
    ('broken.json',b'{',422), ('empty.csv',b'',422),
    ('bad.csv',b'description\n"unclosed',422), ('bad.json',b'\xff',422),
    ('input.exe',b'{}',415), ('input.json',b'[]',422),
    ('input.json',json.dumps([{'description':'x'}]*301).encode(),422),
])
def test_file_level_errors_are_clean_json_before_stream(filename, source, status, calls):
    with api(calls) as client:
        result = client.post('/api/create', files={'file':(filename,source)}, headers={'Accept':'application/x-ndjson'})
    assert result.status_code == status
    assert result.json()['message'] and result.json()['code']
    assert result.json()['requestId'] == result.headers['x-request-id']
    assert calls['classification'] == calls['qwen'] == []


@pytest.mark.parametrize('parts', [
    [('file',('a.json',b'{}')),('file',('b.json',b'{}'))],
    [('file',('a.json',b'{}')),('text',(None,'also text'))],
    [('text',(None,'text')),('title',(None,'a')),('title',(None,'b'))],
    [('file',(None,'not a file'))],
    [('unknown',(None,'text'))],
])
def test_ambiguous_or_invalid_multipart_is_rejected(parts, calls):
    with api(calls) as client:
        result = client.post('/api/create', files=parts)
    assert result.status_code == 422
    assert calls['qwen'] == []


@pytest.mark.parametrize('extra,status', [(0,200),(1,413),(65537,413)])
def test_file_size_limit_excludes_multipart_headers(extra, status, calls):
    record = b'{"description":"x"}'
    body = record + b' ' * (MAX_BODY_BYTES - len(record) + extra)
    with api(calls) as client:
        result = client.post('/api/create?generateText=false', files={'file':('large.json',body)}, data={'title':'x'*100})
    assert result.status_code == status, result.text
    assert calls['qwen'] == []


def test_text_part_size_limit_is_reported_as_413(calls):
    with api(calls) as client:
        result = client.post('/api/create', files={'text':(None,'x'*(MAX_BODY_BYTES+1))})
    assert result.status_code == 413


@pytest.mark.parametrize('kind', ['json','csv'])
@pytest.mark.parametrize('count,status', [(300,200),(301,422)])
def test_upload_record_limit(kind, count, status, calls):
    if kind == 'json':
        source = json.dumps([{'testcase':f'case-{i}', 'description':'Меняем трубы.'} for i in range(count)])
    else:
        source = 'testcase,description\n' + ''.join(f'case-{i},Меняем трубы.\n' for i in range(count))
    with api(calls) as client:
        result = client.post('/api/create?generateText=false', files={'file':(f'input.{kind}',source.encode())},
                             headers={'Accept':'application/x-ndjson'})
    assert result.status_code == status
    if status == 200:
        events = [json.loads(line) for line in result.text.splitlines()]
        assert events[0]['total'] == 300
        assert events[-1] == {'type':'done','total':300,'completed':300,'failed':0}
        items = sorted((e for e in events if e['type'] == 'item_done'), key=lambda e:e['index'])
        assert [e['testcase'] for e in items] == [f'case-{i}' for i in range(300)]
        assert all(len(e['drafts']) == 1 for e in items)
        assert len(calls['classification']) == 300
    else:
        assert '300' in result.json()['message']
        assert calls['classification'] == []
    assert calls['qwen'] == []
