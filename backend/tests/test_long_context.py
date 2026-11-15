import json
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
import pytest

from backend.app.config import Settings
from backend.app.context import parse_selection, source_units, source_windows
from backend.app.main import create_app
from backend.app.schemas import ClassificationRequest


def test_long_input_visits_tail_and_middle_without_cutting_at_12000():
    text=('Фоновый текст.\n'*1500)+'Выполняем штукатурку по маякам.\n'+('Фоновый текст.\n'*1500)+'Работаем только в составе ремонта.'
    windows=list(source_windows(text))
    assert any('Выполняем штукатурку по маякам.' in w for w in windows)
    assert any('Работаем только в составе ремонта.' in w for w in windows)
    assert windows[-1].endswith('Работаем только в составе ремонта.')
    assert ClassificationRequest(description=text).description == text


def test_extractor_can_only_select_original_text_and_rejects_invented_ids():
    units=['Монтируем розетки.','Только в составе ремонта.']
    for invalid in ('[2]','[-1]','[true]','["Даем гарантию 10 лет."]'):
        with pytest.raises(ValueError,match='existing source IDs'):
            parse_selection('{"selected":'+invalid+'}',units)
    assert parse_selection('{"selected":[]}',units) == []
    assert parse_selection('{"selected":[1,0,1]}',units) == units
    text='Работаем только в составе ремонта.\n'*100
    assert ''.join(source_units(text)) == text


def test_long_generation_reads_all_windows_and_preserves_source_restrictions():
    facts=['Устанавливаем розетки.', 'Согласуем стоимость до начала работ.', 'Работаем только в составе ремонта.']
    description=('Фоновый текст.\n'*1500)+facts[0]+'\n'+facts[1]+('Нейтральное описание.\n'*1000)+facts[2]
    seen=[]; generated=[]
    def remote(request):
        body=json.loads(request.content)
        source=json.loads(body['messages'][-1]['content'])
        if request.url.path == '/tokenize':
            too_long=len(source.get('description',''))>9000
            return httpx.Response(200,json={'count':10000 if too_long else 1000,'max_model_len':8192})
        if 'source_units' in source:
            seen.append(''.join(unit['text'] for unit in source['source_units']))
            content={'selected':[unit['id'] for unit in source['source_units'] if any(fact in unit['text'] for fact in facts)]}
        else:
            generated.append(source['description'])
            content={'drafts':[{'categoryId':105,'title':'Установка розеток','text':' '.join(facts)}]}
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content,ensure_ascii=False)}}]})
    with patch('backend.app.main.load_classifier',return_value=None):
        with TestClient(create_app(Settings(),transport=httpx.MockTransport(remote))) as client:
            result=client.post('/api/generate',json={'description':description,'categories':[{'id':105}]})
    assert result.status_code == 200,result.text
    assert seen == list(source_windows(description))
    assert len(generated) == 1
    assert all(fact in generated[0] for fact in facts)
    assert result.json()['drafts'][0]['description'] == ' '.join(facts)
