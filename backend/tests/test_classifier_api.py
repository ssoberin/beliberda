"""The real category route must never silently use the frontend keyword stub."""
from contextlib import contextmanager
from unittest.mock import patch

from fastapi.testclient import TestClient
import httpx
import pytest

from backend.app.classifier import EXPECTED_SHA256, classify, load_classifier
from backend.app.config import Settings
from backend.app.main import create_app


class Pipeline:
    def predict_proba(self, texts):
        assert texts == ['Укладываю плитку и меняю трубы.']
        return [[.1, .51, .8, .49, .5]]


MODEL = {'pipeline': Pipeline(), 'ids': (102, 103, 104, 105, 106), 'thresholds': [.5] * 5}
PAYLOAD = {'title': 'Мастер', 'description': 'Укладываю плитку и меняю трубы.'}


@contextmanager
def client(model):
    with patch('backend.app.main.load_classifier', return_value=model):
        with TestClient(create_app(Settings(classifier_model_path='trusted.joblib'),
                                   transport=httpx.MockTransport(lambda _: httpx.Response(503)))) as api:
            yield api


def test_classification_is_independent_of_qwen_and_filters_source():
    with client(MODEL) as api:
        ready = api.get('/ready')
        assert ready.status_code == 503
        assert ready.json()['categoryGeneration'] is True
        assert api.post('/api/categories', json=PAYLOAD).json() == {'categories': [
            {'id': 103, 'name': 'Плиточные работы'},
            {'id': 104, 'name': 'Сантехника'},
            {'id': 106, 'name': 'Поклейка обоев и малярные работы'},
        ]}
        assert [c['id'] for c in api.post('/api/categories', json={**PAYLOAD, 'mcId': 104}).json()['categories']] == [103, 106]


def test_missing_model_never_falls_back_to_keywords():
    with client(None) as api:
        assert api.get('/ready').json()['categoryGeneration'] is False
        response = api.post('/api/categories', json=PAYLOAD)
        assert response.status_code == 503
        assert 'TF-IDF' in response.json()['message']


@pytest.mark.parametrize('payload', [
    {'description': ' '}, {'description': 'x' * 12001},
    {'description': 'текст', 'mcId': 999}, {'description': 'текст', 'unexpected': 1},
])
def test_invalid_category_requests(payload):
    with client(MODEL) as api:
        assert api.post('/api/categories', json=payload).status_code == 422


def test_classifier_uses_first_6000_characters_and_all_five_thresholds():
    class Spy:
        def predict_proba(self, texts):
            assert texts == ['x' * 6000]
            return [[.5, .49, .5, .99, .1]]
    assert [c['id'] for c in classify({**MODEL, 'pipeline': Spy()}, 'x' * 7000, 105)] == [102, 104]


def test_untrusted_joblib_is_rejected_before_deserializing(tmp_path):
    path = tmp_path / 'unknown.joblib'
    path.write_bytes(b'not a trusted model')
    with patch('joblib.load', side_effect=AssertionError('must not deserialize')) as load:
        with pytest.raises(ValueError, match='Хеш'):
            load_classifier(str(path))
        load.assert_not_called()
    assert load_classifier(str(path.with_name('missing.joblib'))) is None
    assert len(EXPECTED_SHA256) == 64
