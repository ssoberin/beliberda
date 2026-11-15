"""Measured C_syntax is the normal backend classifier."""
from avito_compare.c_syntax import EXPECTED_FILES, ClassifierUnavailable, IDS, load_bundle
from .catalog import NAMES
from .errors import BackendError
EXPECTED_SHA256 = EXPECTED_FILES['catboost.cbm']


def load_classifier(path: str = ''):
    return load_bundle(path)


def classify(model, description: str, source_id: int):
    try:
        probabilities = model['pipeline'].predict_proba([description])[0]
    except ClassifierUnavailable as exc:
        raise BackendError(503, str(exc)) from exc
    if len(probabilities) != len(IDS):
        raise ValueError('Неверный размер ответа C_syntax.')
    return [{'id': mid, 'name': NAMES[mid]} for mid, prob, threshold in
            zip(IDS, probabilities, model['thresholds'])
            if mid != source_id and prob >= threshold]
