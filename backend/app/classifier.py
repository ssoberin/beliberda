"""Measured C_syntax is the normal backend classifier."""
import logging
import time
from .observability import event

from avito_compare.c_syntax import EXPECTED_FILES, ClassifierUnavailable, IDS, load_bundle
from .catalog import NAMES
from .errors import BackendError
logger = logging.getLogger(__name__)
EXPECTED_SHA256 = EXPECTED_FILES['catboost.cbm']


def load_classifier(path: str = ''):
    return load_bundle(path)


def classify(model, description: str, source_id: int):
    started = time.monotonic()
    event(logger, 'classification_start', source_id=source_id, description_chars=len(description), model='C_syntax')
    try:
        probabilities = model['pipeline'].predict_proba([description])[0]
    except ClassifierUnavailable as exc:
        raise BackendError(503, str(exc)) from exc
    if len(probabilities) != len(IDS):
        raise ValueError('Неверный размер ответа C_syntax.')
    result = [{'id': mid, 'name': NAMES[mid]} for mid, prob, threshold in
              zip(IDS, probabilities, model['thresholds']) if mid != source_id and prob >= threshold]
    event(logger, 'classification_complete', categories=[row['id'] for row in result],
          probabilities={mid: float(prob) for mid, prob in zip(IDS, probabilities)},
          elapsed_ms=int((time.monotonic()-started)*1000))
    return result
