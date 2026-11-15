"""Frozen first TF-IDF baseline; never deserialize an unverified artifact."""
import hashlib
from pathlib import Path

from .catalog import NAMES

EXPECTED_SHA256 = '668c95e2cecdaa2b94f1a82470e6b89fe3ab1808c765539b45291a22a15004bc'
IDS = (102, 103, 104, 105, 106)


def load_classifier(path: str):
    if not path:
        return None
    model_path = Path(path)
    if not model_path.is_file():
        return None
    with model_path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    if digest != EXPECTED_SHA256:
        raise ValueError('Хеш весов TF-IDF отличается от зафиксированного baseline; загрузка запрещена.')
    import joblib  # Imported only when classification is configured.
    model = joblib.load(model_path)
    if tuple(map(int, model['ids'])) != IDS or len(model['thresholds']) != len(IDS):
        raise ValueError('Неверный формат весов TF-IDF.')
    if any(float(th) != 0.5 for th in model['thresholds']):
        raise ValueError('Пороги TF-IDF отличаются от зафиксированного baseline.')
    return model


def classify(model, description: str, source_id: int):
    # Baseline was trained and evaluated on the first 6000 description characters.
    probabilities = model['pipeline'].predict_proba([description[:6000]])[0]
    if len(probabilities) != len(IDS):
        raise ValueError('Неверный размер ответа классификатора.')
    return [{'id': mid, 'name': NAMES[mid]} for mid, prob, threshold in
            zip(IDS, probabilities, model['thresholds'])
            if mid != source_id and prob >= threshold]
