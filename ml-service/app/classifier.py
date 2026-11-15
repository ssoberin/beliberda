"""Legacy endpoint uses the same default C_syntax as backend/."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from avito_compare.c_syntax import load_bundle
TARGETS = {102, 103, 104, 105, 106}


def load_model():
    model = load_bundle()
    model['pipeline'].warmup()
    return model


def classify(model: dict, description: str, source_id: int) -> list[int]:
    """One label per target category. Source category is always excluded."""
    probabilities = model["pipeline"].predict_proba([description])[0]
    if len(probabilities) != len(model["ids"]):
        raise ValueError("Classifier probability shape mismatch")
    selected = [int(category_id) for category_id, p, threshold in zip(
        model["ids"], probabilities, model["thresholds"]
    ) if int(category_id) in TARGETS and int(category_id) != source_id and p >= threshold]
    return sorted(set(selected))
