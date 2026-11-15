"""Real retrieval, SGD, Natasha and CatBoost reproduce frozen pipeline outputs."""
import json
from pathlib import Path
import numpy as np
from avito_compare.c_syntax import CSyntaxClassifier, DEFAULT_MODEL_DIR, load_bundle
from backend.app.classifier import classify

def test_frozen_pipeline_equivalence_and_source_exclusion():
    fixtures = Path(__file__).parent / 'fixtures'
    data = json.loads((fixtures / 'c_syntax_reference.json').read_text())
    vectors = np.load(fixtures / 'c_syntax_reference.npz')['embeddings'].astype(np.float32)
    cache = dict(zip(data['chunks'], vectors))
    bundle = load_bundle()
    pipeline = bundle['pipeline']
    assert isinstance(pipeline, CSyntaxClassifier)
    pipeline._encode = lambda chunks: np.vstack([cache[text] for text in chunks])
    for item in data['items']:
        actual = pipeline.predict_proba([item['description']])[0]
        np.testing.assert_allclose(actual, item['probabilities'], rtol=0, atol=1e-12)
        selected = classify(bundle, item['description'], 101)
        assert [c['id'] for c in selected] == item['predictions']
        for source in item['predictions']:
            assert source not in [c['id'] for c in classify(bundle, item['description'], source)]
        assert pipeline._analyzer._sent_cache == {}

def test_default_load_uses_c_syntax_without_loading_qwen_weights():
    bundle = load_bundle()
    assert DEFAULT_MODEL_DIR.name == 'c_syntax'
    assert bundle['ids'] == (102,103,104,105,106)
    assert bundle['thresholds'] == [.5]*5
    assert bundle['pipeline']._encoder is None
