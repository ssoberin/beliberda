"""Measure extra frozen TF-IDF CPU work on calibration texts, without scoring."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import json
import time
from pathlib import Path
import joblib
import numpy as np

P = Path('output/catboost-optimization')
model_path = Path('output/syntax-aware-experiment/models/tfidf_baseline.joblib')
texts = [json.loads(line)['description'] for line in
         Path('output/ngram-comparison/splits/calibration.jsonl').read_text().splitlines()]
model = joblib.load(model_path)
model.predict_proba(texts[:10])
times = []
for _ in range(3):
    start = time.perf_counter()
    model.predict_proba(texts)
    times.append(time.perf_counter() - start)
result = {
    'texts': len(texts), 'domain': 'public calibration', 'warm_runs': 3,
    'batch_seconds': times, 'median_batch_seconds': float(np.median(times)),
    'median_ms_per_ad_amortized': float(np.median(times)*1000/len(texts)),
    'model_bytes': model_path.stat().st_size,
    'word_features': len(model.named_steps['vectorizer'].transformer_list[0][1].vocabulary_),
    'char_features': len(model.named_steps['vectorizer'].transformer_list[1][1].vocabulary_),
    'scope': 'TF-IDF transform plus 5 SGD heads only; batch average, not end-to-end or single-request latency',
}
(P/'tfidf_inference_benchmark.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
