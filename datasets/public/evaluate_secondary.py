"""Read-only benchmark of locally saved classifiers on the secondary Parquet.

Requires pyarrow. Uses the public and secondary tests only for reporting, never
for fitting or threshold selection. The offer-bullet check is explicitly oracle.
"""
import collections
import json
import statistics
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, precision_score, recall_score

R = Path('/srv/Hakaton')
D = R / 'innopolis_2026/datasets'
IDS = (102, 103, 104, 105, 106)
NAMES = {'Штукатурные работы': 102, 'Плиточные работы': 103, 'Сантехника': 104,
         'Электрика': 105, 'Поклейка обоев и малярные работы': 106}
norm = lambda text: ' '.join(text.casefold().split())

def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]

public = pq.read_table(D / 'public/public-test-dataset.parquet').to_pylist()
synthetic = pq.read_table(D / 'public/public-test-synthetic-300.parquet').to_pylist()
val = read(R / 'data/processed/validation.jsonl')
newtest = read(D / 'postoyanka/splits/test.jsonl')
models = {
    'old_api': joblib.load(R / 'models/baseline/tfidf_baseline.joblib'),
    'public_v2_control': joblib.load(R / 'models/experiments/tfidf_public_v2_five_heads_control.joblib'),
    'public_v2_new': joblib.load(R / 'models/experiments/tfidf_public_v2_postoyanka_five_heads.joblib'),
    'six_heads': joblib.load(R / 'models/experiments/tfidf_public_plus_v2_six_heads.joblib'),
}

def gold(row):
    if 'categories' in row:
        return {NAMES[x] for x in row['categories']}
    if 'labels' in row:
        return set(row['labels'])
    return {x['mcId'] for x in row['found_microcategories']}

def report(rows, model):
    t0 = perf_counter()
    scores = model['pipeline'].predict_proba([r['description'] for r in rows])
    elapsed = perf_counter() - t0
    yt = np.asarray([[int(mid in gold(r)) for mid in IDS] for r in rows])
    yp = np.asarray([[int(mid in {int(m) for m, p, th in zip(model['ids'], values, model['thresholds'])
                              if p >= th and int(m) != r.get('mcId', 101)}) for mid in IDS]
                     for r, values in zip(rows, scores)])
    return {
        'n': len(rows), 'precision': round(precision_score(yt, yp, average='micro', zero_division=0), 4),
        'recall': round(recall_score(yt, yp, average='micro', zero_division=0), 4),
        'micro_f1': round(f1_score(yt, yp, average='micro', zero_division=0), 4),
        'macro_f1': round(f1_score(yt, yp, average='macro', zero_division=0), 4),
        'fp': int(((yp == 1) & (yt == 0)).sum()), 'fn': int(((yp == 0) & (yt == 1)).sum()),
        'empty': int(np.sum(yp.sum(axis=1) == 0)),
        'exact': round(np.mean(np.all(yp == yt, axis=1)).item(), 4),
        'by_label': {str(mid): {'p': round(precision_score(yt[:, j], yp[:, j], zero_division=0), 4),
                                'r': round(recall_score(yt[:, j], yp[:, j], zero_division=0), 4),
                                'f1': round(f1_score(yt[:, j], yp[:, j], zero_division=0), 4),
                                'fp': int(((yp[:, j] == 1) & (yt[:, j] == 0)).sum()),
                                'fn': int(((yp[:, j] == 0) & (yt[:, j] == 1)).sum())}
                     for j, mid in enumerate(IDS)},
        'seconds': round(elapsed, 3),
    }

def dist(rows):
    lengths = sorted(len(r['description']) for r in rows)
    cardinality = collections.Counter(len(gold(r) & set(IDS)) for r in rows)
    return {'n': len(rows), 'unique_normalized': len({norm(r['description']) for r in rows}),
            'median_length': lengths[len(rows)//2], 'p90_length': lengths[int(len(rows)*.9)],
            'min_length': lengths[0], 'max_length': lengths[-1],
            'cardinality': dict(cardinality),
            'label_counts': {str(mid): sum(mid in gold(r) for r in rows) for mid in IDS}}

sets = {'public_test': public, 'synthetic_300': synthetic, 'public_validation': val,
        'new_long_test': newtest}
result = {'distribution': {name: dist(rows) for name, rows in sets.items()},
          'metrics': {name: {model_name: report(rows, model) for model_name, model in models.items()}
                      for name, rows in sets.items()}, 'synthetic_subsets': {}}
for name, rows in {
    'synthetic_empty': [r for r in synthetic if not r['categories']],
    'synthetic_positive': [r for r in synthetic if r['categories']],
    'synthetic_refusal_question': [r for r in synthetic if 'В сообщениях часто спрашивают:' in r['description']],
    'synthetic_other_contractor': [r for r in synthetic if 'заказчик поручает другим' in r['description']],
    'synthetic_customer_wishes': [r for r in synthetic if 'это пожелания к готовому помещению' in r['description']],
}.items():
    result['synthetic_subsets'][name] = {key: report(rows, model) for key, model in models.items()}

train_sets = {
    'public_train': read(R / 'data/processed/train.jsonl'),
    'v2_train': read(D / 'hard_dataset/v2/train.jsonl'),
    'new_train': read(D / 'postoyanka/splits/train.jsonl'),
}
train_keys = {name: {norm(r['description']) for r in rows} for name, rows in train_sets.items()}
for name, rows in [('synthetic_300', synthetic), ('public_test', public)]:
    result.setdefault('overlaps', {})[name] = {
        train_name: sum(norm(r['description']) in keys for r in rows)
        for train_name, keys in train_keys.items()
    }
public_keys = {norm(p['description']) for p in public}
result['overlaps']['synthetic_vs_public_test'] = sum(norm(r['description']) in public_keys for r in synthetic)
out = R / 'results/postoyanka_v2/secondary_test_evaluation.json'
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('distribution', result['distribution'])
print('overlaps', result['overlaps'])
for name, scores in result['metrics'].items():
    print(name, [(key, m['micro_f1'], m['precision'], m['recall'], m['fp'], m['fn'], m['empty'], m['exact'])
                 for key, m in scores.items()])
for name, scores in result['synthetic_subsets'].items():
    print(name, [(key, m['n'], m['fp'], m['fn'], m['empty'], m['micro_f1']) for key, m in scores.items()])
print(out)

# Diagnostic only: the generator marks the answer-bearing bullet lines. Reading
# that privileged boundary is NOT a fair model evaluation or deployable segmenter.
positive = [row for row in synthetic if row['categories']]
for model_name, model in models.items():
    gold_matrix = np.asarray([[int(mid in gold(row)) for mid in IDS] for row in positive])
    subset_probs = []
    for row in positive:
        lines = [line[2:] for line in row['description'].splitlines() if line.startswith('• ')]
        probs = model['pipeline'].predict_proba(lines)
        subset_probs.append([max(probs[:, list(model['ids']).index(mid)]) for mid in IDS])
    pred = (np.asarray(subset_probs) >= .5).astype(int)
    print('ORACLE OFFER-BULLET DIAGNOSTIC', model_name,
          'recall', round(recall_score(gold_matrix, pred, average='micro'), 4),
          'precision', round(precision_score(gold_matrix, pred, average='micro', zero_division=0), 4),
          'FN', int(((pred == 0) & (gold_matrix == 1)).sum()),
          'FP within positive', int(((pred == 1) & (gold_matrix == 0)).sum()))
