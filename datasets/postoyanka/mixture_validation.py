"""Predeclared new-data ratios; validation only, never inspect test scores."""
import collections
import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path('/srv/Hakaton')
DATA = ROOT / 'innopolis_2026/datasets'
IDS = (102, 103, 104, 105, 106)


def read(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def labels(row, original):
    if 'offered_microcategories' in row:
        return set(row['offered_microcategories'])
    if 'labels' in row:
        return set(row['labels'])
    found = {item['mcId'] for item in row['found_microcategories']}
    if row['mcId'] != 101:
        assert row['mcId'] in original[row['meta']['source_item_id']]
        found.add(row['mcId'])
    return found


def measure(model, rows):
    start = perf_counter()
    scores = model['pipeline'].predict_proba([row['description'] for row in rows])
    seconds = perf_counter() - start
    gold = np.array([[int(mid in (set(row['labels']) if 'labels' in row else
                                    {item['mcId'] for item in row['found_microcategories']})
                          and row.get('mcId', 101) != mid) for mid in IDS] for row in rows])
    pred = np.array([[int(value >= .5 and row.get('mcId', 101) != mid)
                      for mid, value in zip(IDS, scores_row)] for row, scores_row in zip(rows, scores)])
    return {
        'n': len(rows), 'precision': round(precision_score(gold, pred, average='micro', zero_division=0), 4),
        'recall': round(recall_score(gold, pred, average='micro', zero_division=0), 4),
        'f1': round(f1_score(gold, pred, average='micro', zero_division=0), 4),
        'macro_f1': round(f1_score(gold, pred, average='macro', zero_division=0), 4),
        'fp': int(((pred == 1) & (gold == 0)).sum()),
        'fn': int(((pred == 0) & (gold == 1)).sum()),
        'empty': int((pred.sum(axis=1) == 0).sum()), 'seconds': round(seconds, 3),
    }


def subset(rows, fraction):
    by_set = collections.defaultdict(list)
    for row in rows:
        by_set[row['set']].append(row)
    selected = set()
    for family in sorted(by_set):
        ordered = sorted(by_set[family], key=lambda row: hashlib.sha256(
            str(row['item_id']).encode()).hexdigest())
        selected.update(row['item_id'] for row in ordered[:round(len(ordered) * fraction)])
    # Preserve original row order: SGD shuffling depends on it.
    return [row for row in rows if row['item_id'] in selected]


def main():
    public_train = read(ROOT / 'data/processed/train.jsonl')
    public_validation = read(ROOT / 'data/processed/validation.jsonl')
    v2_train = read(DATA / 'hard_dataset/v2/train.jsonl')
    v2_validation = [row for row in read(DATA / 'hard_dataset/v2/validation.jsonl') if row['mcId'] == 101]
    new_train = read(DATA / 'postoyanka/splits/train.jsonl')
    new_validation = read(DATA / 'postoyanka/splits/validation.jsonl')
    original = {row['item_id']: {item['mcId'] for item in row['found_microcategories']}
                for row in read(ROOT / 'data/raw/public-train-annotated.jsonl')}
    baseline = joblib.load(ROOT / 'models/baseline/tfidf_baseline.joblib')  # Trusted local artifact.
    evaluations = {
        'public_validation': public_validation,
        'v2_validation_source_101': v2_validation,
        'new_validation': new_validation,
        'new_val_real_derived': [row for row in new_validation if row['set'] in ('edit_real', 'mix_real_dirty')],
        'new_val_long_synthetic': [row for row in new_validation if row['set'] not in ('edit_real', 'mix_real_dirty')],
    }
    results = {}
    for fraction in (0, .25, .5, 1):
        addition = subset(new_train, fraction)
        train = public_train + v2_train + addition
        X = [row['description'][:6000] for row in train]
        Y = np.asarray([[int(mid in labels(row, original)) for mid in IDS] for row in train], dtype=np.int8)
        model = {'ids': IDS, 'thresholds': np.full(5, .5), 'pipeline': clone(baseline['pipeline'])}
        start = perf_counter()
        model['pipeline'].fit(X, Y)
        entry = {'ratio': fraction, 'n_new': len(addition), 'n_total': len(train),
                 'fit_seconds': round(perf_counter() - start, 2),
                 'validation': {name: measure(model, rows) for name, rows in evaluations.items()}}
        results[str(fraction)] = entry
        print(fraction, 'new', len(addition),
              [(name, score['f1'], score['precision'], score['recall'])
               for name, score in entry['validation'].items()], flush=True)
    target = ROOT / 'results/postoyanka_v2/mixture_validation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({'settings': 'Same five heads, fixed threshold .5, seed 2026, validation only',
                                  'ratios': results}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
