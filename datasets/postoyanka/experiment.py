"""Leakage-aware split and five-head TF-IDF/SGD benchmark; never deploys a model."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score

IDS = (102, 103, 104, 105, 106)
ROOT = Path('/srv/Hakaton')
REPO = ROOT / 'innopolis_2026'
PUBLIC = ROOT / 'data/processed'
V2 = REPO / 'datasets/hard_dataset/v2'
NEW = REPO / 'datasets/postoyanka/dirty_dataset_v2.jsonl'
SPLITS = REPO / 'datasets/postoyanka/splits'
OUT = ROOT / 'results/postoyanka_v2'
BASELINE = ROOT / 'models/baseline/tfidf_baseline.joblib'


def read(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def norm(text):
    return ' '.join(text.casefold().split())


def labels(row):
    if 'labels' in row:
        return set(row['labels'])
    return {item['mcId'] for item in row['found_microcategories']}


def split_new(rows, old):
    """Group by origin, exact description and embedded core; anchor real edits to old splits."""
    public_raw = {r['item_id']: r for r in read(ROOT / 'data/raw/public-train-annotated.jsonl')}
    anchors = {}
    for split, records in old.items():
        for row in records:
            key = norm(row['description'])
            if key in anchors and anchors[key] != split:
                raise ValueError('Existing public split contains duplicate text across partitions')
            anchors[key] = split
    parent = list(range(len(rows)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    keys = {}
    for i, row in enumerate(rows):
        meta = row.get('meta', {})
        for key in [('text', norm(row['description']))] + (
            [('origin', meta['source_item_id'])] if 'source_item_id' in meta else []
        ) + ([('core', norm(meta['core']))] if meta.get('core') else []):
            if key in keys:
                union(i, keys[key])
            else:
                keys[key] = i

    groups = collections.defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    assigned = {}
    quarantine = []
    free = collections.defaultdict(list)
    for group in groups.values():
        destinations = set()
        text_gold = collections.defaultdict(set)
        for i in group:
            row = rows[i]
            text_gold[norm(row['description'])].add(tuple(sorted(labels(row))))
            if norm(row['description']) in anchors:
                destinations.add(anchors[norm(row['description'])])
            sid = row.get('meta', {}).get('source_item_id')
            if sid is not None:
                destinations.add(anchors[norm(public_raw[sid]['description'])])
        if len(destinations) > 1 or any(len(golds) > 1 for golds in text_gold.values()):
            quarantine.extend(group)
        elif destinations:
            for i in group:
                assigned[i] = next(iter(destinations))
        else:
            free[rows[group[0]]['set']].append(group)

    # A single fixed seed; no choice is made using evaluation labels or scores.
    rng = random.Random(2026)
    for family, family_groups in sorted(free.items()):
        rng.shuffle(family_groups)
        n = len(family_groups)
        for j, group in enumerate(family_groups):
            dest = 'train' if j < round(n * .70) else ('validation' if j < round(n * .85) else 'test')
            for i in group:
                assigned[i] = dest
    if len(assigned) + len(quarantine) != len(rows):
        raise AssertionError('Split did not account for all records')
    result = {name: [row for i, row in enumerate(rows) if assigned.get(i) == name]
              for name in ('train', 'validation', 'test')}
    result['quarantine'] = [rows[i] for i in quarantine]
    # An unmodified old holdout must never have a source or exact text in new train.
    holdout_texts = {norm(r['description']) for r in old['validation'] + old['test']}
    for row in result['train']:
        assert norm(row['description']) not in holdout_texts
        sid = row.get('meta', {}).get('source_item_id')
        if sid is not None:
            assert anchors[norm(public_raw[sid]['description'])] == 'train'
    return result


def gold(row):
    return labels(row) - {row.get('mcId', 101)}


def assess(rows, model):
    t0 = perf_counter()
    if not rows:
        raise ValueError('Empty evaluation slice')
    probabilities = model['pipeline'].predict_proba([r['description'] for r in rows])
    seconds = perf_counter() - t0
    pred = [{int(mid) for mid, p, threshold in zip(model['ids'], values, model['thresholds'])
             if p >= threshold and mid != r.get('mcId', 101)} for r, values in zip(rows, probabilities)]
    yt = np.array([[int(mid in gold(row)) for mid in IDS] for row in rows])
    yp = np.array([[int(mid in ids) for mid in IDS] for ids in pred])
    return {
        'n': len(rows), 'precision': round(precision_score(yt, yp, average='micro', zero_division=0), 4),
        'recall': round(recall_score(yt, yp, average='micro', zero_division=0), 4),
        'micro_f1': round(f1_score(yt, yp, average='micro', zero_division=0), 4),
        'macro_f1': round(f1_score(yt, yp, average='macro', zero_division=0), 4),
        'exact_match_five': round(float(np.all(yt == yp, axis=1).mean()), 4),
        'fp': int(((yp == 1) & (yt == 0)).sum()), 'fn': int(((yp == 0) & (yt == 1)).sum()),
        'empty_predictions': sum(not ids for ids in pred),
        'unavailable_101_gold': sum(101 in gold(row) for row in rows),
        'inference_seconds': round(seconds, 3),
        'f1_by_category': {str(mid): round(f1_score(yt[:, i], yp[:, i], zero_division=0), 4)
                           for i, mid in enumerate(IDS)},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Only validate and inspect split')
    parser.add_argument('--evaluate-only', action='store_true', help='Reuse the locally saved experimental model')
    args = parser.parse_args()
    old = {name: read(PUBLIC / f'{file}.jsonl') for name, file in
           [('train', 'train'), ('validation', 'calibration'), ('test', 'validation')]}
    rows = read(NEW)
    split = split_new(rows, old)
    for name, batch in split.items():
        print(name, len(batch), collections.Counter(r['set'] for r in batch))
    if args.skip_train:
        return
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, batch in split.items():
        (SPLITS / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in batch), encoding='utf-8')
    baseline = joblib.load(BASELINE)  # Known trusted locally trained artifact.
    assert tuple(baseline['ids']) == IDS
    new_train = old['train'] + read(V2 / 'train.jsonl') + split['train']
    # New rows only contain output labels, which exclude the source category.
    # For SRC edits the original public annotation confirms the source category;
    # restore it as an offered training target, not from the held-out output.
    public_raw = {r['item_id']: r for r in read(ROOT / 'data/raw/public-train-annotated.jsonl')}
    source_labels = {}
    for row in split['train']:
        if row['mcId'] != 101:
            original = public_raw[row['meta']['source_item_id']]
            assert row['mcId'] in {item['mcId'] for item in original['found_microcategories']}
            source_labels[row['item_id']] = row['mcId']
    artifact = ROOT / 'models/experiments/tfidf_public_v2_postoyanka_five_heads.joblib'
    control_artifact = ROOT / 'models/experiments/tfidf_public_v2_five_heads_control.joblib'
    if args.evaluate_only:
        model = joblib.load(artifact)  # Locally generated by this experiment.
        fit_seconds = json.loads((OUT / 'metrics.json').read_text())['fit_seconds']
    else:
        x = [r['description'][:6000] for r in new_train]
        y = np.asarray([[int(mid in (r['offered_microcategories'] if 'offered_microcategories' in r
                                     else labels(r) | ({source_labels[r['item_id']]} if r.get('item_id') in source_labels else set())))
                         for mid in IDS] for r in new_train], dtype=np.int8)
        model = {'ids': IDS, 'thresholds': np.full(5, .5), 'pipeline': clone(baseline['pipeline'])}
        t0 = perf_counter()
        model['pipeline'].fit(x, y)
        fit_seconds = round(perf_counter() - t0, 2)
        joblib.dump(model, artifact)
    # Like-for-like ablation: same five heads and training recipe without the new set.
    if args.evaluate_only and control_artifact.exists():
        control = joblib.load(control_artifact)
    else:
        original_train = old['train'] + read(V2 / 'train.jsonl')
        control_y = np.asarray([[int(mid in (r['offered_microcategories'] if 'offered_microcategories' in r
                                            else labels(r))) for mid in IDS]
                                for r in original_train], dtype=np.int8)
        control = {'ids': IDS, 'thresholds': np.full(5, .5), 'pipeline': clone(baseline['pipeline'])}
        control['pipeline'].fit([r['description'][:6000] for r in original_train], control_y)
        joblib.dump(control, control_artifact)
    OUT.mkdir(parents=True, exist_ok=True)
    eval_sets = {
        'public_validation': old['test'],
        'new_validation': split['validation'],
        'new_test': split['test'],
        'new_test_real_derived': [r for r in split['test'] if r['set'] in ('edit_real', 'mix_real_dirty')],
        'new_test_long_synthetic': [r for r in split['test'] if r['set'] not in ('edit_real', 'mix_real_dirty')],
        'v2_validation_source_101': [r for r in read(V2 / 'validation.jsonl') if r['mcId'] == 101],
    }
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print('pyarrow not available: skipping public test (install separately, not in service venv)')
    else:
        names = {'Штукатурные работы': 102, 'Плиточные работы': 103,
                 'Сантехника': 104, 'Электрика': 105,
                 'Поклейка обоев и малярные работы': 106}
        eval_sets['public_test_diagnostic'] = [
            {'description': r['description'], 'mcId': 101,
             'labels': [names[name] for name in r['categories']]}
            for r in pq.read_table(REPO / 'datasets/public/public-test-dataset.parquet').to_pylist()
        ]
    for family in sorted({r['set'] for r in split['test']}):
        eval_sets['new_test_' + family] = [r for r in split['test'] if r['set'] == family]
    outcome = {'dataset_sha256': hashlib.sha256(NEW.read_bytes()).hexdigest(), 'seed': 2026,
               'split_counts': {name: len(batch) for name, batch in split.items()},
               'train_counts': {'public': len(old['train']), 'v2': len(read(V2 / 'train.jsonl')),
                                'new_used': len(split['train']),
                                'new_src_source_label_restored': len(source_labels)},
               'model': str(artifact), 'control_model': str(control_artifact), 'fit_seconds': fit_seconds,
               'evaluation': {name: {'baseline': assess(batch, baseline),
                                     'public_v2_control': assess(batch, control),
                                     'trained': assess(batch, model)}
                              for name, batch in eval_sets.items() if batch}}
    (OUT / 'metrics.json').write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding='utf-8')
    for name, scores in outcome['evaluation'].items():
        print(name, [(key, m['n'], m['micro_f1'], m['precision'], m['recall'], m['fp'], m['fn'])
                     for key, m in scores.items()])
    print('Results:', OUT / 'metrics.json', 'model:', artifact)


if __name__ == '__main__':
    main()
