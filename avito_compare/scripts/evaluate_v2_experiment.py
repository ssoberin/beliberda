"""Evaluate source-aware five-category predictions on v2 test and challenge."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

IDS = (102, 103, 104, 105, 106)


def metrics(gold, predicted):
    return {
        'rows': len(gold),
        'micro_f1': float(f1_score(gold, predicted, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(gold, predicted, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(gold, predicted, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(gold, predicted, average='micro', zero_division=0)),
        'exact_set_accuracy': float(accuracy_score(gold, predicted)),
        'fp': int(((gold == 0) & (predicted == 1)).sum()),
        'fn': int(((gold == 1) & (predicted == 0)).sum()),
        'f1_by_category': dict(zip(map(str, IDS), map(float, f1_score(gold, predicted, average=None, zero_division=0)))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=Path('datasets/hard_dataset/v2'))
    parser.add_argument('--runs', type=Path, default=Path('output/v2-charwb-comparison/runs'))
    parser.add_argument('--old-runs', type=Path, default=Path('output/tfidf-feature-experiments'))
    parser.add_argument('--output', type=Path, default=Path('output/v2-charwb-comparison/evaluation.json'))
    args = parser.parse_args()
    results = {'evaluation_scope': list(IDS), 'source_category_excluded': True,
               'source_commit': '0b1be5d', 'data_sha256': {}, 'models': {}}
    data = {}
    for split in ('test', 'challenge'):
        path = args.dataset / f'{split}.jsonl'
        results['data_sha256'][split] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        gold = np.asarray([[int(mid in {c['mcId'] for c in row['found_microcategories']}) for mid in IDS] for row in rows], dtype=np.int8)
        data[split] = rows, gold
    for prefix, root in (('v2_trained', args.runs), ('public_trained', args.old_runs)):
        for name in ('char_wb', 'char_wb_60k', 'char_wb_100k'):
            model = joblib.load(root / name / 'tfidf_baseline.joblib')
            if tuple(model['ids']) != IDS or not np.array_equal(model['thresholds'], np.full(5, 0.5)):
                raise ValueError('Unexpected category heads or thresholds')
            scored = {}
            for split, (rows, gold) in data.items():
                prediction = (model['pipeline'].predict_proba([row['description'][:6000] for row in rows]) >= model['thresholds']).astype(np.int8)
                for index, row in enumerate(rows):
                    if row['mcId'] in IDS:
                        prediction[index, IDS.index(row['mcId'])] = 0
                split_metrics = metrics(gold, prediction)
                slices = {}
                masks = {'source_101': np.asarray([row['mcId'] == 101 for row in rows]),
                         'empty_targets': gold.sum(axis=1) == 0}
                for field in ('augmentation', 'topic', 'negation_depth'):
                    values = [str(row.get(field, 'unannotated')) for row in rows]
                    for value in sorted(set(values)):
                        masks[f'{field}={value}'] = np.asarray([v == value for v in values])
                for label, mask in masks.items():
                    if mask.any():
                        slices[label] = metrics(gold[mask], prediction[mask])
                split_metrics['slices'] = slices
                scored[split] = split_metrics
                if prefix == 'v2_trained' and split == 'test':
                    saved = json.loads((root / name / 'measurement.json').read_text())
                    if any(saved[key] != split_metrics[key] for key in ('micro_f1', 'macro_f1', 'micro_precision', 'micro_recall')):
                        raise ValueError('Independent evaluation does not match training measurement')
            results['models'][f'{prefix}/{name}'] = scored
            print(f"{prefix}/{name}: test={scored['test']['micro_f1']:.6f}, challenge={scored['challenge']['micro_f1']:.6f}", flush=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
