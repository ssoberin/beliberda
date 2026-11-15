"""Check frozen hashes, OOF grouping, CSV/JSON agreement, and notebook execution."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P = Path('output/catboost-optimization')

def read(name):
    return json.loads((P/name).read_text())

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    final = read('final_metrics.json')
    predictions = pd.read_csv(P/'final_predictions.csv')
    excluded = set(read('public_test_exclusions.json')['excluded_row_indices_zero_based'])
    checked = 0
    for name, result in final.items():
        for dataset, metrics in result['metrics'].items():
            if dataset == 'public_complex_multiservice':
                continue
            base_dataset = 'official_208' if dataset == 'public_test_clean' else dataset
            rows = predictions[(predictions.model == name) & (predictions.dataset == base_dataset)]
            if dataset == 'public_test_clean':
                rows = rows[~rows.row_idx.isin(excluded)]
            assert rows.row_idx.nunique() == metrics['rows'], (name, dataset, 'rows')
            assert not rows.duplicated(['row_idx', 'cat_id']).any()
            gold, pred = rows.gold.to_numpy(), rows.prediction.to_numpy()
            tp = int(((gold == 1) & (pred == 1)).sum())
            fp = int(((gold == 0) & (pred == 1)).sum())
            fn = int(((gold == 1) & (pred == 0)).sum())
            assert (tp, fp, fn) == (metrics['tp'], metrics['fp'], metrics['fn'])
            assert abs(2*tp/(2*tp+fp+fn) - metrics['micro_f1']) < 1e-6
            exact = pd.Series(gold == pred, index=rows.row_idx).groupby(level=0).all().mean()
            assert abs(exact - metrics['exact_match']) < 1e-6
            checked += 1
    for path, expected in read('input_manifest.json')['source_sha256'].items():
        assert sha(path) == expected, path
    for name, expected in read('finalists.json')['model_sha256'].items():
        assert sha(P/'models'/f'{name}.cbm') == expected, name
    groups = np.load(P/'cache/original_oof.npz')['groups']
    holdouts = []
    for fold in read('tfidf_oof_folds.json'):
        train, val = fold['train'], fold['holdout']
        assert not set(groups[train]) & set(groups[val])
        assert len(train) + len(val) == 6319
        holdouts.extend(val)
    assert sorted(holdouts) == list(range(6319))
    notebook = read('results.ipynb')
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    assert all(c['execution_count'] is not None for c in cells)
    assert all(o['output_type'] != 'error' for c in cells for o in c['outputs'])
    verification = read('verification.json')
    verification.update({
        'csv_json_metric_comparisons_passed': checked,
        'tfidf_oof_no_group_overlap': True,
        'tfidf_each_train_row_has_one_oof_prediction': True,
        'executed_notebook_no_errors': True,
    })
    (P/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps(verification, indent=2))

if __name__ == '__main__':
    main()
