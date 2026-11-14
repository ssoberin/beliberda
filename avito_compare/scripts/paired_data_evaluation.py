"""Compare matched public-only and mixed-trained models on identical test sets."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import joblib
import numpy as np

from .evaluate_v2_experiment import IDS, metrics
from .prepare_dataset import normalize

LABELS = {'Штукатурные работы': 102, 'Плиточные работы': 103, 'Сантехника': 104,
          'Электрика': 105, 'Поклейка обоев и малярные работы': 106}


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row):
    return normalize(row['description'][:6000])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def confidence(gold, old, new, groups):
    # Resample source descriptions together, including contradictory duplicates.
    locations = defaultdict(list)
    for i, group in enumerate(groups):
        locations[group].append(i)
    indices = list(locations.values())
    per_group = []
    for pred in (old, new):
        row_counts = np.column_stack([((gold == 1) & (pred == 1)).sum(1),
                                     ((gold == 0) & (pred == 1)).sum(1),
                                     ((gold == 1) & (pred == 0)).sum(1)])
        per_group.append(np.asarray([row_counts[index].sum(0) for index in indices]))
    sample = np.random.default_rng(42).integers(0, len(indices), size=(2000, len(indices)))
    fs = []
    for counts in per_group:
        total = counts[sample].sum(1)
        denominator = 2*total[:, 0] + total[:, 1] + total[:, 2]
        fs.append(np.divide(2*total[:, 0], denominator, out=np.zeros(len(total), dtype=float), where=denominator != 0))
    return np.quantile(fs[1] - fs[0], [0.025, 0.975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/paired-data-evaluation'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    args.output.mkdir(parents=True)
    public_splits = Path('output/ngram-comparison/splits')
    mixed_splits = Path('output/mixed-charwb-comparison/splits')
    parquet = Path('datasets/public/public-test-dataset.parquet')
    if not parquet.exists():
        parquet = Path('mvp/innoglobalhack_avito/innoglobalhack_avito/public-test-dataset.parquet')
    raw = duckdb.connect().execute('SELECT title, description, categories FROM read_parquet(?)', [str(parquet)]).fetchall()
    official = [{'row_index_zero_based': i, 'title': title, 'description': description, 'mcId': 101,
                 'evaluation_labels': sorted(LABELS[c] for c in categories)} for i, (title, description, categories) in enumerate(raw)]
    label_groups = defaultdict(set)
    for row in official:
        label_groups[key(row)].add(tuple(row['evaluation_labels']))
    conflicts = {text for text, labels in label_groups.items() if len(labels) > 1}
    train_keys = {key(row) for row in read(mixed_splits/'train.jsonl')}
    all_public_keys = {key(row) for name in ('train', 'calibration', 'validation') for row in read(public_splits/f'{name}.jsonl')}
    calibration_keys = {key(row) for row in read(mixed_splits/'calibration.jsonl')}
    strict_excluded = all_public_keys | train_keys | calibration_keys
    eligible = [row for row in official if key(row) not in conflicts and key(row) not in strict_excluded]
    clean = list({key(row): row for row in eligible}.values())
    public_holdout = [dict(row, mcId=101, evaluation_labels=row['labels']) for row in read(public_splits/'validation.jsonl')]
    v2_test = [row for row in read(mixed_splits/'validation.jsonl') if row['dataset'] == 'v2']
    challenge = read(mixed_splits/'challenge.jsonl')
    tests = {'official_public_208': official, 'official_public_clean_unique': clean,
             'public_holdout_729': public_holdout, 'v2_test_3016': v2_test, 'challenge_40': challenge}
    paths = {'public_only': Path('output/tfidf-feature-experiments/char_wb'),
             'public_plus_v2': Path('output/mixed-charwb-comparison/runs/char_wb')}
    measurements = {name: json.loads((path/'measurement.json').read_text()) for name, path in paths.items()}
    for parameter in ('vectorizer_parameters', 'estimator_parameters', 'thresholds', 'versions'):
        if measurements['public_only'][parameter] != measurements['public_plus_v2'][parameter]:
            raise ValueError(f'Unmatched experiment: {parameter}')
    artifacts = {name: path/'tfidf_baseline.joblib' for name, path in paths.items()}
    hashes = {name: digest(path) for name, path in artifacts.items()}
    models = {name: joblib.load(path) for name, path in artifacts.items()}
    result = {'model_artifact_sha256': hashes,
              'model_paths': {name: str(path) for name, path in artifacts.items()},
              'official_parquet_sha256': digest(parquet),
              'scope': list(IDS), 'thresholds': [0.5]*5,
              'official_source_mcId_assumption': 101,
              'official_audit': {'rows': len(official), 'unique_normalized_descriptions': len(label_groups),
                 'conflicting_description_groups': len(conflicts),
                 'rows_in_conflicting_groups': sum(key(row) in conflicts for row in official),
                 'rows_overlapping_actual_mixed_train': sum(key(row) in train_keys for row in official),
                 'rows_overlapping_any_public_split': sum(key(row) in all_public_keys for row in official),
                 'clean_eligible_rows_before_dedup': len(eligible), 'clean_unique_rows': len(clean),
                 'clean_positive_rows': sum(bool(row['evaluation_labels']) for row in clean),
                 'policy': 'Clean subset excludes every conflicting group and all text seen in any public split or mixed train/calibration; then deduplicates consistent labels. No labels corrected.'},
              'comparisons': {}}
    for name, rows in tests.items():
        gold = np.asarray([[int(mid in row['evaluation_labels']) for mid in IDS] for row in rows], dtype=np.int8)
        predicted = {}
        scored = {}
        for model_name, model in models.items():
            if tuple(model['ids']) != IDS or not np.array_equal(model['thresholds'], np.full(5, 0.5)):
                raise ValueError('Unexpected heads or thresholds')
            yp = (model['pipeline'].predict_proba([row['description'][:6000] for row in rows]) >= model['thresholds']).astype(np.int8)
            for i, row in enumerate(rows):
                if row['mcId'] in IDS:
                    yp[i, IDS.index(row['mcId'])] = 0
            predicted[model_name] = yp
            scored[model_name] = metrics(gold, yp)
        old, new = scored['public_only'], scored['public_plus_v2']
        ci = confidence(gold, predicted['public_only'], predicted['public_plus_v2'],
                        [row.get('group_id', key(row)) for row in rows])
        scored['delta_new_minus_old'] = {k: new[k]-old[k] for k in ('micro_f1','macro_f1','micro_precision','micro_recall','fp','fn','exact_set_accuracy')}
        scored['paired_micro_f1_delta_95_ci'] = ci
        scored['new_fixes_label_decisions'] = int(((predicted['public_only'] != gold) & (predicted['public_plus_v2'] == gold)).sum())
        scored['new_breaks_label_decisions'] = int(((predicted['public_only'] == gold) & (predicted['public_plus_v2'] != gold)).sum())
        result['comparisons'][name] = scored
        np.savez_compressed(args.output/f'{name}.npz', gold=gold, **predicted)
        print(f"{name} n={len(rows)}: old={old['micro_f1']:.6f} new={new['micro_f1']:.6f} delta={new['micro_f1']-old['micro_f1']:+.6f} CI={ci}", flush=True)
    if hashes != {name: digest(path) for name, path in artifacts.items()}:
        raise ValueError('Model artifacts changed')
    (args.output/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Парное сравнение: public против public + v2', '',
             'Одинаковая архитектура: word(1,3), char_wb(3,5), 18000+28000 признаков, sublinear_tf=True, пять SGD с random_state42, пороги0.5. Различаются только обучающие данные. Использованы сохранённые экспериментальные артефакты, не проверенные серверные production-веса; повторное обучение не выполнялось.',
             'public_only обучена на3398 уникальных public train; public_plus_v2 — на тех же3398 и8464 v2 train. Для оценки источник каждого объявления исключается, scope102–106. Для Parquet принято mcId101.', '',
             '| Проверка | Строк | F1 old | F1 new | Δ, п.п. | FP old→new | FN old→new | Exact old→new |',
             '|---|---:|---:|---:|---:|---|---|---|']
    for name, scored in result['comparisons'].items():
        old,new=scored['public_only'],scored['public_plus_v2']
        lines.append(f"| {name} | {old['rows']} | {old['micro_f1']:.6f} | {new['micro_f1']:.6f} | {100*(new['micro_f1']-old['micro_f1']):+.3f} | {old['fp']}→{new['fp']} | {old['fn']}→{new['fn']} | {old['exact_set_accuracy']:.4f}→{new['exact_set_accuracy']:.4f} |")
    lines += ['', '## Ограничения публичного теста', '', json.dumps(result['official_audit'],ensure_ascii=False,indent=2),
              '', 'Полный Parquet сохраняет исходные противоречия и повторы. Очищенная часть исключает их без исправления меток, поэтому меньше и может быть смещена по составу. Показываются обе оценки. Public holdout и v2 test уже использовались в прошлых экспериментах; это сравнительная диагностика, не новый независимый финальный тест.',
              'Confidence intervals: paired bootstrap2000, seed42; нормализованные описания группируются вместе. Для v2 prepared записей group_id не перенесён, поэтому интервал по строкам может недооценивать зависимость между аугментациями. Полные метрики и исправленные/сломанные решения сохранены вcomparison.json.',
              'Рабочие модели и исходные данные не изменялись. Улучшение сложного теста и отсутствие деградации на реальных данных — отдельные критерии; общий смешанный F1 не заменяет их.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
