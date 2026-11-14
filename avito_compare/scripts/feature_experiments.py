"""Run guarded TF-IDF ablations on the existing, immutable data splits."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


MAIN = [
    ('baseline', []),
    ('char_wb', ['--char-analyzer', 'char_wb']),
    ('uncapped_char', ['--no-max-features']),
    ('uncapped_char_wb', ['--char-analyzer', 'char_wb', '--no-max-features']),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rss_mib(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024
    except FileNotFoundError:
        pass
    return 0


def bootstrap(base_path, other_path):
    with np.load(base_path) as p:
        gold, old = p['gold'], p['predicted']
    with np.load(other_path) as p:
        if not np.array_equal(gold, p['gold']):
            raise ValueError('Evaluation labels differ')
        new = p['predicted']
    indices = np.random.default_rng(42).integers(0, len(gold), size=(2000, len(gold)))
    f = []
    for prediction in (old, new):
        per_row = np.column_stack([
            ((gold == 1) & (prediction == 1)).sum(axis=1),
            ((gold == 0) & (prediction == 1)).sum(axis=1),
            ((gold == 1) & (prediction == 0)).sum(axis=1),
        ])
        counts = per_row[indices].sum(axis=1)
        f.append(2 * counts[:, 0] / (2 * counts[:, 0] + counts[:, 1] + counts[:, 2]))
    return np.quantile(f[1] - f[0], [0.025, 0.975]).tolist()


def verify_reference(reference, measured, allowed_changes):
    if reference.get('exclude_source_category', False) != measured.get('exclude_source_category', False):
        raise ValueError('Different source-category evaluation policy')
    for key in ('split_sha256', 'thresholds', 'versions', 'estimator_parameters'):
        if measured[key] != reference[key]:
            raise ValueError(f'Invariant changed: {key}')
    for branch in ('word', 'char'):
        for key, value in reference['vectorizer_parameters'][branch].items():
            if (branch, key) not in allowed_changes and measured['vectorizer_parameters'][branch][key] != value:
                raise ValueError(f'Unexpected vectorizer change: {branch}.{key}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('output/ngram-comparison/splits'))
    parser.add_argument('--output-dir', type=Path, default=Path('output/tfidf-feature-experiments'))
    public_source = Path('datasets/public/public-train-annotated.jsonl')
    if not public_source.exists():
        public_source = Path('mvp/innoglobalhack_avito/innoglobalhack_avito/public-train-annotated.jsonl')
    parser.add_argument('--source', type=Path, default=public_source)
    parser.add_argument('--rss-limit-mib', type=float, default=2048)
    parser.add_argument('--timeout-seconds', type=float, default=180)
    parser.add_argument('--include-100k', action='store_true',
                        help='Add char/char_wb variants capped at 40000 word + 60000 char features')
    parser.add_argument('--include-60k', action='store_true',
                        help='Add char/char_wb variants capped at 24000 word + 36000 char features')
    parser.add_argument('--resume', action='store_true', help='Reuse completed measurements in this results directory')
    parser.add_argument('--char-wb-sizes', action='store_true', help='Only char_wb at 46k, 60k and 100k')
    parser.add_argument('--exclude-source-category', action='store_true')
    parser.add_argument('--skip-sublinear-check', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists() and not args.resume:
        raise ValueError('Use a fresh output directory')
    args.output_dir.mkdir(parents=True, exist_ok=args.resume)
    (args.output_dir / 'logs').mkdir(exist_ok=args.resume)
    source_hash = sha(args.source)
    split_metadata = json.loads((args.data_dir / 'stats.json').read_text())
    split_hashes = {name: sha(args.data_dir / f'{name}.jsonl') for name in ('train', 'calibration', 'validation')}
    results, statuses = {}, {}
    if args.resume:
        previous = json.loads((args.output_dir / 'summary.json').read_text())
        if previous['source_sha256'] != source_hash or previous['split_sha256'] != split_hashes:
            raise ValueError('Cannot resume on changed data')
        statuses.update(previous['status'])
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}

    def run(name, options):
        measurement_path = args.output_dir / name / 'measurement.json'
        if args.resume and measurement_path.exists():
            measured = json.loads(measurement_path.read_text())
            if measured['split_sha256'] != split_hashes:
                raise ValueError('Cannot reuse measurement with different splits')
            results[name] = measured
            print(f'REUSE {name}', flush=True)
            return True
        cmd = [sys.executable, '-m', 'avito_compare.scripts.measure_training',
               '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir / name),
               '--word-ngram-max', '3', *options]
        if args.exclude_source_category:
            cmd.append('--exclude-source-category')
        started, peak, stopped = time.monotonic(), 0.0, None
        print(f'START {name}', flush=True)
        with (args.output_dir / 'logs' / f'{name}.log').open('w') as log:
            process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            while process.poll() is None:
                peak = max(peak, rss_mib(process.pid))
                if peak > args.rss_limit_mib:
                    stopped = f'RSS exceeded {args.rss_limit_mib} MiB'
                elif time.monotonic() - started > args.timeout_seconds:
                    stopped = f'Wall time exceeded {args.timeout_seconds} seconds'
                if stopped:
                    process.kill()
                    process.wait()
                    break
                time.sleep(0.1)
        statuses[name] = {'status': 'complete' if process.returncode == 0 else 'stopped' if stopped else 'failed',
                          'reason': stopped, 'exit_code': process.returncode,
                          'monitored_peak_process_rss_mib': peak, 'command': cmd,
                          'process_wall_seconds': time.monotonic() - started}
        (args.output_dir / 'status.json').write_text(json.dumps(statuses, indent=2) + '\n')
        if process.returncode == 0:
            measured = json.loads((args.output_dir / name / 'measurement.json').read_text())
            if measured['split_sha256'] != split_hashes:
                raise ValueError('Split files changed')
            results[name] = measured
            print(f"DONE {name}: micro_f1={measured['micro_f1']:.6f} macro_f1={measured['macro_f1']:.6f} fit={measured['fit_wall_seconds']:.2f}s RSS={measured['fit_peak_process_rss_mib']:.1f}MiB dimensions={measured['dimensions']} FP={measured['confusion_counts']['fp']} FN={measured['confusion_counts']['fn']}", flush=True)
        else:
            print(f'STOP {name}: {statuses[name]}', flush=True)
        return process.returncode == 0

    experiments = MAIN + ([
        ('char_100k', ['--word-max-features', '40000', '--char-max-features', '60000']),
        ('char_wb_100k', ['--char-analyzer', 'char_wb', '--word-max-features', '40000', '--char-max-features', '60000']),
    ] if args.include_100k else [])
    if args.include_60k:
        experiments += [
            ('char_60k', ['--word-max-features', '24000', '--char-max-features', '36000']),
            ('char_wb_60k', ['--char-analyzer', 'char_wb', '--word-max-features', '24000', '--char-max-features', '36000']),
        ]
    if args.char_wb_sizes:
        experiments = [
            ('char_wb', ['--char-analyzer', 'char_wb']),
            ('char_wb_60k', ['--char-analyzer', 'char_wb', '--word-max-features', '24000', '--char-max-features', '36000']),
            ('char_wb_100k', ['--char-analyzer', 'char_wb', '--word-max-features', '40000', '--char-max-features', '60000']),
        ]
    baseline_name = 'char_wb' if args.char_wb_sizes else 'baseline'
    for name, options in experiments:
        if name == 'uncapped_char_wb' and statuses.get('uncapped_char', {}).get('status') == 'stopped':
            statuses[name] = {'status': 'skipped', 'reason': 'Prior uncapped configuration exceeded resource budget'}
            continue
        if not run(name, options) and name == baseline_name:
            raise RuntimeError('Baseline failed; inspect its log')
    base = results[baseline_name]
    allowed = {('char', 'analyzer'), ('word', 'max_features'), ('char', 'max_features')}
    for measured in results.values():
        verify_reference(base, measured, allowed)
    top = sorted(results, key=lambda name: (results[name]['micro_f1'], results[name]['macro_f1']), reverse=True)[:2]
    for name in ([] if args.skip_sublinear_check else top):
        # True is already the original setting: test the counterfactual False.
        options = dict(experiments)[name]
        ablation = name + '_linear_tf'
        if run(ablation, [*options, '--no-sublinear-tf']):
            verify_reference(results[name], results[ablation], {('word', 'sublinear_tf'), ('char', 'sublinear_tf')})
    if source_hash != sha(args.source):
        raise ValueError('Raw dataset changed')
    for name, expected in split_hashes.items():
        if sha(args.data_dir / f'{name}.jsonl') != expected:
            raise ValueError('Split dataset changed')

    summary = {'source_sha256': source_hash, 'split_sha256': split_hashes,
               'split_metadata': split_metadata,
               'exclude_source_category': args.exclude_source_category,
               'baseline_name': baseline_name,
               'limits': {'rss_mib': args.rss_limit_mib, 'process_wall_seconds': args.timeout_seconds},
               'main_experiments': [name for name, _ in experiments], 'top_two_by_validation_micro_f1': top,
               'status': statuses, 'measurements': results, 'deltas_from_baseline': {},
               'bootstrap_95_ci_micro_f1_difference_from_baseline': {},
               'checks': 'All measured variants have identical source/splits, thresholds, dependencies and estimator parameters; only requested TF-IDF parameters differ.',
               'sublinear_check_performed': not args.skip_sublinear_check}
    numeric = ['micro_f1', 'macro_f1', 'micro_precision', 'micro_recall', 'fit_wall_seconds', 'fit_peak_process_rss_mib', 'artifact_bytes']
    for name, measured in results.items():
        summary['deltas_from_baseline'][name] = {
            key: {'absolute': measured[key] - base[key], 'relative_percent': 100 * (measured[key] / base[key] - 1)} for key in numeric}
        for key in ('fp', 'fn'):
            old, new = base['confusion_counts'][key], measured['confusion_counts'][key]
            summary['deltas_from_baseline'][name][key] = {'absolute': new-old, 'relative_percent': 100*(new/old-1)}
        for key in ('word', 'char', 'total'):
            old, new = base['dimensions'][key], measured['dimensions'][key]
            summary['deltas_from_baseline'][name][f'{key}_dimensions'] = {'absolute': new-old, 'relative_percent': 100*(new/old-1)}
        if name != baseline_name:
            summary['bootstrap_95_ci_micro_f1_difference_from_baseline'][name] = bootstrap(
                args.output_dir / baseline_name / 'predictions.npz', args.output_dir / name / 'predictions.npz')
    summary['sublinear_true_minus_false'] = {}
    for name in top:
        other = name + '_linear_tf'
        if other in results:
            summary['sublinear_true_minus_false'][name] = {
                key: results[name][key] - results[other][key] for key in numeric}
            summary['sublinear_true_minus_false'][name]['micro_f1_95_ci'] = bootstrap(
                args.output_dir / other / 'predictions.npz', args.output_dir / name / 'predictions.npz')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    lines = ['# TF-IDF feature experiments', '',
             f"Fixed word ngrams (1,3); original sublinear_tf=True; existing immutable splits with split seed={split_metadata['seed']}; SGD random_state=42; thresholds=0.5; source category excluded={args.exclude_source_category}; one fit per variant; single-thread BLAS/OpenMP.",
             'Precision and recall are micro averages. Time includes vectorization and classifier fit. RSS is the absolute process high-water mark at fit completion, not an incremental allocation.', '',
             '| Variant | Micro-F1 | Macro-F1 | Precision | Recall | FP | FN | Fit s | Word dims | Char dims | Model MiB | Peak RSS MiB |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, measured in results.items():
        m = measured
        lines.append(f"| {name} | {m['micro_f1']:.6f} | {m['macro_f1']:.6f} | {m['micro_precision']:.6f} | {m['micro_recall']:.6f} | {m['confusion_counts']['fp']} | {m['confusion_counts']['fn']} | {m['fit_wall_seconds']:.3f} | {m['dimensions']['word']} | {m['dimensions']['char']} | {m['artifact_bytes']/1024**2:.3f} | {m['fit_peak_process_rss_mib']:.1f} |")
    lines += ['', '## Differences from baseline', '', '| Variant | Metric | Absolute difference | Relative difference % |', '|---|---|---:|---:|']
    for name, differences in summary['deltas_from_baseline'].items():
        if name != baseline_name:
            for key, diff in differences.items():
                lines.append(f"| {name} | {key} | {diff['absolute']:+.6f} | {diff['relative_percent']:+.3f} |")
    lines += ['', '## Paired bootstrap (whole ads, 2000 resamples, seed=42)', '']
    for name, ci in summary['bootstrap_95_ci_micro_f1_difference_from_baseline'].items():
        lines.append(f'- {name}: delta Micro-F1 95% interval [{ci[0]:+.6f}, {ci[1]:+.6f}].')
    lines += ['', f'Baseline: {baseline_name}. Ranking by held-out Micro-F1: {top}. Sublinear check performed: {not args.skip_sublinear_check}.',
              '', 'Dataset and supplied splits are unchanged; no resplitting. Exact normalized effective descriptions do not intersect across splits; near-duplicate templates can remain. This is exploratory comparison; the held-out set must not be treated as untouched after model selection. Timing is one run and sensitive to system load. Production artifacts were not replaced.',
              '', 'See summary.json for all exact measurements, configurations, deltas, source hashes and checks.']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:12]), flush=True)
    print('SUMMARY ' + str(args.output_dir / 'summary.json'), flush=True)


if __name__ == '__main__':
    main()
