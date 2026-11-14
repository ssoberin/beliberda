"""Measure the existing trainer without changing its estimator or thresholds."""
import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from . import prepare_dataset, train_baseline


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--word-ngram-max', type=int, choices=(2, 3), default=2)
    parser.add_argument('--char-analyzer', choices=('char', 'char_wb'), default='char')
    parser.add_argument('--no-max-features', action='store_true')
    parser.add_argument('--word-max-features', type=int, default=18000)
    parser.add_argument('--char-max-features', type=int, default=28000)
    parser.add_argument('--sublinear-tf', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--exclude-source-category', action='store_true',
                        help='Evaluate offered labels after removing each row source mcId')
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('Use a new output directory to preserve existing artifacts')

    # Check the effective input after the trainer's 6000-character truncation.
    splits = {}
    for name in ('train', 'calibration', 'validation'):
        texts, _ = train_baseline.load(args.data_dir / f'{name}.jsonl')
        splits[name] = set(map(prepare_dataset.normalize, texts))
        if len(splits[name]) != len(texts):
            raise ValueError(f'Duplicate effective descriptions within {name}')
    for a, b in (('train', 'calibration'), ('train', 'validation'), ('calibration', 'validation')):
        if splits[a] & splits[b]:
            raise ValueError(f'Data leakage between {a} and {b}')

    fit_times = []
    fit_cpu = []
    fit_peak_rss = []
    original_fit = Pipeline.fit

    def measured_fit(self, *fit_args, **fit_kwargs):
        start, cpu_start = time.perf_counter(), time.process_time()
        result = original_fit(self, *fit_args, **fit_kwargs)
        fit_times.append(time.perf_counter() - start)
        fit_cpu.append(time.process_time() - cpu_start)
        fit_peak_rss.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 if sys.platform != 'darwin' else 1024**2))
        return result

    argv = ['train_baseline', '--data-dir', str(args.data_dir), '--output-dir', str(args.output_dir)]
    # The original, unchanged trainer is usable for the baseline measurement.
    if args.word_ngram_max != 2:
        argv += ['--word-ngram-max', str(args.word_ngram_max)]
    if args.char_analyzer != 'char':
        argv += ['--char-analyzer', args.char_analyzer]
    if args.no_max_features:
        argv += ['--no-max-features']
    if args.word_max_features != 18000:
        argv += ['--word-max-features', str(args.word_max_features)]
    if args.char_max_features != 28000:
        argv += ['--char-max-features', str(args.char_max_features)]
    if not args.sublinear_tf:
        argv += ['--no-sublinear-tf']
    with patch.object(sys, 'argv', argv), patch.object(Pipeline, 'fit', measured_fit):
        train_baseline.main()
    if len(fit_times) != 1:
        raise ValueError(f'Expected one pipeline fit, got {len(fit_times)}')

    artifact = args.output_dir / 'tfidf_baseline.joblib'
    model = joblib.load(artifact)
    pipeline = model['pipeline']
    thresholds = model['thresholds']
    if not np.array_equal(thresholds, np.full(5, 0.5)):
        raise ValueError('Comparison requires fixed 0.5 thresholds')
    x, y = train_baseline.load(args.data_dir / 'validation.jsonl')
    predicted = (pipeline.predict_proba(x) >= thresholds).astype(np.int8)
    if args.exclude_source_category:
        eval_rows = [json.loads(line) for line in (args.data_dir / 'validation.jsonl').read_text().splitlines() if line.strip()]
        for index, row in enumerate(eval_rows):
            source_id = row['mcId']
            if source_id in train_baseline.IDS:
                column = train_baseline.IDS.index(source_id)
                predicted[index, column] = 0
                y[index, column] = 0
            if set(np.asarray(train_baseline.IDS)[y[index].astype(bool)]) != set(row['evaluation_labels']):
                raise ValueError('Offered-minus-source does not match target labels')
    transformers = dict(pipeline.named_steps['vectorizer'].transformer_list)
    if transformers['word'].ngram_range != (1, args.word_ngram_max):
        raise ValueError('Unexpected word n-gram range')
    if transformers['char'].analyzer != args.char_analyzer:
        raise ValueError('Unexpected character analyzer')
    expected_caps = (None, None) if args.no_max_features else (args.word_max_features, args.char_max_features)
    if tuple(transformers[name].max_features for name in ('word', 'char')) != expected_caps:
        raise ValueError('Unexpected vocabulary caps')
    if any(v.sublinear_tf != args.sublinear_tf for v in transformers.values()):
        raise ValueError('Unexpected sublinear TF setting')
    dimensions = {name: len(vectorizer.vocabulary_) for name, vectorizer in transformers.items()}
    counts = {}
    per_label = {}
    for key, mask in (
        ('tp', (y == 1) & (predicted == 1)),
        ('fp', (y == 0) & (predicted == 1)),
        ('fn', (y == 1) & (predicted == 0)),
        ('tn', (y == 0) & (predicted == 0)),
    ):
        counts[key] = int(mask.sum())
        per_label[key] = dict(zip(map(str, train_baseline.IDS), map(int, mask.sum(axis=0))))
    result = {
        'word_ngram_range': [1, args.word_ngram_max],
        'char_analyzer': args.char_analyzer,
        'max_features': dict(zip(('word', 'char'), expected_caps)),
        'sublinear_tf': args.sublinear_tf,
        'exclude_source_category': args.exclude_source_category,
        'thresholds': thresholds.tolist(),
        'micro_f1': float(f1_score(y, predicted, average='micro', zero_division=0)),
        'macro_f1': float(f1_score(y, predicted, average='macro', zero_division=0)),
        'micro_precision': float(precision_score(y, predicted, average='micro', zero_division=0)),
        'micro_recall': float(recall_score(y, predicted, average='micro', zero_division=0)),
        'fit_wall_seconds': fit_times[0], 'fit_cpu_seconds': fit_cpu[0],
        'fit_peak_process_rss_mib': fit_peak_rss[0],
        'dimensions': {**dimensions, 'total': sum(dimensions.values())},
        'artifact_bytes': artifact.stat().st_size,
        'confusion_counts': counts, 'confusion_counts_by_category': per_label,
        'f1_by_category': dict(zip(map(str, train_baseline.IDS), map(float, f1_score(y, predicted, average=None, zero_division=0)))),
        'split_sha256': {name: fingerprint(args.data_dir / f'{name}.jsonl') for name in splits},
        'normalized_effective_split_intersections': 0,
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'estimator_parameters': pipeline.named_steps['classifier'].estimator.get_params(),
        'vectorizer_parameters': {name: {key: str(value) if isinstance(value, type) else value for key, value in v.get_params().items()} for name, v in transformers.items()},
        'note': 'Fixed-threshold validation on weak labels. Near-duplicate templates may remain. Fit time includes vectorization and five classifiers, excludes threshold search and serialization.',
    }
    (args.output_dir / 'measurement.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    np.savez_compressed(args.output_dir / 'predictions.npz', gold=y, predicted=predicted)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
