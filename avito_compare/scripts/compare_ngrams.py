"""Verify and summarize paired bigram/trigram training measurements."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('output/ngram-comparison'))
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    measurements, artifacts, predictions = [], [], []
    for name in ('bigrams', 'trigrams'):
        folder = args.results / name
        measurements.append(json.loads((folder / 'measurement.json').read_text()))
        artifacts.append(joblib.load(folder / 'tfidf_baseline.joblib'))
        with np.load(folder / 'predictions.npz') as p:
            predictions.append({key: p[key] for key in p.files})
    a, b = measurements
    assert a['split_sha256'] == b['split_sha256'], 'Different splits'
    assert a['thresholds'] == b['thresholds'] == [0.5] * 5, 'Different thresholds'
    assert a['versions'] == b['versions'], 'Different dependencies'
    assert np.array_equal(predictions[0]['gold'], predictions[1]['gold']), 'Different evaluation labels'
    vectors = [dict(m['pipeline'].named_steps['vectorizer'].transformer_list) for m in artifacts]
    word_params = [v['word'].get_params() for v in vectors]
    assert word_params[0].pop('ngram_range') == (1, 2)
    assert word_params[1].pop('ngram_range') == (1, 3)
    assert word_params[0] == word_params[1], 'Other word parameters changed'
    assert vectors[0]['char'].get_params() == vectors[1]['char'].get_params(), 'Character parameters changed'
    assert vectors[0]['char'].vocabulary_ == vectors[1]['char'].vocabulary_, 'Character vocabulary changed'
    assert np.array_equal(vectors[0]['char'].idf_, vectors[1]['char'].idf_), 'Character IDF changed'
    classifiers = [m['pipeline'].named_steps['classifier'].estimator.get_params() for m in artifacts]
    assert classifiers[0] == classifiers[1], 'Classifier parameters changed'

    # Paired bootstrap resamples entire ads, preserving correlation of their labels.
    gold = predictions[0]['gold']
    indices = np.random.default_rng(42).integers(0, len(gold), size=(2000, len(gold)))
    boot = []
    for p in predictions:
        yp = p['predicted']
        counts = np.column_stack([
            ((gold == 1) & (yp == 1)).sum(axis=1),
            ((gold == 0) & (yp == 1)).sum(axis=1),
            ((gold == 1) & (yp == 0)).sum(axis=1),
        ])
        totals = counts[indices].sum(axis=1)
        boot.append(2 * totals[:, 0] / (2 * totals[:, 0] + totals[:, 1] + totals[:, 2]))
    ci = np.quantile(boot[1] - boot[0], [0.025, 0.975]).tolist()
    if b['micro_f1'] <= a['micro_f1']:
        recommendation = 'Micro-F1 did not improve. Keep bigrams as the default; trigram support remains opt-in.'
    elif ci[0] <= 0 <= ci[1]:
        recommendation = 'The observed gain is small and its interval includes zero. Keep bigrams as the default; trigram support remains opt-in.'
    else:
        recommendation = 'The paired interval supports an improvement on this split. Confirm on independent labels before changing the production default.'

    rows = [
        ('Micro-F1', a['micro_f1'], b['micro_f1']),
        ('Macro-F1', a['macro_f1'], b['macro_f1']),
        ('Precision (micro)', a['micro_precision'], b['micro_precision']),
        ('Recall (micro)', a['micro_recall'], b['micro_recall']),
        ('Training wall time, seconds', a['fit_wall_seconds'], b['fit_wall_seconds']),
        ('Training CPU time, seconds', a['fit_cpu_seconds'], b['fit_cpu_seconds']),
        ('TF-IDF dimensions', a['dimensions']['total'], b['dimensions']['total']),
        ('Artifact bytes', a['artifact_bytes'], b['artifact_bytes']),
        ('False positives', a['confusion_counts']['fp'], b['confusion_counts']['fp']),
        ('False negatives', a['confusion_counts']['fn'], b['confusion_counts']['fn']),
    ]
    delta = [{'metric': name, 'bigrams': old, 'trigrams': new, 'absolute_difference': new-old,
              'relative_difference_percent': 100*(new-old)/old if old else None} for name, old, new in rows]
    result = {
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'split_sha256': a['split_sha256'], 'comparison': delta,
        'paired_bootstrap_micro_f1_difference_95_ci': ci,
        'bootstrap_resamples': 2000, 'bootstrap_seed': 42,
        'changed_label_decisions': int((predictions[0]['predicted'] != predictions[1]['predicted']).sum()),
        'word_vocabulary_sizes': [len(v['word'].vocabulary_) for v in vectors],
        'trigrams_in_new_word_vocabulary': sum(len(token.split()) == 3 for token in vectors[1]['word'].vocabulary_),
        'checks': {'same_split_hashes': True, 'same_thresholds': True, 'same_classifier_parameters': True,
                   'same_character_parameters_vocabulary_and_idf': True, 'only_word_ngram_parameter_changed': True},
        'versions': a['versions'],
        'recommendation': recommendation,
    }
    split_stats = json.loads((args.results / 'splits' / 'stats.json').read_text())
    lines = [
        '# Word n-gram comparison', '',
        f"Source: {args.source.name}. Existing normalized deduplication and seed={split_stats['seed']} split: {split_stats['splits']}.",
        'Evaluation: validation only; fixed 0.5 thresholds for all five labels. Individual tuned thresholds in the trainer metrics are NOT used here.',
        'Training timing: one fit per variant, same environment, BLAS/OpenMP threads=1; includes TF-IDF and SGD training, excludes threshold search and artifact serialization.', '',
        '| Metric | Bigrams (1,2) | Trigrams (1,3) | Absolute difference | Relative difference |',
        '|---|---:|---:|---:|---:|',
    ]
    for row in delta:
        lines.append(f"| {row['metric']} | {row['bigrams']:.6f} | {row['trigrams']:.6f} | {row['absolute_difference']:+.6f} | {row['relative_difference_percent']:+.4f}% |")
    lines += ['', '## Errors by category', '', '| mcId | FP bigrams | FP trigrams | Delta FP | FN bigrams | FN trigrams | Delta FN |', '|---|---:|---:|---:|---:|---:|---:|']
    for mid in ('102', '103', '104', '105', '106'):
        old, new = a['confusion_counts_by_category'], b['confusion_counts_by_category']
        lines.append(f"| {mid} | {old['fp'][mid]} | {new['fp'][mid]} | {new['fp'][mid]-old['fp'][mid]:+d} | {old['fn'][mid]} | {new['fn'][mid]} | {new['fn'][mid]-old['fn'][mid]:+d} |")
    lines += [
        '', f"Paired bootstrap 95% interval for Micro-F1 difference: [{ci[0]:+.6f}, {ci[1]:+.6f}] (2000 resamples of whole ads).",
        f"Selected word trigrams: {result['trigrams_in_new_word_vocabulary']}; unchanged max_features means trigrams compete with unigrams and bigrams.",
        '', recommendation,
        'Weak automated labels and one split do not establish performance on new distributions. Normalized exact descriptions are disjoint; similar templates can remain across splits. The source dataset and split algorithm were not changed.',
        'Both artifacts are experimental copies. Production model paths were not touched.',
        '', 'See comparison.json for source/split hashes, parameter checks and exact measurements.',
    ]
    (args.results / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (args.results / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
