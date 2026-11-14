# Word n-gram comparison

Source: public-train-annotated.jsonl. Existing normalized deduplication and seed=42 split: {'train': 3398, 'calibration': 728, 'validation': 729}.
Evaluation: validation only; fixed 0.5 thresholds for all five labels. Individual tuned thresholds in the trainer metrics are NOT used here.
Training timing: one fit per variant, same environment, BLAS/OpenMP threads=1; includes TF-IDF and SGD training, excludes threshold search and artifact serialization.

| Metric | Bigrams (1,2) | Trigrams (1,3) | Absolute difference | Relative difference |
|---|---:|---:|---:|---:|
| Micro-F1 | 0.934646 | 0.936217 | +0.001571 | +0.1681% |
| Macro-F1 | 0.934682 | 0.936225 | +0.001543 | +0.1651% |
| Precision (micro) | 0.936438 | 0.939044 | +0.002606 | +0.2783% |
| Recall (micro) | 0.932860 | 0.933406 | +0.000546 | +0.0585% |
| Training wall time, seconds | 7.801622 | 8.210605 | +0.408984 | +5.2423% |
| Training CPU time, seconds | 7.787722 | 8.209601 | +0.421879 | +5.4172% |
| TF-IDF dimensions | 46000.000000 | 46000.000000 | +0.000000 | +0.0000% |
| Artifact bytes | 2777562.000000 | 2886202.000000 | +108640.000000 | +3.9113% |
| False positives | 116.000000 | 111.000000 | -5.000000 | -4.3103% |
| False negatives | 123.000000 | 122.000000 | -1.000000 | -0.8130% |

## Errors by category

| mcId | FP bigrams | FP trigrams | Delta FP | FN bigrams | FN trigrams | Delta FN |
|---|---:|---:|---:|---:|---:|---:|
| 102 | 28 | 28 | +0 | 25 | 22 | -3 |
| 103 | 35 | 31 | -4 | 20 | 20 | +0 |
| 104 | 21 | 20 | -1 | 26 | 28 | +2 |
| 105 | 17 | 17 | +0 | 19 | 19 | +0 |
| 106 | 15 | 15 | +0 | 33 | 33 | +0 |

Paired bootstrap 95% interval for Micro-F1 difference: [-0.000651, +0.003758] (2000 resamples of whole ads).
Selected word trigrams: 6728; unchanged max_features means trigrams compete with unigrams and bigrams.

The observed gain is small and its interval includes zero. Keep bigrams as the default; trigram support remains opt-in.
Weak automated labels and one split do not establish performance on new distributions. Normalized exact descriptions are disjoint; similar templates can remain across splits. The source dataset and split algorithm were not changed.
Both artifacts are experimental copies. Production model paths were not touched.

See comparison.json for source/split hashes, parameter checks and exact measurements.
