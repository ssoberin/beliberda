# TF-IDF feature experiments

Fixed word ngrams (1,3); original sublinear_tf=True; existing immutable splits with split seed=2026; SGD random_state=42; thresholds=0.5; source category excluded=True; one fit per variant; single-thread BLAS/OpenMP.
Precision and recall are micro averages. Time includes vectorization and classifier fit. RSS is the absolute process high-water mark at fit completion, not an incremental allocation.

| Variant | Micro-F1 | Macro-F1 | Precision | Recall | FP | FN | Fit s | Word dims | Char dims | Model MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| char_wb | 0.460813 | 0.461267 | 0.663274 | 0.353047 | 794 | 2866 | 1.729 | 6483 | 4209 | 0.747 | 231.3 |
| char_wb_60k | 0.460813 | 0.461267 | 0.663274 | 0.353047 | 794 | 2866 | 1.613 | 6483 | 4209 | 0.747 | 231.9 |
| char_wb_100k | 0.460813 | 0.461267 | 0.663274 | 0.353047 | 794 | 2866 | 1.561 | 6483 | 4209 | 0.747 | 230.0 |

## Differences from baseline

| Variant | Metric | Absolute difference | Relative difference % |
|---|---|---:|---:|
| char_wb_60k | micro_f1 | +0.000000 | +0.000 |
| char_wb_60k | macro_f1 | +0.000000 | +0.000 |
| char_wb_60k | micro_precision | +0.000000 | +0.000 |
| char_wb_60k | micro_recall | +0.000000 | +0.000 |
| char_wb_60k | fit_wall_seconds | -0.115707 | -6.694 |
| char_wb_60k | fit_peak_process_rss_mib | +0.617188 | +0.267 |
| char_wb_60k | artifact_bytes | +0.000000 | +0.000 |
| char_wb_60k | fp | +0.000000 | +0.000 |
| char_wb_60k | fn | +0.000000 | +0.000 |
| char_wb_60k | word_dimensions | +0.000000 | +0.000 |
| char_wb_60k | char_dimensions | +0.000000 | +0.000 |
| char_wb_60k | total_dimensions | +0.000000 | +0.000 |
| char_wb_100k | micro_f1 | +0.000000 | +0.000 |
| char_wb_100k | macro_f1 | +0.000000 | +0.000 |
| char_wb_100k | micro_precision | +0.000000 | +0.000 |
| char_wb_100k | micro_recall | +0.000000 | +0.000 |
| char_wb_100k | fit_wall_seconds | -0.167408 | -9.685 |
| char_wb_100k | fit_peak_process_rss_mib | -1.339844 | -0.579 |
| char_wb_100k | artifact_bytes | +0.000000 | +0.000 |
| char_wb_100k | fp | +0.000000 | +0.000 |
| char_wb_100k | fn | +0.000000 | +0.000 |
| char_wb_100k | word_dimensions | +0.000000 | +0.000 |
| char_wb_100k | char_dimensions | +0.000000 | +0.000 |
| char_wb_100k | total_dimensions | +0.000000 | +0.000 |

## Paired bootstrap (whole ads, 2000 resamples, seed=42)

- char_wb_60k: delta Micro-F1 95% interval [+0.000000, +0.000000].
- char_wb_100k: delta Micro-F1 95% interval [+0.000000, +0.000000].

Baseline: char_wb. Ranking by held-out Micro-F1: ['char_wb', 'char_wb_60k']. Sublinear check performed: False.

Dataset and supplied splits are unchanged; no resplitting. Exact normalized effective descriptions do not intersect across splits; near-duplicate templates can remain. This is exploratory comparison; the held-out set must not be treated as untouched after model selection. Timing is one run and sensitive to system load. Production artifacts were not replaced.

See summary.json for all exact measurements, configurations, deltas, source hashes and checks.
