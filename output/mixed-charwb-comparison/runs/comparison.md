# TF-IDF feature experiments

Fixed word ngrams (1,3); original sublinear_tf=True; existing immutable splits with split seed={'public_split': 42, 'v2_supplied_split': 2026}; SGD random_state=42; thresholds=0.5; source category excluded=True; one fit per variant; single-thread BLAS/OpenMP.
Precision and recall are micro averages. Time includes vectorization and classifier fit. RSS is the absolute process high-water mark at fit completion, not an incremental allocation.

| Variant | Micro-F1 | Macro-F1 | Precision | Recall | FP | FN | Fit s | Word dims | Char dims | Model MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| char_wb | 0.619331 | 0.619556 | 0.795540 | 0.507027 | 816 | 3087 | 7.943 | 18000 | 28000 | 2.768 | 586.9 |
| char_wb_60k | 0.617829 | 0.617992 | 0.800508 | 0.503034 | 785 | 3112 | 8.080 | 24000 | 36000 | 3.633 | 590.2 |
| char_wb_100k | 0.616570 | 0.616725 | 0.807762 | 0.498563 | 743 | 3140 | 8.058 | 40000 | 60000 | 6.093 | 572.8 |

## Differences from baseline

| Variant | Metric | Absolute difference | Relative difference % |
|---|---|---:|---:|
| char_wb_60k | micro_f1 | -0.001502 | -0.243 |
| char_wb_60k | macro_f1 | -0.001563 | -0.252 |
| char_wb_60k | micro_precision | +0.004968 | +0.625 |
| char_wb_60k | micro_recall | -0.003992 | -0.787 |
| char_wb_60k | fit_wall_seconds | +0.136842 | +1.723 |
| char_wb_60k | fit_peak_process_rss_mib | +3.273438 | +0.558 |
| char_wb_60k | artifact_bytes | +906832.000000 | +31.245 |
| char_wb_60k | fp | -31.000000 | -3.799 |
| char_wb_60k | fn | +25.000000 | +0.810 |
| char_wb_60k | word_dimensions | +6000.000000 | +33.333 |
| char_wb_60k | char_dimensions | +8000.000000 | +28.571 |
| char_wb_60k | total_dimensions | +14000.000000 | +30.435 |
| char_wb_100k | micro_f1 | -0.002761 | -0.446 |
| char_wb_100k | macro_f1 | -0.002830 | -0.457 |
| char_wb_100k | micro_precision | +0.012222 | +1.536 |
| char_wb_100k | micro_recall | -0.008464 | -1.669 |
| char_wb_100k | fit_wall_seconds | +0.115197 | +1.450 |
| char_wb_100k | fit_peak_process_rss_mib | -14.125000 | -2.407 |
| char_wb_100k | artifact_bytes | +3486688.000000 | +120.133 |
| char_wb_100k | fp | -73.000000 | -8.946 |
| char_wb_100k | fn | +53.000000 | +1.717 |
| char_wb_100k | word_dimensions | +22000.000000 | +122.222 |
| char_wb_100k | char_dimensions | +32000.000000 | +114.286 |
| char_wb_100k | total_dimensions | +54000.000000 | +117.391 |

## Paired bootstrap (whole ads, 2000 resamples, seed=42)

- char_wb_60k: delta Micro-F1 95% interval [-0.003206, +0.000181].
- char_wb_100k: delta Micro-F1 95% interval [-0.005214, -0.000386].

Baseline: char_wb. Ranking by held-out Micro-F1: ['char_wb', 'char_wb_60k']. Sublinear check performed: False.

Dataset and supplied splits are unchanged; no resplitting. Exact normalized effective descriptions do not intersect across splits; near-duplicate templates can remain. This is exploratory comparison; the held-out set must not be treated as untouched after model selection. Timing is one run and sensitive to system load. Production artifacts were not replaced.

See summary.json for all exact measurements, configurations, deltas, source hashes and checks.
