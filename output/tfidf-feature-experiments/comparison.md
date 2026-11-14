# TF-IDF feature experiments

Fixed word ngrams (1,3); original sublinear_tf=True; existing immutable train/calibration/validation split; seed=42; thresholds=0.5; one fit per variant; single-thread BLAS/OpenMP.
Precision and recall are micro averages. Time includes vectorization and classifier fit. RSS is the absolute process high-water mark at fit completion, not an incremental allocation.

| Variant | Micro-F1 | Macro-F1 | Precision | Recall | FP | FN | Fit s | Word dims | Char dims | Model MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.936217 | 0.936225 | 0.939044 | 0.933406 | 111 | 122 | 8.454 | 18000 | 28000 | 2.752 | 538.9 |
| char_wb | 0.937124 | 0.937196 | 0.938664 | 0.935590 | 112 | 118 | 6.095 | 18000 | 28000 | 2.754 | 473.6 |
| uncapped_char | 0.928610 | 0.928400 | 0.927095 | 0.930131 | 134 | 128 | 7.700 | 111157 | 153393 | 12.624 | 617.6 |
| uncapped_char_wb | 0.934789 | 0.934799 | 0.934534 | 0.935044 | 120 | 119 | 5.488 | 111157 | 68433 | 9.389 | 503.3 |
| char_100k | 0.930997 | 0.930906 | 0.934066 | 0.927948 | 120 | 132 | 8.448 | 40000 | 60000 | 6.080 | 555.8 |
| char_wb_100k | 0.934354 | 0.934322 | 0.936404 | 0.932314 | 116 | 124 | 5.984 | 40000 | 60000 | 6.073 | 487.3 |
| char_60k | 0.933260 | 0.933222 | 0.939193 | 0.927402 | 110 | 133 | 8.501 | 24000 | 36000 | 3.614 | 540.0 |
| char_wb_60k | 0.935342 | 0.935300 | 0.938944 | 0.931769 | 111 | 125 | 6.072 | 24000 | 36000 | 3.614 | 480.2 |
| char_wb_linear_tf | 0.930921 | 0.930901 | 0.935022 | 0.926856 | 118 | 134 | 5.939 | 18000 | 28000 | 2.754 | 476.5 |
| baseline_linear_tf | 0.929685 | 0.929613 | 0.931980 | 0.927402 | 124 | 133 | 8.248 | 18000 | 28000 | 2.752 | 539.0 |

## Differences from baseline

| Variant | Metric | Absolute difference | Relative difference % |
|---|---|---:|---:|
| char_wb | micro_f1 | +0.000907 | +0.097 |
| char_wb | macro_f1 | +0.000971 | +0.104 |
| char_wb | micro_precision | -0.000381 | -0.041 |
| char_wb | micro_recall | +0.002183 | +0.234 |
| char_wb | fit_wall_seconds | -2.359439 | -27.909 |
| char_wb | fit_peak_process_rss_mib | -65.281250 | -12.114 |
| char_wb | artifact_bytes | +1312.000000 | +0.045 |
| char_wb | fp | +1.000000 | +0.901 |
| char_wb | fn | -4.000000 | -3.279 |
| char_wb | word_dimensions | +0.000000 | +0.000 |
| char_wb | char_dimensions | +0.000000 | +0.000 |
| char_wb | total_dimensions | +0.000000 | +0.000 |
| uncapped_char | micro_f1 | -0.007606 | -0.812 |
| uncapped_char | macro_f1 | -0.007825 | -0.836 |
| uncapped_char | micro_precision | -0.011950 | -1.273 |
| uncapped_char | micro_recall | -0.003275 | -0.351 |
| uncapped_char | fit_wall_seconds | -0.753756 | -8.916 |
| uncapped_char | fit_peak_process_rss_mib | +78.667969 | +14.598 |
| uncapped_char | artifact_bytes | +10350976.000000 | +358.637 |
| uncapped_char | fp | +23.000000 | +20.721 |
| uncapped_char | fn | +6.000000 | +4.918 |
| uncapped_char | word_dimensions | +93157.000000 | +517.539 |
| uncapped_char | char_dimensions | +125393.000000 | +447.832 |
| uncapped_char | total_dimensions | +218550.000000 | +475.109 |
| uncapped_char_wb | micro_f1 | -0.001428 | -0.153 |
| uncapped_char_wb | macro_f1 | -0.001426 | -0.152 |
| uncapped_char_wb | micro_precision | -0.004511 | -0.480 |
| uncapped_char_wb | micro_recall | +0.001638 | +0.175 |
| uncapped_char_wb | fit_wall_seconds | -2.966068 | -35.084 |
| uncapped_char_wb | fit_peak_process_rss_mib | -35.613281 | -6.609 |
| uncapped_char_wb | artifact_bytes | +6958400.000000 | +241.092 |
| uncapped_char_wb | fp | +9.000000 | +8.108 |
| uncapped_char_wb | fn | -3.000000 | -2.459 |
| uncapped_char_wb | word_dimensions | +93157.000000 | +517.539 |
| uncapped_char_wb | char_dimensions | +40433.000000 | +144.404 |
| uncapped_char_wb | total_dimensions | +133590.000000 | +290.413 |
| char_100k | micro_f1 | -0.005220 | -0.558 |
| char_100k | macro_f1 | -0.005319 | -0.568 |
| char_100k | micro_precision | -0.004979 | -0.530 |
| char_100k | micro_recall | -0.005459 | -0.585 |
| char_100k | fit_wall_seconds | -0.005974 | -0.071 |
| char_100k | fit_peak_process_rss_mib | +16.855469 | +3.128 |
| char_100k | artifact_bytes | +3489296.000000 | +120.896 |
| char_100k | fp | +9.000000 | +8.108 |
| char_100k | fn | +10.000000 | +8.197 |
| char_100k | word_dimensions | +22000.000000 | +122.222 |
| char_100k | char_dimensions | +32000.000000 | +114.286 |
| char_100k | total_dimensions | +54000.000000 | +117.391 |
| char_wb_100k | micro_f1 | -0.001862 | -0.199 |
| char_wb_100k | macro_f1 | -0.001903 | -0.203 |
| char_wb_100k | micro_precision | -0.002641 | -0.281 |
| char_wb_100k | micro_recall | -0.001092 | -0.117 |
| char_wb_100k | fit_wall_seconds | -2.469599 | -29.212 |
| char_wb_100k | fit_peak_process_rss_mib | -51.597656 | -9.575 |
| char_wb_100k | artifact_bytes | +3481552.000000 | +120.627 |
| char_wb_100k | fp | +5.000000 | +4.505 |
| char_wb_100k | fn | +2.000000 | +1.639 |
| char_wb_100k | word_dimensions | +22000.000000 | +122.222 |
| char_wb_100k | char_dimensions | +32000.000000 | +114.286 |
| char_wb_100k | total_dimensions | +54000.000000 | +117.391 |
| char_60k | micro_f1 | -0.002957 | -0.316 |
| char_60k | macro_f1 | -0.003003 | -0.321 |
| char_60k | micro_precision | +0.000148 | +0.016 |
| char_60k | micro_recall | -0.006004 | -0.643 |
| char_60k | fit_wall_seconds | +0.046551 | +0.551 |
| char_60k | fit_peak_process_rss_mib | +1.125000 | +0.209 |
| char_60k | artifact_bytes | +903408.000000 | +31.301 |
| char_60k | fp | -1.000000 | -0.901 |
| char_60k | fn | +11.000000 | +9.016 |
| char_60k | word_dimensions | +6000.000000 | +33.333 |
| char_60k | char_dimensions | +8000.000000 | +28.571 |
| char_60k | total_dimensions | +14000.000000 | +30.435 |
| char_wb_60k | micro_f1 | -0.000874 | -0.093 |
| char_wb_60k | macro_f1 | -0.000925 | -0.099 |
| char_wb_60k | micro_precision | -0.000101 | -0.011 |
| char_wb_60k | micro_recall | -0.001638 | -0.175 |
| char_wb_60k | fit_wall_seconds | -2.381841 | -28.174 |
| char_wb_60k | fit_peak_process_rss_mib | -58.695312 | -10.892 |
| char_wb_60k | artifact_bytes | +903104.000000 | +31.290 |
| char_wb_60k | fp | +0.000000 | +0.000 |
| char_wb_60k | fn | +3.000000 | +2.459 |
| char_wb_60k | word_dimensions | +6000.000000 | +33.333 |
| char_wb_60k | char_dimensions | +8000.000000 | +28.571 |
| char_wb_60k | total_dimensions | +14000.000000 | +30.435 |
| char_wb_linear_tf | micro_f1 | -0.005296 | -0.566 |
| char_wb_linear_tf | macro_f1 | -0.005324 | -0.569 |
| char_wb_linear_tf | micro_precision | -0.004022 | -0.428 |
| char_wb_linear_tf | micro_recall | -0.006550 | -0.702 |
| char_wb_linear_tf | fit_wall_seconds | -2.514921 | -29.748 |
| char_wb_linear_tf | fit_peak_process_rss_mib | -62.429688 | -11.585 |
| char_wb_linear_tf | artifact_bytes | +1312.000000 | +0.045 |
| char_wb_linear_tf | fp | +7.000000 | +6.306 |
| char_wb_linear_tf | fn | +12.000000 | +9.836 |
| char_wb_linear_tf | word_dimensions | +0.000000 | +0.000 |
| char_wb_linear_tf | char_dimensions | +0.000000 | +0.000 |
| char_wb_linear_tf | total_dimensions | +0.000000 | +0.000 |
| baseline_linear_tf | micro_f1 | -0.006531 | -0.698 |
| baseline_linear_tf | macro_f1 | -0.006612 | -0.706 |
| baseline_linear_tf | micro_precision | -0.007064 | -0.752 |
| baseline_linear_tf | micro_recall | -0.006004 | -0.643 |
| baseline_linear_tf | fit_wall_seconds | -0.206288 | -2.440 |
| baseline_linear_tf | fit_peak_process_rss_mib | +0.132812 | +0.025 |
| baseline_linear_tf | artifact_bytes | +0.000000 | +0.000 |
| baseline_linear_tf | fp | +13.000000 | +11.712 |
| baseline_linear_tf | fn | +11.000000 | +9.016 |
| baseline_linear_tf | word_dimensions | +0.000000 | +0.000 |
| baseline_linear_tf | char_dimensions | +0.000000 | +0.000 |
| baseline_linear_tf | total_dimensions | +0.000000 | +0.000 |

## Paired bootstrap (whole ads, 2000 resamples, seed=42)

- char_wb: delta Micro-F1 95% interval [-0.001248, +0.003243].
- uncapped_char: delta Micro-F1 95% interval [-0.012681, -0.002828].
- uncapped_char_wb: delta Micro-F1 95% interval [-0.005579, +0.002799].
- char_100k: delta Micro-F1 95% interval [-0.008576, -0.002058].
- char_wb_100k: delta Micro-F1 95% interval [-0.005541, +0.001795].
- char_60k: delta Micro-F1 95% interval [-0.006035, +0.000011].
- char_wb_60k: delta Micro-F1 95% interval [-0.004071, +0.002160].
- char_wb_linear_tf: delta Micro-F1 95% interval [-0.010509, -0.000037].
- baseline_linear_tf: delta Micro-F1 95% interval [-0.011935, -0.001320].

Top two selected by validation Micro-F1: ['char_wb', 'baseline']. Original sublinear_tf already True; *_linear_tf disables it in both branches for the additional comparison.

The original weakly labelled dataset and split algorithm are unchanged. Exact normalized effective descriptions do not intersect across splits; near-duplicate templates can remain. This is exploratory model selection on one validation split, not an independent final test. Timing is one run and sensitive to system load. Production artifacts were not replaced.

See summary.json for all exact measurements, configurations, deltas, source hashes and checks.
