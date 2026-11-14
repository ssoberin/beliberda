# Public train +0/20/60/100% synthetic train

Public train3398 unchanged in all runs. Synthetic pool=v3train3000+new authored100. Samples nested, seed42, authored pairs kept together and proportionally sampled by topic. Fixed model word(1,3)+char_wb(3,5),18k+28k,SGD unchanged,thresholds0.5. No test data in fit.

| Synthetic fraction | Added rows | Public208 F1 | Public validation729 F1 | V3test600 F1 | New20 F1 | New20 exact pairs | Fit s | Model MiB |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 0% | 0 | 0.814815 | 0.937124 | 0.183099 | 0.190476 | 0/10 | 6.16 | 2.75 |
| 20% | 620 | 0.820588 | 0.935086 | 0.605280 | 0.529412 | 1/10 | 7.65 | 2.76 |
| 60% | 1860 | 0.819383 | 0.935150 | 0.662569 | 0.484848 | 1/10 | 8.99 | 2.76 |
| 100% | 3100 | 0.822320 | 0.934426 | 0.681818 | 0.457143 | 1/10 | 10.40 | 2.77 |

## All measured metrics

| Fraction | Set | Micro-F1 | Macro-F1 | P | R | FP | FN | Exact-set |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0% | official_public_test_208 | 0.814815 | 0.802431 | 0.723684 | 0.932203 | 105 | 20 | 0.673077 |
| 0% | official_clean_unique | 0.742358 | 0.740202 | 0.664062 | 0.841584 | 43 | 16 | 0.728070 |
| 0% | public_validation_729 | 0.937124 | 0.937196 | 0.938664 | 0.935590 | 112 | 118 | 0.817558 |
| 0% | v3_test_600 | 0.183099 | 0.158663 | 0.737589 | 0.104523 | 37 | 891 | 0.200000 |
| 0% | new_control_20 | 0.190476 | 0.146667 | 0.400000 | 0.125000 | 3 | 14 | 0.250000 |
| 0% | v3_challenge_144 | 0.517241 | 0.530101 | 0.535714 | 0.500000 | 65 | 75 | 0.291667 |
| 20% | official_public_test_208 | 0.820588 | 0.809008 | 0.724675 | 0.945763 | 106 | 16 | 0.682692 |
| 20% | official_clean_unique | 0.758621 | 0.757606 | 0.671756 | 0.871287 | 43 | 13 | 0.736842 |
| 20% | public_validation_729 | 0.935086 | 0.935070 | 0.938428 | 0.931769 | 112 | 125 | 0.809328 |
| 20% | v3_test_600 | 0.605280 | 0.602243 | 0.842294 | 0.472362 | 88 | 525 | 0.383333 |
| 20% | new_control_20 | 0.529412 | 0.467619 | 0.500000 | 0.562500 | 9 | 7 | 0.300000 |
| 20% | v3_challenge_144 | 0.676923 | 0.685756 | 0.628571 | 0.733333 | 65 | 40 | 0.444444 |
| 60% | official_public_test_208 | 0.819383 | 0.808272 | 0.722798 | 0.945763 | 107 | 16 | 0.673077 |
| 60% | official_clean_unique | 0.760684 | 0.760185 | 0.669173 | 0.881188 | 44 | 12 | 0.728070 |
| 60% | public_validation_729 | 0.935150 | 0.935197 | 0.933624 | 0.936681 | 122 | 116 | 0.805213 |
| 60% | v3_test_600 | 0.662569 | 0.659440 | 0.852848 | 0.541709 | 93 | 456 | 0.420000 |
| 60% | new_control_20 | 0.484848 | 0.436508 | 0.470588 | 0.500000 | 9 | 8 | 0.350000 |
| 60% | v3_challenge_144 | 0.757576 | 0.765967 | 0.694444 | 0.833333 | 55 | 25 | 0.590278 |
| 100% | official_public_test_208 | 0.822320 | 0.810912 | 0.725389 | 0.949153 | 106 | 15 | 0.682692 |
| 100% | official_clean_unique | 0.763948 | 0.763415 | 0.674242 | 0.881188 | 43 | 12 | 0.736842 |
| 100% | public_validation_729 | 0.934426 | 0.934491 | 0.935449 | 0.933406 | 118 | 122 | 0.807956 |
| 100% | v3_test_600 | 0.681818 | 0.678704 | 0.841950 | 0.572864 | 107 | 425 | 0.418333 |
| 100% | new_control_20 | 0.457143 | 0.411111 | 0.421053 | 0.500000 | 11 | 8 | 0.300000 |
| 100% | v3_challenge_144 | 0.738462 | 0.745513 | 0.685714 | 0.800000 | 55 | 30 | 0.555556 |

Official208 includes conflicts and training overlap; clean_unique is a smaller diagnostic subset. Public729 is validation, not final test. Synthetic and new20 are authored, not independent human gold. New20 contains10scenarios, not20independent stories. Ratios were predeclared; no threshold or hyperparameter selection. Timings are single fits. Production model unchanged.

## Train versus held-out diagnostic

Inference only on the four existing saved models. No new fitting, threshold changes or selection.

| Synthetic fraction | Public train F1 | Public validation F1 | Synthetic train F1 | V3 held-out test F1 | New20 control F1 |
|---|---:|---:|---:|---:|---:|
| 0% | 0.994505 | 0.937124 | — | 0.183099 | 0.190476 |
| 20% | 0.992266 | 0.935086 | 0.914046 | 0.605280 | 0.529412 |
| 60% | 0.988679 | 0.935150 | 0.926517 | 0.662569 | 0.484848 |
| 100% | 0.986198 | 0.934426 | 0.927934 | 0.681818 | 0.457143 |

A material train/held-out gap exists. For the 100% mixture synthetic training F1 is 0.927934 versus 0.681818 on v3 test (24.61 percentage points). This is consistent with overfitting or distribution/template differences; the gap alone does not establish the cause. V3 test improves monotonically as training additions increase, so there is no observed collapse on that test. The separate small control peaks at 20% and then deteriorates; it contains only 10 paired scenarios and cannot reliably select a mixture. Public validation drops 0.27 percentage points relative to public-only baseline. These results do not justify claiming absence of overfitting or promoting a final model. See overfitting_diagnostic.json for full training/held-out scores.
