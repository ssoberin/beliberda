# Оптимизация CatBoost поверх исходной C_syntax

Проведено 45 обучений CatBoost на исходных public 3398 + V3 clean 2921; новые hard negatives не использовались. Qwen3-Embedding-0.6B, fixed chunker, Top-3 и полные SGD-головы baseline не менялись. Исходный grouped nested OOF восстановлен отдельно; TF-IDF OOF получен с обучением векторизатора только внутри каждого train-фолда. Все вычисления выполнены локально на CPU; Colab не использовался.

## Протокол отбора

В поиске участвуют только train и calibration: 1328 объявлений (public 728 + V3 validation 600). Early stopping — по calibration Logloss с прежним patience=50. Основной score отбора: 0.65 × public calibration F1 + 0.35 × V3 calibration F1. До финального скоринга зафиксированы четыре различных финалиста: сбалансированный, лучший public calibration с ограничением падения V3, вариант с низким FP и ограничением Recall, лучший вариант с фиксированным порогом 0.5. Пороги 0.5, общий и покатегорийные сравнивались только на calibration. Финалисты и SHA-256 весов зафиксированы в коммите fff2d4b до оценки на проверочных наборах. После просмотра public test новые конфигурации не запускались.

## Итоговая таблица

| Модель | pub_validation | public_test_208 | public_test_clean | v3_test |
|---|---:|---:|---:|---:|
| C_syntax_baseline__fixed | 0.956124 | 0.819767 | 0.812680 | 0.915916 |
| adaptive_0_3__per_category | 0.961869 | 0.827786 | 0.824207 | 0.918946 |
| adaptive_0_2__per_category | 0.959893 | 0.830189 | 0.828986 | 0.913767 |
| l2_20__global | 0.952843 | 0.818049 | 0.809249 | 0.917918 |
| adaptive_1_0__fixed | 0.956732 | 0.820961 | 0.815029 | 0.926829 |

Clean — 152 строки: исключены все 56 строк из 13 групп с противоречивым gold при одинаковом нормализованном description. Это **не** прежние 114 уникальных строк без пересечений. Здесь сохраняются непротиворечивые повторы и возможные пересечения с train. Все модели используют один список public_test_exclusions.json; gold не исправлялся.

## FP/FN, Precision/Recall и Exact Match

| Модель | Набор | ΔF1,п.п. | P | R | TP | FP(Δ) | FN(Δ) | Exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| C_syntax_baseline__fixed | pub_validation | +0.000 | 0.9432 | 0.9694 | 1776 | 107(+0) | 56(+0) | 0.8381 |
| C_syntax_baseline__fixed | official_208 | +0.000 | 0.7176 | 0.9559 | 282 | 111(+0) | 13(+0) | 0.6779 |
| C_syntax_baseline__fixed | public_test_clean | +0.000 | 0.7231 | 0.9276 | 141 | 54(+0) | 11(+0) | 0.7500 |
| C_syntax_baseline__fixed | v3_test | +0.000 | 0.9123 | 0.9196 | 915 | 88(+0) | 80(+0) | 0.7567 |
| adaptive_0_3__per_category | pub_validation | +0.575 | 0.9466 | 0.9776 | 1791 | 101(-6) | 41(-15) | 0.8697 |
| adaptive_0_3__per_category | official_208 | +0.802 | 0.7222 | 0.9695 | 286 | 110(-1) | 9(-4) | 0.7019 |
| adaptive_0_3__per_category | public_test_clean | +1.153 | 0.7333 | 0.9408 | 143 | 52(-2) | 9(-2) | 0.7763 |
| adaptive_0_3__per_category | v3_test | +0.303 | 0.9265 | 0.9116 | 907 | 72(-16) | 88(+8) | 0.7600 |
| adaptive_0_2__per_category | pub_validation | +0.377 | 0.9408 | 0.9798 | 1795 | 113(+6) | 37(-19) | 0.8669 |
| adaptive_0_2__per_category | official_208 | +1.042 | 0.7259 | 0.9695 | 286 | 108(-3) | 9(-4) | 0.7019 |
| adaptive_0_2__per_category | public_test_clean | +1.631 | 0.7409 | 0.9408 | 143 | 50(-4) | 9(-2) | 0.7763 |
| adaptive_0_2__per_category | v3_test | -0.215 | 0.9170 | 0.9106 | 906 | 82(-6) | 89(+9) | 0.7483 |
| l2_20__global | pub_validation | -0.328 | 0.9409 | 0.9651 | 1768 | 111(+4) | 64(+8) | 0.8258 |
| l2_20__global | official_208 | -0.172 | 0.7168 | 0.9525 | 281 | 111(+0) | 14(+1) | 0.6779 |
| l2_20__global | public_test_clean | -0.343 | 0.7216 | 0.9211 | 140 | 54(+0) | 12(+1) | 0.7500 |
| l2_20__global | v3_test | +0.200 | 0.9143 | 0.9216 | 917 | 86(-2) | 78(-2) | 0.7600 |
| adaptive_1_0__fixed | pub_validation | +0.061 | 0.9423 | 0.9716 | 1780 | 109(+2) | 52(-4) | 0.8450 |
| adaptive_1_0__fixed | official_208 | +0.119 | 0.7194 | 0.9559 | 282 | 110(-1) | 13(+0) | 0.6779 |
| adaptive_1_0__fixed | public_test_clean | +0.235 | 0.7268 | 0.9276 | 141 | 53(-1) | 11(+0) | 0.7500 |
| adaptive_1_0__fixed | v3_test | +1.091 | 0.9181 | 0.9357 | 931 | 83(-5) | 64(-16) | 0.7800 |

## Параметры финалистов и стоимость

- **C_syntax_baseline__fixed**: {}; пороги=[0.5, 0.5, 0.5, 0.5, 0.5]; trees=959; модель=0.381MiB.
- **adaptive_0_3__per_category**: {"name": "adaptive_0_3", "tfidf": true, "params": {"depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 15}}; пороги=[0.6, 0.525, 0.375, 0.375, 0.25]; trees=1000; модель=1.106MiB.
- **adaptive_0_2__per_category**: {"name": "adaptive_0_2", "tfidf": true, "params": {"depth": 5}, "v3_weight": 0.5}; пороги=[0.575, 0.525, 0.25, 0.275, 0.275]; trees=519; модель=0.336MiB.
- **l2_20__global**: {"name": "l2_20", "params": {"l2_leaf_reg": 20}}; пороги=[0.525, 0.525, 0.525, 0.525, 0.525]; trees=991; модель=0.385MiB.
- **adaptive_1_0__fixed**: {"name": "adaptive_1_0", "params": {"depth": 8, "l2_leaf_reg": 15}}; пороги=[0.5, 0.5, 0.5, 0.5, 0.5]; trees=374; модель=1.499MiB.

Суммарное время 45 fit CatBoost: 488.85 с (без подготовки OOF). Восстановление исходного nested SGD OOF: 457.22 с. Дополнительный сохранённый TF-IDF занимает 2.768 MiB. На 728 calibration-текстах медиана трёх прогретых пакетных запусков TF-IDF + 5 SGD-голов — 0.872 с, или 1.198 мс на объявление в среднем по пакету. Это измерение добавочной CPU-работы, без Qwen и Natasha; оно не является задержкой одиночного запроса.

Варианты с TF-IDF добавляют существующую разреженную векторизацию и пять линейных голов к инференсу; encoder, chunker и retrieval прежние. Использованы словесные триграммы (1,3) и char_wb (3,5), с прежними лимитами 18k/28k. Замеры prediction_seconds относятся только к пакетному CatBoost и не являются полной задержкой инференса. Полная задержка дополненного пайплайна не измерялась. Веса лучших вариантов сохранены отдельно; baseline не заменялся.

## Все эксперименты: отбор по calibration

| № | Конфигурация | Изменения | Cal public F1(0.5) | Cal V3 F1(0.5) | Лучший режим | Лучший score | Fit,с |
|---|---|---|---:|---:|---|---:|---:|
| 1 | baseline_refit | {} | 0.969001 | 0.953751 | per_category | 0.967283 | 9.73 |
| 2 | depth_5 | {"params": {"depth": 5}} | 0.968207 | 0.957458 | per_category | 0.966449 | 9.14 |
| 3 | depth_6 | {"params": {"depth": 6}} | 0.968742 | 0.955897 | per_category | 0.967066 | 11.35 |
| 4 | depth_8 | {"params": {"depth": 8}} | 0.970021 | 0.955362 | per_category | 0.968614 | 12.25 |
| 5 | l2_1 | {"params": {"l2_leaf_reg": 1}} | 0.967914 | 0.954289 | per_category | 0.966819 | 9.97 |
| 6 | l2_3 | {"params": {"l2_leaf_reg": 3}} | 0.970069 | 0.953309 | per_category | 0.967049 | 9.97 |
| 7 | l2_10 | {"params": {"l2_leaf_reg": 10}} | 0.967139 | 0.957414 | per_category | 0.965900 | 13.48 |
| 8 | l2_20 | {"params": {"l2_leaf_reg": 20}} | 0.969519 | 0.955224 | per_category | 0.966915 | 11.87 |
| 9 | l2_50 | {"params": {"l2_leaf_reg": 50}} | 0.968466 | 0.949328 | per_category | 0.966222 | 13.09 |
| 10 | lr_0.025 | {"params": {"learning_rate": 0.025}} | 0.968691 | 0.954779 | per_category | 0.966874 | 14.30 |
| 11 | lr_0.035 | {"params": {"learning_rate": 0.035}} | 0.967932 | 0.957458 | per_category | 0.966214 | 14.07 |
| 12 | lr_0.075 | {"params": {"learning_rate": 0.075}} | 0.968207 | 0.955362 | per_category | 0.967132 | 7.43 |
| 13 | lr_0.1 | {"params": {"learning_rate": 0.1}} | 0.968725 | 0.957588 | per_category | 0.966224 | 4.10 |
| 14 | random_0 | {"params": {"random_strength": 0}} | 0.969017 | 0.955316 | per_category | 0.965589 | 8.84 |
| 15 | random_2 | {"params": {"random_strength": 2}} | 0.969502 | 0.956388 | per_category | 0.967276 | 13.03 |
| 16 | random_5 | {"params": {"random_strength": 5}} | 0.967380 | 0.953165 | per_category | 0.966272 | 13.19 |
| 17 | random_10 | {"params": {"random_strength": 10}} | 0.969292 | 0.956433 | per_category | 0.967274 | 13.83 |
| 18 | bagging_0 | {"params": {"bootstrap_type": "Bayesian", "bagging_temperature": 0}} | 0.968758 | 0.955807 | per_category | 0.966281 | 7.78 |
| 19 | bagging_0.5 | {"params": {"bootstrap_type": "Bayesian", "bagging_temperature": 0.5}} | 0.967139 | 0.955270 | per_category | 0.966333 | 14.66 |
| 20 | bagging_2 | {"params": {"bootstrap_type": "Bayesian", "bagging_temperature": 2}} | 0.965480 | 0.950854 | per_category | 0.964516 | 9.93 |
| 21 | drop_syn_global_turnkey_only | {"drop": ["syn_global_turnkey_only"]} | 0.966863 | 0.952087 | per_category | 0.964702 | 10.90 |
| 22 | drop_syn_global_partial_refusal | {"drop": ["syn_global_partial_refusal"]} | 0.967156 | 0.959058 | per_category | 0.966886 | 10.89 |
| 23 | drop_syn_global_outside_top3 | {"drop": ["syn_global_outside_top3"]} | 0.969826 | 0.955852 | per_category | 0.966731 | 11.18 |
| 24 | drop_all_global | {"drop": ["syn_global_turnkey_only", "syn_global_partial_refusal", "syn_global_outside_top3"]} | 0.965259 | 0.956790 | per_category | 0.964126 | 7.54 |
| 25 | v3_weight_0.25 | {"v3_weight": 0.25} | 0.966124 | 0.948995 | per_category | 0.965955 | 9.98 |
| 26 | v3_weight_0.5 | {"v3_weight": 0.5} | 0.967174 | 0.954779 | per_category | 0.966542 | 13.71 |
| 27 | v3_weight_0.75 | {"v3_weight": 0.75} | 0.967725 | 0.955897 | per_category | 0.967357 | 14.44 |
| 28 | v3_weight_1.5 | {"v3_weight": 1.5} | 0.968190 | 0.956923 | per_category | 0.966770 | 14.49 |
| 29 | v3_weight_2 | {"v3_weight": 2} | 0.967380 | 0.955852 | per_category | 0.965580 | 9.09 |
| 30 | cat_numeric | {"cat_mode": "numeric"} | 0.967139 | 0.956923 | per_category | 0.966021 | 8.59 |
| 31 | cat_onehot | {"cat_mode": "onehot"} | 0.968122 | 0.953799 | per_category | 0.964883 | 10.90 |
| 32 | tfidf | {"tfidf": true} | 0.975753 | 0.935024 | per_category | 0.967523 | 8.70 |
| 33 | tfidf_depth5 | {"tfidf": true, "params": {"depth": 5}} | 0.975193 | 0.938389 | per_category | 0.968705 | 8.27 |
| 34 | tfidf_no_turnkey | {"tfidf": true, "drop": ["syn_global_turnkey_only"]} | 0.974700 | 0.930578 | per_category | 0.967169 | 8.90 |
| 35 | adaptive_0_0 | {"tfidf": true, "params": {"depth": 5, "l2_leaf_reg": 15}} | 0.976521 | 0.934599 | per_category | 0.967800 | 9.96 |
| 36 | adaptive_0_1 | {"tfidf": true, "params": {"depth": 5, "random_strength": 2}} | 0.975740 | 0.937269 | per_category | 0.968318 | 9.31 |
| 37 | adaptive_0_2 | {"tfidf": true, "params": {"depth": 5}, "v3_weight": 0.5} | 0.976533 | 0.928950 | per_category | 0.968369 | 8.10 |
| 38 | adaptive_0_3 | {"tfidf": true, "params": {"depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 15}} | 0.977067 | 0.935924 | per_category | 0.969570 | 17.38 |
| 39 | adaptive_1_0 | {"params": {"depth": 8, "l2_leaf_reg": 15}} | 0.971108 | 0.958483 | per_category | 0.968290 | 10.59 |
| 40 | adaptive_1_1 | {"params": {"depth": 8, "random_strength": 2}} | 0.969810 | 0.957545 | per_category | 0.968656 | 12.86 |
| 41 | adaptive_1_2 | {"params": {"depth": 8}, "v3_weight": 0.5} | 0.969260 | 0.954779 | per_category | 0.967854 | 10.63 |
| 42 | adaptive_1_3 | {"params": {"depth": 6, "learning_rate": 0.035, "l2_leaf_reg": 15}} | 0.969260 | 0.956343 | per_category | 0.968004 | 13.42 |
| 43 | adaptive_2_0 | {"tfidf": true, "params": {"l2_leaf_reg": 15}} | 0.976273 | 0.934955 | per_category | 0.966173 | 10.28 |
| 44 | adaptive_2_1 | {"tfidf": true, "params": {"random_strength": 2}} | 0.974933 | 0.935024 | per_category | 0.967727 | 9.57 |
| 45 | adaptive_2_2 | {"tfidf": true, "params": {}, "v3_weight": 0.5} | 0.976013 | 0.931071 | per_category | 0.966954 | 7.14 |

Полные параметры, включая дефолты и колонки, — experiments.json. Все три режима порогов с FP/FN — calibration_metrics.csv. Категориальный cat_id в baseline уже был корректным; numeric и one-hot — альтернативные представления. Bayesian bagging исследован с заменой исходного MVS. Проверены 34 исходных конфигурации и 12 адаптивных комбинаций; одна комбинация совпала с уже выполненной, поэтому всего 45 обучений, включая воспроизведение baseline.

## Почему publictest хуже validation

- pub_validation: P=0.9432, R=0.9694, FP=107, FN=56; gold=[] в 203 строках, FP на пустом gold=56, FPR=0.0590.
- official_208: P=0.7176, R=0.9559, FP=111, FN=13; gold=[] в 106 строках, FP на пустом gold=82, FPR=0.1490.

В official_208 отсутствует source_mcId; 101 — допущение, а не значение parquet. В 20 строках есть точное совпадение с фактическим public train. Метрики baseline по срезам: {"conflicting_rows": {"rows": 56, "tp": 141, "fp": 57, "fn": 2, "f1": 0.8269794721407625}, "nonconflicting_rows": {"rows": 152, "tp": 141, "fp": 54, "fn": 11, "f1": 0.8126801152737753}, "train_overlap": {"rows": 20, "tp": 27, "fp": 5, "fn": 0, "f1": 0.9152542372881356}, "no_train_overlap": {"rows": 188, "tp": 255, "fp": 106, "fn": 13, "f1": 0.8108108108108109}}.

Precision снижается по двум различимым причинам. Есть реальные ошибки переноса: выравнивание деревянного дома принимается за штукатурку, обсуждение отопления — за сантехнику. Одновременно часть формальных FP — явно предлагаемые услуги, отсутствующие в gold. Пустых gold на public test 51.0%, на validation 27.8%; FPR также выше (14.9% против 5.9%), поэтому одной долей отрицательных строк разрыв не объясняется. Удаление конфликтных групп не исправляет остальные спорные метки. По этим данным нельзя точно разложить весь разрыв на ошибку модели и ошибку разметки.

## Какие признаки повышают ошибочные score

| Признак | Mean SHAP на FP | Mean abs SHAP на FP | Mean SHAP на TP |
|---|---:|---:|---:|
| p_sgd | +1.5830 | 1.9479 | +2.5066 |
| syn_service_mentioned_ad | +1.2487 | 1.2673 | +1.1988 |
| syn_service_mentioned_top3 | +0.5744 | 0.5956 | +0.5602 |
| ad_char_len | -0.2413 | 0.3570 | -0.3176 |
| n_chunks | +0.1202 | 0.2422 | +0.0955 |
| cat_id | +0.1206 | 0.1567 | +0.1133 |
| sim_margin_01 | -0.0627 | 0.1526 | -0.0540 |
| service_link_status | +0.1078 | 0.1295 | +0.1228 |
| syn_net_polarity_score | +0.1139 | 0.1289 | +0.1203 |
| syn_min_dist_service_neg | +0.0623 | 0.1283 | +0.0884 |
| syn_top_neg_sim | +0.1085 | 0.1258 | +0.1025 |
| sim_mean | +0.0735 | 0.0987 | +0.0724 |

SHAP выражен в logit и описывает атрибуцию baseline, без доказательства причинности или валидности gold. На FP больше всего положительно влияют p_sgd и флаги упоминания услуги во всём объявлении и Top-3. В 81 из 111 FP нет признака явного положительного предложения; у 74 связь с услугой orphan. Это согласуется с зависимостью от лексического упоминания, но не доказывает причину каждой ошибки. syn_global_turnkey_only равен 0 во всех 111 FP: нельзя объяснить эти ошибки только этим флагом. Подробные распределения: public_test_analysis.json.

## Изменения по категориям

| Модель | Набор | Категория | F1 | FP(Δ) | FN(Δ) |
|---|---|---|---:|---:|---:|
| adaptive_0_3__per_category | pub_validation | 102 | 0.9543 | 18(-8) | 16(-1) |
| adaptive_0_3__per_category | pub_validation | 103 | 0.9642 | 22(-3) | 4(-1) |
| adaptive_0_3__per_category | pub_validation | 104 | 0.9556 | 23(+2) | 9(-5) |
| adaptive_0_3__per_category | pub_validation | 105 | 0.9678 | 20(+1) | 3(-4) |
| adaptive_0_3__per_category | pub_validation | 106 | 0.9670 | 18(+2) | 9(-4) |
| adaptive_0_3__per_category | official_208 | 102 | 0.7436 | 36(-5) | 4(-2) |
| adaptive_0_3__per_category | official_208 | 103 | 0.7907 | 16(+0) | 2(+1) |
| adaptive_0_3__per_category | official_208 | 104 | 0.8649 | 19(+1) | 1(-1) |
| adaptive_0_3__per_category | official_208 | 105 | 0.8521 | 23(+1) | 2(+0) |
| adaptive_0_3__per_category | official_208 | 106 | 0.8788 | 16(+2) | 0(-2) |
| adaptive_0_3__per_category | v3_test | 102 | 0.9282 | 11(-3) | 17(+7) |
| adaptive_0_3__per_category | v3_test | 103 | 0.8936 | 9(-4) | 31(+7) |
| adaptive_0_3__per_category | v3_test | 104 | 0.9046 | 25(-8) | 14(+0) |
| adaptive_0_3__per_category | v3_test | 105 | 0.9545 | 8(-1) | 10(+0) |
| adaptive_0_3__per_category | v3_test | 106 | 0.9132 | 19(+0) | 16(-6) |
| adaptive_0_2__per_category | pub_validation | 102 | 0.9531 | 20(-6) | 15(-2) |
| adaptive_0_2__per_category | pub_validation | 103 | 0.9627 | 21(-4) | 6(+1) |
| adaptive_0_2__per_category | pub_validation | 104 | 0.9497 | 33(+12) | 4(-10) |
| adaptive_0_2__per_category | pub_validation | 105 | 0.9666 | 22(+3) | 2(-5) |
| adaptive_0_2__per_category | pub_validation | 106 | 0.9670 | 17(+1) | 10(-3) |
| adaptive_0_2__per_category | official_208 | 102 | 0.7403 | 35(-6) | 5(-1) |
| adaptive_0_2__per_category | official_208 | 103 | 0.8000 | 15(-1) | 2(+1) |
| adaptive_0_2__per_category | official_208 | 104 | 0.8725 | 19(+1) | 0(-2) |
| adaptive_0_2__per_category | official_208 | 105 | 0.8521 | 23(+1) | 2(+0) |
| adaptive_0_2__per_category | official_208 | 106 | 0.8788 | 16(+2) | 0(-2) |
| adaptive_0_2__per_category | v3_test | 102 | 0.9254 | 11(-3) | 18(+8) |
| adaptive_0_2__per_category | v3_test | 103 | 0.8787 | 9(-4) | 36(+12) |
| adaptive_0_2__per_category | v3_test | 104 | 0.9021 | 31(-2) | 10(-4) |
| adaptive_0_2__per_category | v3_test | 105 | 0.9604 | 11(+2) | 5(-5) |
| adaptive_0_2__per_category | v3_test | 106 | 0.9000 | 20(+1) | 20(-2) |
| l2_20__global | pub_validation | 102 | 0.9358 | 27(+1) | 21(+4) |
| l2_20__global | pub_validation | 103 | 0.9574 | 25(+0) | 6(+1) |
| l2_20__global | pub_validation | 104 | 0.9465 | 21(+0) | 17(+3) |
| l2_20__global | pub_validation | 105 | 0.9607 | 21(+2) | 7(+0) |
| l2_20__global | pub_validation | 106 | 0.9631 | 17(+1) | 13(+0) |
| l2_20__global | official_208 | 102 | 0.6962 | 41(+0) | 7(+1) |
| l2_20__global | official_208 | 103 | 0.8046 | 16(+0) | 1(+0) |
| l2_20__global | official_208 | 104 | 0.8630 | 18(+0) | 2(+0) |
| l2_20__global | official_208 | 105 | 0.8571 | 22(+0) | 2(+0) |
| l2_20__global | official_208 | 106 | 0.8750 | 14(+0) | 2(+0) |
| l2_20__global | v3_test | 102 | 0.9293 | 14(+0) | 14(+4) |
| l2_20__global | v3_test | 103 | 0.9010 | 12(-1) | 26(+2) |
| l2_20__global | v3_test | 104 | 0.9048 | 31(-2) | 9(-5) |
| l2_20__global | v3_test | 105 | 0.9576 | 10(+1) | 7(-3) |
| l2_20__global | v3_test | 106 | 0.8967 | 19(+0) | 22(+0) |
| adaptive_1_0__fixed | pub_validation | 102 | 0.9456 | 26(+0) | 15(-2) |
| adaptive_1_0__fixed | pub_validation | 103 | 0.9603 | 26(+1) | 3(-2) |
| adaptive_1_0__fixed | pub_validation | 104 | 0.9480 | 21(+0) | 16(+2) |
| adaptive_1_0__fixed | pub_validation | 105 | 0.9634 | 19(+0) | 7(+0) |
| adaptive_1_0__fixed | pub_validation | 106 | 0.9657 | 17(+1) | 11(-2) |
| adaptive_1_0__fixed | official_208 | 102 | 0.7089 | 40(-1) | 6(+0) |
| adaptive_1_0__fixed | official_208 | 103 | 0.8046 | 16(+0) | 1(+0) |
| adaptive_1_0__fixed | official_208 | 104 | 0.8630 | 18(+0) | 2(+0) |
| adaptive_1_0__fixed | official_208 | 105 | 0.8571 | 22(+0) | 2(+0) |
| adaptive_1_0__fixed | official_208 | 106 | 0.8750 | 14(+0) | 2(+0) |
| adaptive_1_0__fixed | v3_test | 102 | 0.9447 | 12(-2) | 10(+0) |
| adaptive_1_0__fixed | v3_test | 103 | 0.9096 | 12(-1) | 23(-1) |
| adaptive_1_0__fixed | v3_test | 104 | 0.9091 | 29(-4) | 9(-5) |
| adaptive_1_0__fixed | v3_test | 105 | 0.9602 | 10(+1) | 6(-4) |
| adaptive_1_0__fixed | v3_test | 106 | 0.9109 | 20(+1) | 16(-6) |

## Интервалы и регрессии

- adaptive_0_3__per_category/pub_validation: {"both_correct": 3445, "base_wrong_cand_right_fixed": 58, "base_right_cand_wrong_broken": 37, "both_wrong": 105, "net_gain": 21, "fixed_breakdown": {"fn_to_tp": 28, "fp_to_tn": 30}, "broken_breakdown": {"tp_to_fn": 13, "tn_to_fp": 24}, "new_erroneous_empty_ads": 2}; pairedCI={"delta_mean": 0.005769, "ci95_low": 4.9e-05, "ci95_high": 0.011289, "p_positive": 0.9755}; groupedCI=null.
- adaptive_0_3__per_category/official_208: {"both_correct": 907, "base_wrong_cand_right_fixed": 14, "base_right_cand_wrong_broken": 9, "both_wrong": 110, "net_gain": 5, "fixed_breakdown": {"fn_to_tp": 6, "fp_to_tn": 8}, "broken_breakdown": {"tp_to_fn": 2, "tn_to_fp": 7}, "new_erroneous_empty_ads": 1}; pairedCI={"delta_mean": 0.007817, "ci95_low": -0.005524, "ci95_high": 0.023073, "p_positive": 0.856}; groupedCI={"groups": 139, "ci95_low": -0.005761321099281216, "ci95_high": 0.026377648959708785, "delta_mean": 0.008565934069404714}.
- adaptive_0_3__per_category/v3_test: {"both_correct": 2296, "base_wrong_cand_right_fixed": 44, "base_right_cand_wrong_broken": 36, "both_wrong": 124, "net_gain": 8, "fixed_breakdown": {"fn_to_tp": 23, "fp_to_tn": 21}, "broken_breakdown": {"tp_to_fn": 31, "tn_to_fp": 5}, "new_erroneous_empty_ads": 10}; pairedCI={"delta_mean": 0.003092, "ci95_low": -0.005542, "ci95_high": 0.011793, "p_positive": 0.75}; groupedCI=null.
- adaptive_0_2__per_category/pub_validation: {"both_correct": 3436, "base_wrong_cand_right_fixed": 59, "base_right_cand_wrong_broken": 46, "both_wrong": 104, "net_gain": 13, "fixed_breakdown": {"fn_to_tp": 31, "fp_to_tn": 28}, "broken_breakdown": {"tp_to_fn": 12, "tn_to_fp": 34}, "new_erroneous_empty_ads": 1}; pairedCI={"delta_mean": 0.003746, "ci95_low": -0.002063, "ci95_high": 0.009676, "p_positive": 0.8965}; groupedCI=null.
- adaptive_0_2__per_category/official_208: {"both_correct": 906, "base_wrong_cand_right_fixed": 17, "base_right_cand_wrong_broken": 10, "both_wrong": 107, "net_gain": 7, "fixed_breakdown": {"fn_to_tp": 6, "fp_to_tn": 11}, "broken_breakdown": {"tp_to_fn": 2, "tn_to_fp": 8}, "new_erroneous_empty_ads": 1}; pairedCI={"delta_mean": 0.010304, "ci95_low": -0.003679, "ci95_high": 0.025666, "p_positive": 0.9145}; groupedCI={"groups": 139, "ci95_low": -0.004611009858858484, "ci95_high": 0.029850357581303232, "delta_mean": 0.010983982627686031}.
- adaptive_0_2__per_category/v3_test: {"both_correct": 2287, "base_wrong_cand_right_fixed": 42, "base_right_cand_wrong_broken": 45, "both_wrong": 126, "net_gain": -3, "fixed_breakdown": {"fn_to_tp": 24, "fp_to_tn": 18}, "broken_breakdown": {"tp_to_fn": 33, "tn_to_fp": 12}, "new_erroneous_empty_ads": 10}; pairedCI={"delta_mean": -0.002039, "ci95_low": -0.011808, "ci95_high": 0.007449, "p_positive": 0.347}; groupedCI=null.
- l2_20__global/pub_validation: {"both_correct": 3466, "base_wrong_cand_right_fixed": 4, "base_right_cand_wrong_broken": 16, "both_wrong": 159, "net_gain": -12, "fixed_breakdown": {"fn_to_tp": 1, "fp_to_tn": 3}, "broken_breakdown": {"tp_to_fn": 9, "tn_to_fp": 7}, "new_erroneous_empty_ads": 0}; pairedCI={"delta_mean": -0.003347, "ci95_low": -0.005969, "ci95_high": -0.00099, "p_positive": 0.0025}; groupedCI=null.
- l2_20__global/official_208: {"both_correct": 915, "base_wrong_cand_right_fixed": 0, "base_right_cand_wrong_broken": 1, "both_wrong": 124, "net_gain": -1, "fixed_breakdown": {"fn_to_tp": 0, "fp_to_tn": 0}, "broken_breakdown": {"tp_to_fn": 1, "tn_to_fp": 0}, "new_erroneous_empty_ads": 0}; pairedCI={"delta_mean": -0.001652, "ci95_low": -0.005719, "ci95_high": 0.0, "p_positive": 0.0}; groupedCI={"groups": 139, "ci95_low": -0.006524990651875839, "ci95_high": 0.0, "delta_mean": -0.0017481605810721556}.
- l2_20__global/v3_test: {"both_correct": 2320, "base_wrong_cand_right_fixed": 16, "base_right_cand_wrong_broken": 12, "both_wrong": 152, "net_gain": 4, "fixed_breakdown": {"fn_to_tp": 12, "fp_to_tn": 4}, "broken_breakdown": {"tp_to_fn": 10, "tn_to_fp": 2}, "new_erroneous_empty_ads": 4}; pairedCI={"delta_mean": 0.002005, "ci95_low": -0.00335, "ci95_high": 0.007509, "p_positive": 0.756}; groupedCI=null.
- adaptive_1_0__fixed/pub_validation: {"both_correct": 3471, "base_wrong_cand_right_fixed": 13, "base_right_cand_wrong_broken": 11, "both_wrong": 150, "net_gain": 2, "fixed_breakdown": {"fn_to_tp": 7, "fp_to_tn": 6}, "broken_breakdown": {"tp_to_fn": 3, "tn_to_fp": 8}, "new_erroneous_empty_ads": 1}; pairedCI={"delta_mean": 0.000566, "ci95_low": -0.002045, "ci95_high": 0.003288, "p_positive": 0.6645}; groupedCI=null.
- adaptive_1_0__fixed/official_208: {"both_correct": 915, "base_wrong_cand_right_fixed": 2, "base_right_cand_wrong_broken": 1, "both_wrong": 122, "net_gain": 1, "fixed_breakdown": {"fn_to_tp": 0, "fp_to_tn": 2}, "broken_breakdown": {"tp_to_fn": 0, "tn_to_fp": 1}, "new_erroneous_empty_ads": 0}; pairedCI={"delta_mean": 0.001228, "ci95_low": -0.002522, "ci95_high": 0.005534, "p_positive": 0.6075}; groupedCI={"groups": 139, "ci95_low": -0.0026975395085119536, "ci95_high": 0.005761833245915337, "delta_mean": 0.0012155603599418616}.
- adaptive_1_0__fixed/v3_test: {"both_correct": 2324, "base_wrong_cand_right_fixed": 29, "base_right_cand_wrong_broken": 8, "both_wrong": 139, "net_gain": 21, "fixed_breakdown": {"fn_to_tp": 22, "fp_to_tn": 7}, "broken_breakdown": {"tp_to_fn": 6, "tn_to_fp": 2}, "new_erroneous_empty_ads": 4}; pairedCI={"delta_mean": 0.010923, "ci95_low": 0.005136, "ci95_high": 0.016965, "p_positive": 1.0}; groupedCI=null.

## Ответ на критерий трёх наборов

По измеренному F1 улучшили public test без снижения public validation и V3: adaptive_0_3__per_category, adaptive_1_0__fixed.

**Основной кандидат по балансу calibration — adaptive_0_3__per_category:** глубина 6, learning_rate=0.035, l2_leaf_reg=15, дополнительная OOF-вероятность TF-IDF. Пороги категорий 102–106: [0.600, 0.525, 0.375, 0.375, 0.250]. На public test выигрыш +0.802 п.п.; на clean +1.153 п.п. Но на V3 FN выросли с 80 до 88, а 10 объявлений с непустым gold впервые получили пустой ответ. Поэтому рост F1 не означает улучшение всех типов ошибок.

**Более простой кандидат — adaptive_1_0__fixed:** глубина 8, l2_leaf_reg=15, остальные параметры baseline, прежние признаки и порог 0.5. V3 F1 вырос с 0.915916 до 0.926829; FN сократились с 80 до 64. Выигрыш на public test всего +0.119 п.п. и семантически смешанный: одно формальное исправление подавляет явно предложенные малярные работы (строка 30). Это нельзя считать доказанным улучшением public test.

**Статистическая убедительность:** у основного кандидата групповой парный bootstrap CI изменения public test F1 составляет [-0.576; +2.638] п.п. и включает ноль. Шесть исправленных строк о деревянных домах — копии одного текста; их нельзя трактовать как шесть независимых подтверждений. На V3 CI основного кандидата также включает ноль. У глубины 8 прирост V3 устойчивее: CI [+0.514; +1.696] п.п., но на обоих публичных наборах интервал включает ноль. Это интервалы по данным наборам, без поправки на историческое многократное использование проверок.

Максимальный сырой public test F1=0.830189 у adaptive_0_2__per_category, но V3 снизился до 0.913767 и FN выросли до 89. Выбирать его только ради public test не рекомендуется. Вариант l2_20__global не прошёл итоговую проверку на публичных наборах.

**Рекомендация:** сохранить baseline основной моделью, а основной TF-IDF-финалист и глубину 8 — отдельными кандидатами. Измеренный совместный рост F1 получен, но статистически и семантически убедительное улучшение public test без компромиссов не доказано. Для дальнейшего выбора нужен независимый контроль; повторная настройка по этим тестам не проводилась.

Смысловая проверка конкретных исправлений и регрессий: [public_test_semantic_review.md](public_test_semantic_review.md). На срезе 377 публичных объявлений о комплексном ремонте с несколькими gold-услугами F1 baseline=0.977054, основного кандидата=0.981145, глубины 8=0.978398. Этот срез задан текстовым фильтром и непустым gold; он не заменяет ручной проверки всех перечислений услуг.

Все рекомендации ограничены этой серией; независимого финального теста нет. Отбор 45 моделей и порогов на одной calibration также создаёт риск переоптимизации. Baseline не заменялся; датасеты и тестовые gold не изменялись.
