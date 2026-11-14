# Парное сравнение: public против public + v2

Одинаковая архитектура: word(1,3), char_wb(3,5), 18000+28000 признаков, sublinear_tf=True, пять SGD с random_state42, пороги0.5. Различаются только обучающие данные. Использованы сохранённые экспериментальные артефакты, не проверенные серверные production-веса; повторное обучение не выполнялось.
public_only обучена на3398 уникальных public train; public_plus_v2 — на тех же3398 и8464 v2 train. Для оценки источник каждого объявления исключается, scope102–106. Для Parquet принято mcId101.

| Проверка | Строк | F1 old | F1 new | Δ, п.п. | FP old→new | FN old→new | Exact old→new |
|---|---:|---:|---:|---:|---|---|---|
| official_public_208 | 208 | 0.814815 | 0.792169 | -2.265 | 105→106 | 20→32 | 0.6731→0.6250 |
| official_public_clean_unique | 114 | 0.742358 | 0.750000 | +0.764 | 43→44 | 16→14 | 0.7281→0.7105 |
| public_holdout_729 | 729 | 0.937124 | 0.930016 | -0.711 | 112→125 | 118→131 | 0.8176→0.7888 |
| v2_test_3016 | 3016 | 0.355373 | 0.447005 | +9.163 | 781→691 | 3304→2956 | 0.2023→0.2460 |
| challenge_40 | 40 | 0.490196 | 0.488889 | -0.131 | 29→20 | 23→26 | 0.3000→0.3750 |

## Ограничения публичного теста

{
  "rows": 208,
  "unique_normalized_descriptions": 139,
  "conflicting_description_groups": 13,
  "rows_in_conflicting_groups": 56,
  "rows_overlapping_actual_mixed_train": 20,
  "rows_overlapping_any_public_split": 24,
  "clean_eligible_rows_before_dedup": 132,
  "clean_unique_rows": 114,
  "clean_positive_rows": 41,
  "policy": "Clean subset excludes every conflicting group and all text seen in any public split or mixed train/calibration; then deduplicates consistent labels. No labels corrected."
}

Полный Parquet сохраняет исходные противоречия и повторы. Очищенная часть исключает их без исправления меток, поэтому меньше и может быть смещена по составу. Показываются обе оценки. Public holdout и v2 test уже использовались в прошлых экспериментах; это сравнительная диагностика, не новый независимый финальный тест.
Confidence intervals: paired bootstrap2000, seed42; нормализованные описания группируются вместе. Для v2 prepared записей group_id не перенесён, поэтому интервал по строкам может недооценивать зависимость между аугментациями. Полные метрики и исправленные/сломанные решения сохранены вcomparison.json.
Рабочие модели и исходные данные не изменялись. Улучшение сложного теста и отсутствие деградации на реальных данных — отдельные критерии; общий смешанный F1 не заменяет их.
