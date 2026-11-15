# CatBoost optimization: воспроизведение

Запускать из корня репозитория. Это завершённая серия из 45 обучений на исходной C_syntax без дополнительных 200 отрицательных примеров. Результаты и решение — в [report.md](report.md), смысловая проверка public test — в [public_test_semantic_review.md](public_test_semantic_review.md).

Нужны исходные данные, сохранённые Qwen embeddings и синтаксические признаки из `output/cat-aware-v2/` и `output/syntax-aware-experiment/`. Большие кэши и бинарные веса сохранены локально, в Git не включаются. Для численно точного воспроизведения использовать исходные артефакты и проверять SHA-256 из `input_manifest.json`, `cache/prepare_signature.json` и `finalists.json`. Для визуализации нужны nbformat, nbclient, ipykernel и matplotlib; notebook этой серии выполнен через `/tmp/innopolis-ngram-venv/bin/python`. Рабочий Python обучения этой серии: `/tmp/innopolis-qwen-emb-venv/bin/python` (версии пакетов — `environment.json`).

Последовательность для нового запуска с исходными артефактами:

```bash
mkdir -p output/catboost-optimization/cache output/catboost-optimization/models
python output/catboost-optimization/prepare.py
python output/catboost-optimization/search.py
python output/catboost-optimization/evaluate_finalists.py
python output/catboost-optimization/analyze_public.py
python output/catboost-optimization/benchmark_tfidf.py
python output/catboost-optimization/build_report.py
python output/catboost-optimization/build_notebook.py
python output/catboost-optimization/verify_results.py
```

`prepare.py` восстанавливает nested OOF для исходного SGD и отдельный grouped OOF для TF-IDF. Векторизатор обучается внутри train каждого фолда; near-duplicate группы не пересекаются между train и holdout. Исходная CatBoost воспроизведена с нулевой разницей вероятностей на calibration. Старые OOF от дополненного train не использованы.

`search.py` читает только train/calibration. 34 исходные конфигурации и 12 адаптивных комбинаций дали 45 различных обучений, включая baseline. Все параметры, колонки, длительности и результаты трёх режимов порогов сохранены в `experiments.json` и `calibration_metrics.csv`. Тестовые метки не участвуют в отборе. Последний шаг сохраняет `finalists.json` с хешами весов.

`evaluate_finalists.py` проверяет хеши и оценивает финалистов один раз. При существующем `final_metrics.json` повторный запуск запрещён. Для завершённой серии не удалять этот файл ради повторного подбора. Скрипты анализа и построения отчёта используют только уже зафиксированные результаты. Групповой bootstrap учитывает повторяющиеся тексты public test.

`public_test_clean` содержит 152 строки: удалены все 56 строк из 13 групп с конфликтующим gold. Список фиксирован в `public_test_exclusions.json` и одинаков для всех моделей. Gold не исправлялся; остальные повторы и пересечения с train сохранены. `source_mcId=101` для official_208 — документированное допущение: поля в parquet нет.

Финалисты сохранены отдельно в `models/`, с параметрами, порогами, колонками и SHA-256 в `configurations/`. Дополнительный TF-IDF использует прежний `tfidf_baseline.joblib`; основная Qwen/SGD-часть не меняется. Основная модель не заменена. Colab не использовался.

`final_predictions.csv` содержит вероятности и решения по парам «объявление × категория». `changed_examples.json` — исходные тексты всех изменившихся предсказаний. `results.ipynb` и `comparison.png` строятся по готовым метрикам без нового обучения и отбора.
