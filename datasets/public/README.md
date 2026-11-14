# Данные организаторов

- `public-train-annotated.jsonl`: 10 000 объявлений со слабой автоматической разметкой.
- `public-test-dataset.parquet`: 208 записей с названиями целевых категорий.

Исходная категория — 101. Полные правила и описание схем:
[README-files.txt](../../mvp/innoglobalhack_avito/innoglobalhack_avito/README-files.txt).
Содержимое файлов при переносе не изменено; контрольные суммы находятся
в [migration_manifest.json](../migration_manifest.json).

В тесте есть одинаковые описания с разными метками и пересечения с обучением.
Подробности — в [EDA](../eda/EDA.md). Тест не следует добавлять к обучающим данным.
