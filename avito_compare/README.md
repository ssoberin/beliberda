# Сравнение классификаторов Авито

Здесь хранится **исходный код** подготовки данных, обучения TF-IDF и сравнения с
zero-shot Qwen через локальный vLLM. Данные, веса, кэши предсказаний и результаты
вычислений остаются на сервере и не входят в эту директорию Git.

Запускайте команды из `/srv/Hakaton/avito_compare` на сервере, где уже есть
`data/`, `artifacts/` и `.venv/`. Для новой установки скопируйте исходный код
из этого репозитория в отдельную рабочую директорию; установите зависимости из
`requirements.txt` и предоставьте входной датасет отдельно. Не загружайте
непроверенные файлы joblib: их десериализация может исполнить код.

```bash
cd /srv/Hakaton/avito_compare
source .venv/bin/activate
python -m scripts.prepare_dataset --input data/raw/public-train-annotated.jsonl
python -m scripts.train_baseline --data-dir data/processed --output-dir artifacts
python -m scripts.smoke_cases  # нужен локальный vLLM на 127.0.0.1:8000
python -m scripts.compare --count 50
```

`prepare_dataset.py` разделяет уникальные нормализованные описания на train,
calibration и validation. `train_baseline.py` обучает word/char TF-IDF + пять
SGD-классификаторов. `compare.py` оценивает обе модели на одинаковых примерах;
`qwen_classifier.py` используется только для эксперимента, а не в REST API.
`predict.py` выполняет локальное предсказание, `smoke_cases.py` проверяет
сложные примеры.

Метки исходных данных автоматические: метрики не равны официальной оценке.
Перед повторным сравнением с другой версией модели/промпта используйте новый
`--cache`, иначе будут взяты старые ответы Qwen. Вычисленные пороги в метриках
отличаются от порогов 0.5, сохранённых текущим `train_baseline.py` в модель.
Не запускайте обучение без явного намерения заменить артефакт.
