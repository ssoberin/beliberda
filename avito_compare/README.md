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

## Эксперимент со словесными триграммами

В `train_baseline.py` доступен `--word-ngram-max 3`. По умолчанию остаётся `2`;
символьные признаки, SGD и остальные параметры не меняются.

Для сравнения из корня репозитория, в окружении с зависимостями из этого каталога:

```bash
python -m avito_compare.scripts.prepare_dataset --input datasets/public/public-train-annotated.jsonl --output output/ngram-comparison/splits --seed 42
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m avito_compare.scripts.measure_training --data-dir output/ngram-comparison/splits --output-dir output/ngram-comparison/bigrams
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m avito_compare.scripts.measure_training --data-dir output/ngram-comparison/splits --output-dir output/ngram-comparison/trigrams --word-ngram-max 3
python -m avito_compare.scripts.compare_ngrams --source datasets/public/public-train-annotated.jsonl
```

Для повторного эксперимента используйте новый каталог результатов во всех
командах, включая `--results` у `compare_ngrams`: измеритель отказывается
перезаписывать существующие артефакты. Исходный датасет не редактируется;
подготовка применяет существующую нормализацию и удаление повторов.

Сравнение использует только пороги `0.5` для обеих моделей. Отдельно рассчитанные
пороги из `metrics.json` в сравнение не входят. `measurement.json` содержит
Macro-F1, micro Precision/Recall/F1, FP/FN, размерность, время fit и версии
зависимостей. `comparison.json` сохраняет контрольные суммы, проверки параметров
и парный bootstrap по объявлениям; `comparison.md` — итоговую таблицу.

Нормализованные эффективные описания в трёх частях проверяются на пересечение.
Близкие шаблоны с различающимися формулировками всё ещё могут присутствовать:
это ограничение существующего разбиения. Время измеряется по одному обучению
каждого варианта, поэтому разница во времени является приблизительной.

## char_wb и снятие лимитов словаря

```bash
python -m avito_compare.scripts.feature_experiments
```

Команда использует сохранённые `output/ngram-comparison/splits`, фиксирует
словесные триграммы и запускает четыре конфигурации: `char` / `char_wb` с
исходными лимитами 18000/28000 и без лимитов. Рабочая модель не заменяется.
Результаты по умолчанию записываются в новый каталог
`output/tfidf-feature-experiments`; для повторения задайте новый `--output-dir`.

Исходный `sublinear_tf=True` сохраняется во всех четырёх вариантах. Для двух
лучших по validation Micro-F1 дополнительно запускается `sublinear_tf=False`
в обоих векторизаторах. Все остальные параметры и пороги остаются одинаковыми.

Измеряются время fit, пиковый RSS процесса, размер сохранённого артефакта,
размерности обоих словарей и метрики на validation. Проверяются неизменность
данных, параметров SGD и всех параметров векторизации вне абляций. Каждый
вариант запускается в отдельном процессе с одним потоком BLAS/OpenMP.

Пределы запуска: `--rss-limit-mib 2048` и `--timeout-seconds 180`. При превышении
процесс останавливается, причина сохраняется в `status.json`. Если вариант
`uncapped_char` превышает бюджет, комбинированный вариант без лимитов также
пропускается. Логи запусков сохраняются отдельно. Сводка и абсолютные и
процентные изменения относительно baseline находятся в `summary.json` и
`comparison.md`. Подбор вариантов по этой validation не заменяет последующую
независимую проверку на новых данных.

Продолжить завершённую серию двумя вариантами с общей размерностью 100000
(40000 словесных + 60000 символьных признаков), используя прежние результаты:

```bash
python -m avito_compare.scripts.feature_experiments --include-100k --resume
```

`--resume` требует совпадения контрольных сумм исходника и разбиения. Завершённые
измерения используются повторно; новые модели сохраняются отдельно. Лимиты
можно задать и напрямую через `--word-max-features` и `--char-max-features` у
тренировщика или измерителя. Значения по умолчанию сохраняются: 18000 и 28000.

Добавить к той же серии варианты 60000 (24000 словесных + 36000 символьных):

```bash
python -m avito_compare.scripts.feature_experiments --include-100k --include-60k --resume
```

## Корпус v2 с готовыми выборками

Для v2 используйте готовые train/validation/test/challenge. Адаптер не делит
данные повторно: train остаётся train, validation становится calibration для
старого тренера, test становится его validation. На вход TF-IDF подаётся только
description. Метки обучения — offered_microcategories в проекции 102–106;
оценка — found_microcategories в той же проекции после исключения mcId строки.

```bash
python -m avito_compare.scripts.prepare_v2_experiment --output output/v2-charwb-comparison/splits
python -m avito_compare.scripts.feature_experiments --data-dir output/v2-charwb-comparison/splits --output-dir output/v2-charwb-comparison/runs --source datasets/hard_dataset/v2/train.jsonl --char-wb-sizes --exclude-source-category --skip-sublinear-check
python -m avito_compare.scripts.evaluate_v2_experiment
```

Сохранены seed разбиения корпуса 2026 и random_state SGD 42. Все пороги — 0.5.
Индивидуальные пороги в `metrics.json` не используются в эксперименте;
этот старый отчёт тренера считает offered-метки без исключения mcId. Конечные
метрики берите из `measurement.json`, `summary.json` и `evaluation.json`:
они учитывают исходную категорию. Последняя команда дополнительно проверяет
challenge и срезы, а также старые public-модели, если сохранены результаты
предыдущих экспериментов. Прежние рабочие артефакты не заменяются.

## Смесь исходного public и v2

```bash
python -m avito_compare.scripts.prepare_mixed_experiment
python -m avito_compare.scripts.feature_experiments --data-dir output/mixed-charwb-comparison/splits --output-dir output/mixed-charwb-comparison/runs --source datasets/hard_dataset/v2/train.jsonl --char-wb-sizes --exclude-source-category --skip-sublinear-check
python -m avito_compare.scripts.evaluate_mixed_experiment
```

Используются прежние разбиения: 3398 public + 8464 v2 для fit, 728 + 2440
для calibration, 729 + 3016 для проверки. Public уже очищен от одинаковых
нормализованных описаний до разделения; v2 сохраняет готовые группы и семейства.
Адаптер проверяет отсутствие одинаковых effective descriptions между частями.
TF-IDF и SGD обучаются с нуля для каждого размера. Веса строк и доли источников
дополнительно не настраиваются. Результаты public/v2/challenge выводятся отдельно
в `output/mixed-charwb-comparison/report.md`; общий F1 не подменяет эти срезы.
