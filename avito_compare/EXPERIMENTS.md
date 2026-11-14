# Прогресс экспериментов — test-branch

Эта ветка хранит код, измерения и выводы экспериментов. Веса моделей не продвигаются
в production автоматически. Данные v2 проверены на snapshot main `0b1be5d`;
следующее объединение с main включает и дальнейшие изменения проекта.

## Что открыть

- [Общий notebook с выполненными ячейками](../output/experiment-review/experiments.ipynb)
- [Длины текстов: notebook и графики](../output/dataset-length-comparison/lengths.ipynb)
- [Парное сравнение public-only / public+v2](../output/paired-data-evaluation/report.md)
- [Результаты смешанного обучения](../output/mixed-charwb-comparison/report.md)
- [EDA v2 и реалистичность сложных примеров](../output/v2-eda/report.md)

## Сохранённые точки

| Точка | Данные / изменение | Результат | Измерения |
|---|---|---|---|
| 1 | Public, word(1,2) →(1,3), analyzer=char | Micro-F1 0.93465→0.93622; слабый прирост | [ngram](../output/ngram-comparison/comparison.json) |
| 2 | Public, char→char_wb, лимиты / без лимитов | Лучший измеренный char_wb46k; снятие лимитов хуже | [features](../output/tfidf-feature-experiments/summary.json) |
| 3 | Public, 60k и100k вместо46k | Качество не улучшилось | [таблица](../output/tfidf-feature-experiments/comparison.md) |
| 4 | Только v2 train, char_wb46/60/100k | Доступно10692 признака; модели одинаковы, testF1 0.46081 | [v2](../output/v2-charwb-comparison/report.md) |
| 5 | Public train3398 +v2 train8464 | Реальный holdout около0.93, v2test около0.447 | [mixed](../output/mixed-charwb-comparison/evaluation.json) |
| 6 | Одинаковая архитектура: до/после добавления v2 | V2 0.35537→0.44701; public holdout0.93712→0.93002; official208 0.81481→0.79217 | [paired](../output/paired-data-evaluation/comparison.json) |
| 7 | Независимый EDA v2 и длин | Дедуп public median1467 против v2train143 символов; высокая доля искусственных отрицаний | [lengths](../output/dataset-length-comparison/stats.json) |

Все сравнения используют scope102–106, исключение исходного mcId и пороги0.5.
Исключение — старые вспомогательные поля `validation_calibrated` в metrics.json:
это диагностический поиск порогов, он не используется в основных таблицах.
Macro-F1, precision/recall, FP/FN, время fit, размерности, память и размеры
артефактов находятся в measurement/summary JSON. Замеры времени одиночные.

Полный public208 содержит противоречивые метки и пересечения с исходным корпусом.
Дополнительно показаны114 уникальных записей без конфликтов и пересечений:
F1 0.74236→0.75000. Размер поднабора мал, состав изменён, уверенного улучшения нет.
Public-разметка слабая автоматическая; v2 не представляет обычный реальный поток.
Выбор конфигураций уже использовал эти тесты — нужна независимая финальная проверка.

## Как воспроизводить

Создайте отдельное Python-окружение и установите
`pip install -r avito_compare/requirements-experiments.txt`.
Команды подготовки и обучения: [README](README.md). Все исходники находятся
в `datasets/`; старые пути mvp в исторических замерах сохранены как provenance.
Для воспроизведения notebook не нужны веса или подготовленные выборки:

```bash
python -m avito_compare.scripts.build_experiment_notebook
python output/dataset-length-comparison/build_notebook.py
python -m unittest discover -s avito_compare/tests -p 'test_*.py' -v
```

`output/` содержит только выбранные отчёты, JSON-метрики, графики, выполненные
notebook, скрипты EDA и небольшие логи/манифесты. Большие `.joblib`, подготовленные
JSONL, NPZ с предсказаниями, временные окружения и кэши не коммитятся. Сводка
переносится между машинами без абсолютных путей; исторические команды в логах
могут сохранять путь использованного временного Python.

## Следующая точка

Подобрать долю/вес синтетики по реальной validation и отдельно оценивать обычные
отрицания и формальные стрессовые конструкции. Каждую новую серию записывать
в новый каталог, сохранять snapshot исходников, SHA-256 данных/разбиений,
параметры, версии, все срезы и вывод. Старые измерения не заменять новыми.
