# Все датасеты проекта

| Папка | Назначение |
| --- | --- |
| [own_v3_clean](own_v3_clean/README.md) | Очищенный v3 для следующих экспериментов: train2921, validation600, test600, challenge144; исключены79текстов |
| [own_v3_revision3](own_v3_revision3/README.md) | Эксперимент с250заменами; по измерениям не рекомендован как основнойtrain |
| [context_pairs_100](context_pairs_100/README.md) | Отдельные100обучающих и20контрольныхтекстов |
| [hard_dataset/v2](hard_dataset/v2/README.md) | Исторический корпус: 13 960 записей, готовые train/validation/test/challenge |
| [public](public/README.md) | Данные организаторов: 10 000 train и 208 test |
| distillate_dataset | 500 синтетических объявлений и 500 правок реальных объявлений |
| hard_negation_300 | 300 примеров с отрицаниями, включая исправленную разметку |
| hard_illiterate_200 | 200 примеров с языковыми ошибками |
| hard_dataset | Архив v1: 1 840 записей; генераторы, проверки и отчёты аудита |
| [eda](eda/EDA.ipynb) | Jupyter Notebook с выполненными ячейками и графиками; также HTML, Markdown и статистика |

**Для следующих экспериментов с v3 используйте `own_v3_clean/train.jsonl`.**
Результат модели после очистки ещё не измерен. Publictrainиавторские100при смешивании сохраняются отдельно.
Исторический v2 уже содержит v1: не складывайте их вместе. Чистые и искажённые варианты, контрастные пары и
семейства формулировок должны оставаться в своих готовых выборках.
Рекомендации по смешиванию с реальными данными находятся в EDA.

14 файлов датасетов перенесены сюда без изменения содержимого. Генераторы
перемещены вместе с соответствующими наборами. Старые пути и SHA-256 указаны
в `migration_manifest.json`. Исторические `audit_before.json` и
`legacy_corrections.json` сохраняют пути на момент проверки; актуальные пути
есть в манифесте переноса и обновлённом `audit_report.json`.

Из корня репозитория:

```bash
python3 datasets/hard_dataset/build_v2.py
python3 -m unittest discover -s datasets/hard_dataset -p 'test_*.py' -v
python3 datasets/hard_dataset/audit.py
python3 datasets/eda.py
```

Для EDA и чтения Parquet установите зависимости из `requirements-eda.txt`
в используемое окружение Python. Генераторы JSONL работают без сторонних пакетов.

### EDA в Jupyter Notebook

Откройте [eda/EDA.ipynb](eda/EDA.ipynb) в Jupyter Notebook, JupyterLab или VS Code:
шесть графиков и таблицы сохранены внутри файла и видны без запуска кода.
Для пересчёта выберите Python-ядро с зависимостями из
`requirements-notebook.txt` и нажмите **Run All**. Репозиторий с данными должен
оставаться рядом с notebook.

Полностью пересобрать и выполнить notebook из корня проекта:

```bash
python3 datasets/build_notebook.py
```
