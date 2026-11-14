# Все датасеты проекта

| Папка | Назначение |
| --- | --- |
| [own_v3](own_v3/README.md) | Новый собственный корпус: 4 200 объявлений и 144 диагностические записи; контролируемые категории, длины и шум |
| [hard_dataset/v2](hard_dataset/v2/README.md) | Предыдущий стрессовый корпус: 13 960 записей, готовые train/validation/test/challenge |
| [public](public/README.md) | Данные организаторов: 10 000 train и 208 test |
| distillate_dataset | 500 синтетических объявлений и 500 правок реальных объявлений |
| hard_negation_300 | 300 примеров с отрицаниями, включая исправленную разметку |
| hard_illiterate_200 | 200 примеров с языковыми ошибками |
| hard_dataset | Архив v1: 1 840 записей; генераторы, проверки и отчёты аудита |
| [eda](eda/EDA.ipynb) | Jupyter Notebook с выполненными ячейками и графиками; также HTML, Markdown и статистика |

**Для нового обучения на собственных данных начните с `own_v3/train.jsonl`.**
Основная доля ответов содержит 1–3 категории; есть короткие и развёрнутые
объявления с небольшим количеством ошибок. Public не используется генератором.
Не смешивайте готовые выборки разных версий. V2 содержит v1 целиком;
его можно оценивать отдельно как стрессовый корпус.

14 исходных файлов датасетов перенесены сюда без изменения содержимого. Генераторы
перемещены вместе с соответствующими наборами. Старые пути и SHA-256 указаны
в `migration_manifest.json`. Исторические `audit_before.json` и
`legacy_corrections.json` сохраняют пути на момент проверки; актуальные пути
есть в манифесте переноса и обновлённом `audit_report.json`.
Исторические EDA и аудит описывают эти исходные 14 файлов. Новая версия имеет
отдельные отчёт REPORT.md, stats.json и quality.json в own_v3/.

Из корня репозитория:

```bash
python3 -B datasets/own_v3/generate.py
python3 -B datasets/own_v3/quality.py
python3 -B -m unittest discover -s datasets/own_v3 -p 'test_*.py' -v
```

Для прежних корпусов и исторической EDA:

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
