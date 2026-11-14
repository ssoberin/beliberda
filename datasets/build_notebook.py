"""Build and execute EDA.ipynb, storing tables and six PNG outputs in the file."""
import argparse
from pathlib import Path
import textwrap

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent


def notebook():
    cells = []
    def md(text): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))
    def code(text): cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

    md("""
    # EDA датасетов Авито

    Исследование всех наборов: микрокатегории, длины описаний, отрицания,
    языковые ошибки, дубликаты и пересечения выборок.

    **Графики и таблицы уже сохранены в выходах ячеек.** Для просмотра запускать
    код не требуется. Откройте файл в Jupyter Notebook, JupyterLab или VS Code.

    Для пересчёта выберите Python-ядро с зависимостями из
    `datasets/requirements-notebook.txt` и выполните **Run All / Restart & Run All**.
    Для повторного запуска нужен репозиторий с папкой `datasets/`.
    Таблицы, графики и численные выводы рассчитываются заново из исходных файлов.
    """)
    md("""
    ## 1. Подготовка и загрузка

    Используем функции анализа из `datasets/eda.py`. Все 14 JSONL/Parquet
    читаются полностью. Файлы данных не изменяются. Для Parquet названия категорий
    переводятся в mcId; источник 101 принят по описанию данных организаторов.
    """)
    code('''
    from pathlib import Path
    from collections import Counter
    import hashlib
    import importlib.util
    import json
    import tempfile
    from IPython.display import display, HTML, Image, Markdown

    # Работает при запуске из корня проекта, datasets/ или datasets/eda/.
    project_root = next(
        (p for p in (Path.cwd(), *Path.cwd().parents)
         if (p / "datasets" / "eda.py").is_file()), None
    )
    if project_root is None:
        raise FileNotFoundError("Откройте notebook внутри репозитория с datasets/eda.py")
    spec = importlib.util.spec_from_file_location("avito_eda", project_root / "datasets" / "eda.py")
    eda = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eda)

    def table(headers, rows):
        report = eda.Report()
        report.table(headers, rows)
        display(HTML(report.html[0]))

    data, inventory = eda.load()
    stats = {name: eda.summarize(rows) for name, rows in data.items()}
    overlap = eda.intersections(data)
    public_train = stats[eda.PUBLIC_TRAIN]
    public_test = stats[eda.PUBLIC_TEST]
    v2_train = stats[eda.V2 + "train.jsonl"]
    display(Markdown(
        f"**Файлов:** {len(data)} · **Строк:** {sum(len(rows) for rows in data.values()):,} · "
        f"**Уникальных описаний:** {overlap['unique_texts_all_files']:,}"
    ))
    ''')
    code('''
    table(
        ["Файл относительно datasets/", "Строк", "Уникальных текстов", "Повторов сверх первого", "Пустых ответов, %"],
        [[name, s["rows"], s["unique_texts"], s["duplicate_extra_rows"],
          round(100 * s["empty_targets"] / s["rows"], 1)] for name, s in stats.items()]
    )
    ''')
    md("""
    Сумма строк всех файлов не равна размеру независимой обучающей выборки:
    v2 уже содержит v1. Контрастные пары и их искажённые версии также связаны
    между собой. Пустая целевая разметка означает отсутствие **дополнительных**
    категорий, а не обязательно отсутствие предлагаемых услуг.
    """)
    code('''
    table(
        ["Выборка v2", "Строк", "Смысловых групп", "Вариантов смысла (группа × 2)"],
        [[Path(name).stem, len(rows), len({r["group_id"] for r in rows}),
          2 * len({r["group_id"] for r in rows})]
         for name, rows in data.items() if name.startswith(eda.V2)]
    )
    print("Коротких описаний (<20 символов) в public train:", public_train["under_20_chars"])
    ''')
    md("""
    ## 2. Построение графиков

    Все шесть графиков пересчитываются по загруженным данным. Временные PNG
    используются только для передачи изображения в вывод ячейки; затем удаляются.
    Копии изображений сохраняются **внутри notebook**, а не внешними ссылками.
    """)
    code('''
    with tempfile.TemporaryDirectory(prefix="avito-notebook-figures-") as tmp:
        figure_info = eda.plots(data, stats, Path(tmp))
        figures = {name: (Path(tmp) / name).read_bytes() for name, _ in figure_info}
    captions = dict(figure_info)

    def show_chart(name):
        display(Image(data=figures[name], format="png"))
        display(Markdown("*" + captions[name] + "*"))

    print(f"Построено графиков: {len(figures)}")
    ''')
    md("""
    ## 3. Микрокатегории

    Сравниваем целевые метки, то есть `found_microcategories`. Исходная категория
    `mcId` из них исключается. Поэтому отсутствие цели 101 в public train
    не означает отсутствия комплексного ремонта: 101 там является источником.
    """)
    code('''
    train_rows = data[eda.V2 + "train.jsonl"]
    table(
        ["mcId", "Микрокатегория", "Предлагается (v2 train)", "Целевая метка", "Исключено как источник"],
        [[c, name, sum(c in r["offered_microcategories"] for r in train_rows),
          v2_train["label_counts"][str(c)],
          sum(c in r["offered_microcategories"] and r["mcId"] == c for r in train_rows)]
         for c, name in zip(eda.IDS, eda.NAMES)]
    )
    counts = list(v2_train["label_counts"].values())
    print(f"Отношение максимальной частоты метки к минимальной: {max(counts) / min(counts):.2f}")
    ''')
    code('''show_chart("01_category_prevalence.png")''')
    md("""
    ### Влияние повторов на распределение категорий

    Частоты по строкам и по уникальным описаниям могут различаться: повторяющиеся
    объявления получают больший вес. Прежде чем оставить один экземпляр текста,
    проверяем отсутствие конфликтов внутри public train.
    """)
    code('''
    public_groups = {}
    for row in data[eda.PUBLIC_TRAIN]:
        key = row["_key"]
        if key in public_groups:
            assert public_groups[key]["_labels"] == row["_labels"], "Конфликт меток: нужна проверка"
        public_groups[key] = row
    unique_public = list(public_groups.values())
    table(
        ["mcId", "По строкам, %", "По уникальным описаниям, %"],
        [[c, round(100 * public_train["label_counts"][str(c)] / public_train["rows"], 1),
          round(100 * sum(c in r["_labels"] for r in unique_public) / len(unique_public), 1)]
         for c in eda.IDS]
    )
    print("Размер самой большой группы повторов:", public_train["largest_duplicate_group"])
    ''')
    md("""
    ### Число меток на объявление и их совместное появление

    На диагонали матрицы — частота отдельной метки. В остальных ячейках — доля
    всех строк v2 train, содержащих обе метки, а не условная вероятность.
    """)
    code('''show_chart("03_label_cardinality.png")''')
    code('''show_chart("06_label_cooccurrence.png")''')
    md("""
    ## 4. Длины описаний

    Длина измеряется в символах. «Слово» — последовательность латинских или
    кириллических букв, а не токен LLM. На boxplot используется логарифмическая
    шкала; скрытые на графике выбросы остаются в расчётах статистики.
    """)
    code('''
    table(
        ["Набор", "Символов p50", "p95", "max", "Слов p50", ">6000 символов", ">12000 символов"],
        [[eda.short(name), s["characters"]["median"], s["characters"]["p95"],
          s["characters"]["max"], s["words"]["median"], s["over_6000_chars"], s["over_12000_chars"]]
         for name, s in stats.items()]
    )
    show_chart("02_text_lengths.png")
    print("Описаний длиннее лимита обучения 6000:", sum(s["over_6000_chars"] for s in stats.values()))
    print("Описаний длиннее лимита API 12000:", sum(s["over_12000_chars"] for s in stats.values()))
    ''')
    md("""
    ## 5. Отрицания и языковые ошибки

    Количество отдельных слов НЕ/НИ/НЕТ — лексический показатель, не глубина
    отрицания. Например, «неделаю» не попадёт в этот счётчик. Смешение кириллицы
    и латиницы может быть обфускацией, артикулом или названием; оно не доказывает
    грамматическую ошибку. Глубина и тип искажений взяты из аннотаций синтетики.
    «Не размечено» не означает отсутствие отрицаний или ошибок.
    """)
    code('''
    marker_fields = ["standalone_negation_rows", "two_or_more_negation_tokens_rows",
                     "only_marker_rows", "mixed_script_word_rows"]
    table(
        ["Набор", "НЕ/НИ/НЕТ, %", "≥2 таких слов, %", "ТОЛЬКО/ЛИШЬ/ИСКЛЮЧИТЕЛЬНО, %", "Смешение алфавитов, %"],
        [[eda.short(name), *[round(100 * s[key] / s["rows"], 1) for key in marker_fields]]
         for name, s in stats.items()]
    )
    show_chart("05_v2_coverage.png")
    ''')
    md("""
    ## 6. Дубликаты и пересечения выборок

    Нормализация: `casefold` и свёртка пробелов. Пунктуация, частицы и опечатки
    сохраняются. Заголовок не учитывается, поскольку сервис получает описание.
    Разные заголовки могут объяснять часть исходной разметки, но для модели,
    которая получает только описание, неоднозначность сохраняется.
    """)
    code('''
    show_chart("04_text_overlap.png")
    table(
        ["Файл A", "Файл B", "Общих уникальных текстов", "Строк A", "Строк B"],
        [[p["a"], p["b"], p["unique_texts"], p["rows_a"], p["rows_b"]]
         for p in overlap["overlaps"]]
    )
    ''')
    md("""
    ### Конфликтующие метки

    Ниже — адреса всех строк конфликтных групп. Одинаковое описание считается
    конфликтующим, только если при одинаковом исходном `mcId` есть разные наборы
    целей. Автоматически выбирать разметку большинством здесь нельзя.
    """)
    code('''
    conflicts = overlap["conflicting_text_groups"]
    print("Групп конфликтующих описаний:", len(conflicts))
    conflict_rows = []
    for group_no, group in enumerate(conflicts, 1):
        for entry in group["occurrences"]:
            conflict_rows.append([group_no, entry["file"], entry["line"], entry["item_id"],
                                  entry["source"], ", ".join(map(str, entry["labels"])) or "∅"])
    table(["Группа", "Файл", "Строка", "item_id", "Источник", "Целевые категории"], conflict_rows)
    ''')
    md("""
    ### Проверка разбиения и происхождения

    Контролируемые правки должны оставаться в одной выборке со своими исходниками
    по `source_item_id`. Для v2 отдельно проверяем тексты, смысловые группы и
    семейства формулировок. Отсутствие точных совпадений не доказывает полной
    семантической независимости: общая лексика и логические операции сохраняются.
    """)
    code('''
    assert not overlap["v2_group_leaks"], "Смысловые группы пересекают выборки"
    assert not overlap["v2_family_leaks"], "Семейства шаблонов пересекают выборки"
    assert not any(p["a"].startswith(eda.V2) and p["b"].startswith(eda.V2)
                   for p in overlap["overlaps"]), "Тексты пересекают выборки v2"
    edits = data["distillate_dataset/hard_edits_500.jsonl"]
    print("v2: пересечений текстов, групп и семейств между выборками нет.")
    print("Контролируемых правок:", len(edits),
          "| Исходных объявлений:", len({r["source_item_id"] for r in edits}))
    table(["Входной файл", "SHA-256"], [[r["path"], r["sha256"]] for r in inventory])
    ''')
    md("""
    ## 7. Выводы по текущим данным
    """)
    code('''
    common = next(p for p in overlap["overlaps"]
                  if {p["a"], p["b"]} == {eda.PUBLIC_TRAIN, eda.PUBLIC_TEST})
    train_matches = common["rows_a"] if common["a"] == eda.PUBLIC_TRAIN else common["rows_b"]
    test_matches = common["rows_a"] if common["a"] == eda.PUBLIC_TEST else common["rows_b"]
    display(Markdown(
        f"- Public train: **{public_train['unique_texts']:,} уникальных описаний** "
        f"из {public_train['rows']:,} строк; "
        f"{100 * public_train['duplicate_extra_rows'] / public_train['rows']:.1f}% — повторы сверх первого.\\n"
        f"- Public train/test: **{common['unique_texts']} общих описаний**, "
        f"{train_matches} строк train и {test_matches} строк test.\\n"
        f"- Найдено **{len(conflicts)} конфликтных групп**.\\n"
        f"- V2 train: медианная длина **{v2_train['characters']['median']:g} символа**, "
        f"public train — **{public_train['characters']['median']:g}**.\\n"
        f"- Пустые ответы в v2 train: **{100 * v2_train['empty_targets'] / v2_train['rows']:.1f}%**.\\n"
        "- Внутри v2 пересечений текстов, групп и семейств между выборками нет."
    ))
    ''')
    md("""
    ## 8. Рекомендации и ограничения

    - Использовать готовые train/validation/test v2; **не добавлять v1 повторно**.
      Реальные объявления разделять после дедупликации, вместе с их производными.
    - Публичный тест оставить вне обучения. Для оценки обобщения нужна независимая
      проверенная реальная выборка без конфликтов и пересечений с обучением.
    - Подбирать долю синтетики на реальной валидации. Короткие формальные фразы
      не заменяют длинные реальные объявления.
    - Для модели только по описанию обучать на `offered_microcategories`, затем
      исключать исходный `mcId`. Для модели с описанием и `mcId` допустимы цели
      `found_microcategories`. Текущий baseline поддерживает только 102–106;
      для шести категорий требуется добавить 101.
    - Не передавать модели поля `note`, `type`, `group_id`, `variant`,
      `augmentation`, `negation_depth` и метки в качестве входных признаков.
    - Измерять micro-F1, F1 по категориям, точное совпадение, ложные положительные
      на пустых ответах, полностью правильные контрастные пары и устойчивость
      к искажённым версиям. Пороги выбирать на validation, не на test.

    Модель в этом notebook не обучается. EDA не доказывает качество каждой метки
    и не заменяет ручную разметку. Смысловая глубина — аннотация сценария;
    автоматического надёжного подсчёта всех грамматических ошибок здесь нет.
    Подробный исторический аудит: `datasets/hard_dataset/AUDIT.md`.
    """)
    return nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "eda" / "EDA.ipynb")
    args = parser.parse_args()
    nb = notebook()
    NotebookClient(nb, timeout=180, kernel_name="python3",
                   resources={"metadata": {"path": str(ROOT.parent)}}).execute()
    nbformat.validate(nb)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    images = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
    assert images == 6, images
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, args.output)
    print(f"Saved {args.output}: {len(code_cells)} executed code cells, {images} embedded PNGs")


if __name__ == "__main__":
    main()
