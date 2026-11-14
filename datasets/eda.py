"""Reproducible EDA of every JSONL/Parquet corpus, with standalone HTML and PNGs."""
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/avito-eda-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
IDS = list(range(101, 107))
NAMES = ["Ремонт под ключ", "Штукатурные работы", "Плиточные работы", "Сантехника", "Электрика", "Поклейка обоев и малярные работы"]
BY_NAME = dict(zip(NAMES, IDS))
NEG = re.compile(r"\b(?:не|ни|нет)\b", re.I)
WORDS = re.compile(r"[A-Za-zА-Яа-яЁё]+")
V2 = "hard_dataset/v2/"
PUBLIC_TRAIN = "public/public-train-annotated.jsonl"
PUBLIC_TEST = "public/public-test-dataset.parquet"


def normalized(text):
    return " ".join(text.casefold().split())


def load():
    datasets, manifest = {}, []
    for folder in ("public", "distillate_dataset", "hard_dataset", "hard_negation_300", "hard_illiterate_200"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.suffix not in {".jsonl", ".parquet"}:
                continue
            name = path.relative_to(ROOT).as_posix()
            if path.suffix == ".parquet":
                rows = pq.read_table(path).to_pylist()
                for i, row in enumerate(rows, 1):
                    row.update(item_id=f"parquet-row-{i}", mcId=101,
                               found_microcategories=[{"mcId": BY_NAME[n], "label": n} for n in row["categories"]])
            else:
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            for row in rows:
                row.setdefault("mcId", 101)
                row["_labels"] = sorted(m["mcId"] for m in row["found_microcategories"])
                row["_key"] = normalized(row["description"])
                assert set(row["_labels"]) <= set(IDS)
                assert row["mcId"] not in row["_labels"]
            datasets[name] = rows
            manifest.append({"path": name, "rows": len(rows), "bytes": path.stat().st_size,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return dict(sorted(datasets.items())), manifest


def distribution(values):
    return dict(sorted(Counter(values).items(), key=lambda kv: str(kv[0])))


def summarize(rows):
    lengths = np.array([len(r["description"]) for r in rows])
    word_lengths = np.array([len(WORDS.findall(r["description"])) for r in rows])
    neg_counts = [len(NEG.findall(r["description"])) for r in rows]
    keys = Counter(r["_key"] for r in rows)
    return {
        "rows": len(rows), "unique_texts": len(keys), "duplicate_extra_rows": len(rows) - len(keys),
        "duplicate_groups": sum(n > 1 for n in keys.values()),
        "largest_duplicate_group": max(keys.values()),
        "empty_targets": sum(not r["_labels"] for r in rows),
        "label_counts": {str(c): sum(c in r["_labels"] for r in rows) for c in IDS},
        "source_counts": distribution(r["mcId"] for r in rows),
        "cardinality": distribution(len(r["_labels"]) for r in rows),
        "characters": {k: round(float(v), 1) for k, v in zip(("min", "p25", "median", "p75", "p95", "p99", "max"), np.percentile(lengths, [0, 25, 50, 75, 95, 99, 100]))},
        "words": {k: round(float(v), 1) for k, v in zip(("median", "p95", "max"), np.percentile(word_lengths, [50, 95, 100]))},
        "over_6000_chars": int((lengths > 6000).sum()), "over_12000_chars": int((lengths > 12000).sum()),
        "under_20_chars": sum(len(r["description"].strip()) < 20 for r in rows),
        "standalone_negation_rows": sum(n > 0 for n in neg_counts),
        "two_or_more_negation_tokens_rows": sum(n >= 2 for n in neg_counts),
        "only_marker_rows": sum(bool(re.search(r"\b(?:только|лишь|исключительно)\b", r["description"], re.I)) for r in rows),
        "mixed_script_word_rows": sum(any(re.search('[A-Za-z]', w) and re.search('[А-Яа-яЁё]', w) for w in WORDS.findall(r["description"])) for r in rows),
        "augmentations": distribution(r.get("augmentation", "not_annotated") for r in rows),
        "negation_depth": distribution(r.get("negation_depth", "not_annotated") for r in rows),
        "topics": distribution(r.get("topic", "not_annotated") for r in rows),
    }


def intersections(data):
    keys = {name: {r["_key"] for r in rows} for name, rows in data.items()}
    pairs = []
    names = list(data)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            common = keys[a] & keys[b]
            if common:
                pairs.append({"a": a, "b": b, "unique_texts": len(common),
                              "rows_a": sum(r["_key"] in common for r in data[a]),
                              "rows_b": sum(r["_key"] in common for r in data[b])})
    all_texts = defaultdict(list)
    for name, rows in data.items():
        for i, row in enumerate(rows, 1):
            all_texts[row["_key"]].append({"file": name, "line": i, "item_id": row["item_id"],
                                           "source": row["mcId"], "labels": row["_labels"]})
    conflicts = []
    for key, entries in all_texts.items():
        states = defaultdict(set)
        for e in entries:
            states[e["source"]].add(tuple(e["labels"]))
        if any(len(s) > 1 for s in states.values()):
            conflicts.append({"text_sha256": hashlib.sha256(key.encode()).hexdigest(), "occurrences": entries})
    v2_names = [n for n in names if n.startswith(V2)]
    group_leaks = []
    family_leaks = []
    for field, result in (("group_id", group_leaks), ("type", family_leaks)):
        owners = defaultdict(set)
        for name in v2_names:
            for row in data[name]:
                owners[row[field]].add(name)
        result.extend({"value": k, "splits": sorted(v)} for k, v in owners.items() if len(v) > 1)
    return {"unique_texts_all_files": len(all_texts), "overlaps": pairs,
            "conflicting_text_groups": conflicts, "v2_group_leaks": group_leaks, "v2_family_leaks": family_leaks}


def short(name):
    if name.startswith(V2): return "v2 / " + Path(name).stem
    if name.startswith("hard_dataset/"): return "v1 / " + Path(name).stem
    return {PUBLIC_TRAIN: "public / train", PUBLIC_TEST: "public / test"}.get(name, Path(name).stem)


def plots(data, stats, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f8fafc", "axes.facecolor": "#ffffff",
                         "savefig.facecolor": "#f8fafc", "axes.titleweight": "bold"})
    names = list(data)
    figures = []
    def save(fig, name, caption):
        fig.savefig(output / name, dpi=155, bbox_inches="tight")
        plt.close(fig)
        figures.append((name, caption))

    fig, ax = plt.subplots(figsize=(11, 7.5), layout="constrained")
    matrix = np.array([[100 * stats[n]["label_counts"][str(c)] / stats[n]["rows"] for c in IDS] for n in names])
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set(xticks=range(6), xticklabels=IDS, yticks=range(len(names)), yticklabels=[short(n) for n in names], title="Доля записей с целевой микрокатегорией, %", xlabel="Исходная категория исключена из целевых меток")
    for i in range(len(names)):
        for j in range(6): ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", color="white" if matrix[i,j] > 55 else "#14263d", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=.8, label="% записей")
    save(fig, "01_category_prevalence.png", "Частоты целевых категорий по всем файлам. Частоты зависят также от распределения исходного mcId.")

    fig, ax = plt.subplots(figsize=(11, 7.5), layout="constrained")
    ax.boxplot([[len(r["description"]) for r in data[n]] for n in names], orientation="horizontal", tick_labels=[short(n) for n in names], showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#99d6d0"}, medianprops={"color": "#075b58"})
    ax.set(xscale="log", xlabel="Символы, логарифмическая шкала", title="Длина описаний: синтетические и реальные данные")
    ax.axvline(6000, color="#c04f27", linestyle="--", label="Обрезка train_baseline: 6 000")
    ax.axvline(12000, color="#7d415c", linestyle=":", label="Лимит API: 12 000")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "02_text_lengths.png", "Коробки показывают медиану и квартили; выбросы скрыты только на графике, статистика рассчитана по всем строкам.")

    selected = [PUBLIC_TRAIN, PUBLIC_TEST, V2+"train.jsonl", V2+"validation.jsonl", V2+"test.jsonl", V2+"challenge.jsonl"]
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    bottom = np.zeros(len(selected))
    for c, color in zip(range(6), ["#cbd5e1", "#9ccbc7", "#4caaa4", "#23847e", "#315779", "#17324d"]):
        heights = [100 * stats[n]["cardinality"].get(c, 0) / len(data[n]) for n in selected]
        ax.bar(range(len(selected)), heights, bottom=bottom, color=color, label=str(c))
        bottom += heights
    ax.set(xticks=range(len(selected)), xticklabels=[short(n) for n in selected], ylabel="% записей", title="Число дополнительных категорий в одном объявлении", ylim=(0, 100))
    ax.legend(title="Целевых меток", ncols=6, loc="upper center", bbox_to_anchor=(.5, -.12))
    save(fig, "03_label_cardinality.png", "Ноль означает отсутствие дополнительных черновиков, а не обязательно отсутствие предлагаемых услуг.")

    keys = {n: {r["_key"] for r in rows} for n, rows in data.items()}
    matrix = np.array([[len(keys[a] & keys[b]) if a != b else 0 for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(12, 10), layout="constrained")
    im = ax.imshow(np.log1p(matrix), cmap="Blues")
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=[short(n) for n in names], yticklabels=[short(n) for n in names], title="Пересечения: число одинаковых нормализованных описаний")
    plt.setp(ax.get_xticklabels(), rotation=55, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if matrix[i,j]: ax.text(j, i, str(matrix[i,j]), ha="center", va="center", fontsize=8, color="white" if matrix[i,j] > 100 else "#17324d")
    fig.colorbar(im, ax=ax, shrink=.7, label="Цвет: ln(1 + пересечение); числа: тексты")
    save(fig, "04_text_overlap.png", "Диагональ скрыта. Повтор v1 внутри v2 ожидаем; пересечение public train/test влияет на независимость оценки.")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    splits = [V2+s+".jsonl" for s in ("train", "validation", "test")]
    modes = ["not_annotated", "clean", "spelling", "grammar_spacing", "authored_errors"]
    labels = ["Унаследовано v1", "Исходные v2", "Опечатки", "Грамматика", "Разговорные"]
    colors = ["#cbd5e1", "#17324d", "#23847e", "#6bbcb5", "#de9a57"]
    bottom = np.zeros(3)
    for mode, label, color in zip(modes, labels, colors):
        values = [100 * stats[n]["augmentations"].get(mode, 0) / len(data[n]) for n in splits]
        axes[0].bar(range(3), values, bottom=bottom, label=label, color=color)
        bottom += values
    bottom = np.zeros(3)
    for depth, color in zip(["not_annotated", 1, 2, 3, 4], colors):
        values = [100 * stats[n]["negation_depth"].get(depth, 0) / len(data[n]) for n in splits]
        axes[1].bar(range(3), values, bottom=bottom, label="Не размечено" if isinstance(depth, str) else str(depth), color=color)
        bottom += values
    for ax, title in zip(axes, ["Тип искажения", "Авторская глубина отрицания"]):
        ax.set(xticks=range(3), xticklabels=["train", "validation", "test"], ylim=(0,100), ylabel="% записей", title=title)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(.5,-.12), ncols=2)
    save(fig, "05_v2_coverage.png", "Глубина — аннотация сценария, не подсчёт НЕ. Не размечено не означает отсутствие отрицаний.")

    train = data[V2+"train.jsonl"]
    matrix = np.array([[100 * sum(a in r["_labels"] and b in r["_labels"] for r in train) / len(train) for b in IDS] for a in IDS])
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    im = ax.imshow(matrix, cmap="YlGnBu")
    ax.set(xticks=range(6), xticklabels=IDS, yticks=range(6), yticklabels=IDS, title="Совместное появление целевых меток: v2 train")
    for i in range(6):
        for j in range(6): ax.text(j,i,f"{matrix[i,j]:.1f}%",ha="center",va="center",color="white" if matrix[i,j]>20 else "#17324d")
    fig.colorbar(im, ax=ax, label="% всех записей train")
    save(fig, "06_label_cooccurrence.png", "Диагональ — частота отдельной метки; остальные ячейки — доля записей, содержащих обе метки.")
    return figures


class Report:
    def __init__(self): self.md, self.html = [], []
    def heading(self, text, level=2):
        self.md.append("#" * level + " " + text)
        self.html.append(f"<h{level}>{html.escape(text)}</h{level}>")
    def paragraph(self, text):
        self.md.append(text)
        self.html.append("<p>" + html.escape(text) + "</p>")
    def table(self, headers, rows):
        self.md.append("\n".join(["| " + " | ".join(map(str, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(map(str,row)) + " |" for row in rows]))
        self.html.append('<div class="table"><table><thead><tr>' + ''.join('<th>'+html.escape(str(v))+'</th>' for v in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>')
    def figure(self, output, name, caption):
        self.md.append(f"![{caption}]({name})")
        encoded = base64.b64encode((output/name).read_bytes()).decode()
        self.html.append(f'<figure><img alt="{html.escape(caption, quote=True)}" src="data:image/png;base64,{encoded}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    def save(self, output):
        (output/"EDA.md").write_text("\n\n".join(self.md)+"\n", encoding="utf-8")
        style = "body{margin:0;background:#f2f5f8;color:#17324d;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px}h1{font-size:38px;line-height:1.2}h2{margin-top:46px;border-top:2px solid #d8e4eb;padding-top:22px}p{max-width:1000px}.table{overflow-x:auto;background:white;border-radius:10px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e8f0}th{background:#17324d;color:white;white-space:nowrap}figure{margin:24px 0;background:white;padding:12px;border-radius:12px}img{width:100%;height:auto}figcaption{font-size:13px;color:#526779;padding:8px}footer{margin-top:40px;color:#526779}@media print{body{background:white}main{padding:0}figure,table{break-inside:avoid}h2{break-after:avoid}}"
        page = '<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Avito — EDA датасетов</title><style>'+style+'</style></head><body><main>'+''.join(self.html)+'<footer>Воспроизводимый анализ • datasets/eda.py • без внешних ресурсов</footer></main></body></html>'
        (output/"EDA.html").write_text(page,encoding="utf-8")


def write_report(data, stats, overlap, manifest, figures, output):
    report = Report()
    total = sum(len(r) for r in data.values())
    ptrain, ptest = stats[PUBLIC_TRAIN], stats[PUBLIC_TEST]
    train = stats[V2+"train.jsonl"]
    public_common = next(p for p in overlap["overlaps"] if {p["a"],p["b"]} == {PUBLIC_TRAIN,PUBLIC_TEST})
    common_train = public_common['rows_a'] if public_common['a'] == PUBLIC_TRAIN else public_common['rows_b']
    common_test = public_common['rows_a'] if public_common['a'] == PUBLIC_TEST else public_common['rows_b']
    report.heading("EDA датасетов Авито", 1)
    report.paragraph(f"Проверены все {len(data)} файлов: {total:,} строк, {overlap['unique_texts_all_files']:,} уникальных нормализованных описаний. Это инвентаризация всех версий, а не единая обучающая выборка. V2 содержит v1 целиком. Содержимое всех 14 файлов при переносе в datasets/ сохранено; SHA-256 зафиксированы в migration_manifest.json и inventory.json.")
    report.heading("Основные выводы")
    report.paragraph(f"В public train {ptrain['rows']:,} строк, но только {ptrain['unique_texts']:,} уникальных описаний ({100*ptrain['duplicate_extra_rows']/ptrain['rows']:.1f}% строк — дополнительные повторы). Случайное разделение по строкам создаёт риск утечки и завышения метрик. В public test — {ptest['unique_texts']} уникальных описаний из {ptest['rows']} строк.")
    report.paragraph(f"Public train и test пересекаются по {public_common['unique_texts']} уникальным описаниям: это затрагивает {common_train} строк public train и {common_test} строки public test. Во всех данных найдено {len(overlap['conflicting_text_groups'])} групп одинаковых описаний с разными целевыми метками при одинаковом исходном mcId. Все такие случаи доступны построчно в overlaps.json.")
    report.paragraph(f"V2 train содержит {train['rows']:,} строк. Доля пустых ответов — {100*train['empty_targets']/train['rows']:.1f}%. Медиана длины — {train['characters']['median']:g} символов против {ptrain['characters']['median']:g} в public train. Синтетические конструкции заметно отличаются от реальных объявлений; качество на них нельзя переносить на реальный поток без проверки.")
    report.paragraph(f"В v2 пересечений нормализованных текстов между выборками нет; пересечений group_id: {len(overlap['v2_group_leaks'])}, семейств type: {len(overlap['v2_family_leaks'])}. Модель не обучалась: этот отчёт описывает данные, а не точность классификации.")
    report.heading("1. Состав и качество структуры")
    report.table(["Файл относительно datasets/", "Строк", "Уникальных текстов", "Повторов сверх первого", "Пустых ответов, %"], [[n, s["rows"], s["unique_texts"], s["duplicate_extra_rows"], f"{100*s['empty_targets']/s['rows']:.1f}"] for n,s in stats.items()])
    report.paragraph("JSONL читается целиком, категории проверяются по диапазону 101–106. В Parquet поле categories содержит названия категорий, которые сопоставлены mcId. Для публичных файлов источник 101 задан по README организаторов. В остальных файлах используется явный mcId. Заголовок не входит в сравнение текстов: текущий сервис получает description. Различающиеся заголовки могут объяснять часть исходных меток, но не устраняют неоднозначность для такого сервиса.")
    report.paragraph(f"В public train есть {ptrain['under_20_chars']} описаний короче 20 символов, в том числе одиночные буквы и числовые заглушки. Эти строки стоит отдельно проверить на информативность; краткость сама по себе не доказывает ошибку метки. Они сохранены в данных и учтены во всех расчётах.")
    report.table(["Выборка v2", "Строк", "Смысловых групп", "Вариантов смысла (группа × 2)"], [[Path(n).stem, len(rows), len({r['group_id'] for r in rows}), 2*len({r['group_id'] for r in rows})] for n,rows in data.items() if n.startswith(V2)])
    report.paragraph("Смысловая группа объединяет контрастную пару и её искажённые версии. Эти строки полезны для обучения устойчивости, но не являются независимыми новыми сюжетами.")
    report.heading("2. Микрокатегории и число меток")
    report.table(["ID", "Микрокатегория", "Предлагается: v2 train", "Целевая: v2 train", "Исключено как источник"], [[c,n, sum(c in r['offered_microcategories'] for r in data[V2+'train.jsonl']), train['label_counts'][str(c)], sum(c in r['offered_microcategories'] and r['mcId']==c for r in data[V2+'train.jsonl'])] for c,n in zip(IDS,NAMES)])
    values = list(train["label_counts"].values())
    report.paragraph(f"Отношение максимальной частоты целевой метки к минимальной в v2 train: {max(values)/min(values):.2f}. Нулевую частоту 101 в public нельзя считать отсутствием комплексного ремонта: 101 там является исходной категорией и исключается из ответа. Для модели только по тексту обучающая цель — offered_microcategories; удалять исходный mcId нужно после предсказания. Текущий сервис и train_baseline поддерживают только 102–106, поэтому для полной задачи потребуется голова 101.")
    unique_public = list({r['_key']:r for r in data[PUBLIC_TRAIN]}.values())
    report.table(["Категория", "Public train: по строкам, %", "По уникальным текстам, %"], [[c, f"{100*ptrain['label_counts'][str(c)]/ptrain['rows']:.1f}", f"{100*sum(c in r['_labels'] for r in unique_public)/len(unique_public):.1f}"] for c in IDS])
    report.paragraph(f"Максимальная группа одинаковых описаний в public train содержит {ptrain['largest_duplicate_group']} строк. Таблица показывает, как повторы меняют вес категорий; внутри public train совпавшие описания не имеют конфликтующих меток, поэтому здесь допустимо сравнение с одним экземпляром текста.")
    for i in (0,2,5): report.figure(output,*figures[i])
    report.heading("3. Длины текстов и ограничения входа")
    report.table(["Набор", "Символов p50", "p95", "max", "Слов p50", ">6000", ">12000"], [[short(n), s['characters']['median'], s['characters']['p95'], s['characters']['max'], s['words']['median'], s['over_6000_chars'], s['over_12000_chars']] for n,s in stats.items()])
    report.figure(output,*figures[1])
    report.paragraph("Все просмотренные описания короче 6 000 символов: текущий срез в avito_compare/scripts/train_baseline.py не обрезает эти данные. API принимает до 12 000 символов. Слово здесь — последовательность латинских или кириллических букв, а не токен языковой модели.")
    report.heading("4. Отрицания, ограничения и искажения")
    report.table(["Набор", "НЕ/НИ/НЕТ, % строк", "≥2 таких слов, %", "ТОЛЬКО/ЛИШЬ, %", "Смешанная кириллица/латиница, %"], [[short(n), *[f"{100*s[k]/s['rows']:.1f}" for k in ('standalone_negation_rows','two_or_more_negation_tokens_rows','only_marker_rows','mixed_script_word_rows')]] for n,s in stats.items()])
    report.figure(output,*figures[4])
    report.paragraph("Число НЕ/НИ/НЕТ — лексическая характеристика, не смысловая глубина. Слитное «неделаю» и ошибочное «ниправда» этим счётчиком не учитываются. Вложенность размечена авторами синтетических сценариев. Смешение алфавитов — индикатор возможной обфускации; оно также встречается в артикулах и названиях и не является доказательством ошибки. Автоматического надёжного подсчёта всех грамматических ошибок в реальных объявлениях здесь нет.")
    report.heading("5. Дубликаты, разбиение и происхождение")
    report.figure(output,*figures[3])
    report.table(["Файл A", "Файл B", "Общих уникальных текстов", "Строк A", "Строк B"], [[p['a'],p['b'],p['unique_texts'],p['rows_a'],p['rows_b']] for p in overlap['overlaps']])
    edits = data['distillate_dataset/hard_edits_500.jsonl']
    report.paragraph(f"Все {len(edits)} контролируемых правок hard_edits имеют source_item_id; они происходят от {len({r['source_item_id'] for r in edits})} исходных объявлений. Их необходимо группировать с исходниками при разделении даже при различающихся текстах. Семейства v2 разделены между train/validation/test, но используют общую лексику и сходные логические операции: отсутствие совпавших строк не доказывает полную семантическую независимость.")
    audit_path = ROOT/'hard_dataset/audit_report.json'
    audit = json.loads(audit_path.read_text())
    flagged = {(f['file'],f['line']) for f in audit['findings'] if f['severity']=='review'}
    report.paragraph(f"Актуальный аудит отмечает {len(flagged)} уникальных строк старых наборов для дополнительной проверки: medium-confidence, предположение о категории 101 и неоднозначные операции. Это кандидаты на проверку, не доказанные ошибки. Четыре подтверждённые ошибки ранее исправлены с сохранением старых меток в legacy_corrections.json. Исходный публичный тест не переразмечен автоматически.")
    report.heading("6. Рекомендации для обучения и оценки")
    for text in [
        "Использовать готовые train/validation/test v2 и не добавлять к ним v1. При присоединении реальных данных сначала убрать повторы и разделить исходники вместе с их производными по source_item_id. Одинаковые тексты с разными метками отправлять на проверку, не выбирать метку большинством автоматически.",
        "Публичный тест оставить вне обучения. Из-за пересечений и конфликтов дополнительно подготовить независимую проверенную выборку реальных объявлений; отдельно показывать метрики на конфликтных и неконфликтных подмножествах, не выдавая очищенную метрику за официальную.",
        "Долю синтетики подбирать на реальной валидации. Сравнить реальные данные, реальные плюс синтетика и реальные плюс синтетика с ошибками при одинаковой оценочной выборке. Не выбирать коэффициент смешивания по синтетическому test.",
        "Измерять micro-F1, F1 по каждой из шести категорий, точное совпадение набора, ложные положительные на пустых ответах, долю полностью правильных контрастных пар и изменение ответа после внесения ошибок. Отдельно показывать глубины отрицания и 101.",
        "Не включать служебные поля type, note, group_id, variant, augmentation, negation_depth и сами метки во вход модели. Настраивать пороги на validation; test и challenge использовать для финальной оценки. Одинаковый preprocessing и лимит длины применять при обучении и инференсе.",
    ]: report.paragraph(text)
    report.heading("7. Воспроизведение и границы анализа")
    report.paragraph("Запуск из корня: python3 datasets/eda.py. Зависимости: datasets/requirements-eda.txt. EDA.html автономен и содержит изображения внутри файла; EDA.md использует PNG рядом. metrics.json содержит все распределения, inventory.json — размеры и SHA-256, overlaps.json — пересечения и адреса конфликтующих строк. Данные не меняются при анализе.")
    report.paragraph("Нормализация для дубликатов: casefold и свёртка пробельных символов. Пунктуация, опечатки и отрицания сохраняются. Доли считаются по строкам каждого файла; версии и аугментации не считаются независимыми наблюдениями. Не выполнялись семантическое сравнение всех пар, ручная переразметка всего корпуса, оценка модели или причинный вывод об эффективности синтетики.")
    report.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"eda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, manifest = load()
    stats = {name: summarize(rows) for name,rows in data.items()}
    overlap = intersections(data)
    assert not overlap['v2_group_leaks'] and not overlap['v2_family_leaks']
    assert not any(p['a'].startswith(V2) and p['b'].startswith(V2) for p in overlap['overlaps'])
    for name, value in (("metrics.json",stats),("inventory.json",manifest),("overlaps.json",overlap)):
        (args.output_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    figures = plots(data,stats,args.output_dir)
    write_report(data,stats,overlap,manifest,figures,args.output_dir)
    print(json.dumps({"files":len(data),"rows":sum(len(r) for r in data.values()),"unique_texts":overlap['unique_texts_all_files'],"conflicting_text_groups":len(overlap['conflicting_text_groups']),"output":str(args.output_dir)},ensure_ascii=False))


if __name__ == "__main__":
    main()
