"""Build charts, markdown report, and executed notebook from ngram-expansion analysis.json."""
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbclient import NotebookClient
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/ngram-expansion-experiment"

SHORT_LABELS = {
    "ref_46k": "Ref 46k\n(1,3) cap 18k",
    "A_1_3_uncapped": "A (1,3)\nuncapped",
    "B_1_4_uncapped": "B (1,4)\nuncapped",
    "C_1_5_uncapped": "C (1,5)\nuncapped",
    "D_1_6_uncapped": "D (1,6)\nuncapped",
}

TABLE_NAMES = {
    "ref_46k": "Ref 46k (word 1–3 cap 18k + char_wb 28k)",
    "A_1_3_uncapped": "A: word (1,3) без лимита + char_wb 28k",
    "B_1_4_uncapped": "B: word (1,4) без лимита + char_wb 28k",
    "C_1_5_uncapped": "C: word (1,5) без лимита + char_wb 28k",
    "D_1_6_uncapped": "D: word (1,6) без лимита + char_wb 28k",
}


def make_charts(data: dict) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    configs = data["configs"]
    x = np.arange(len(configs))
    xlabels = [SHORT_LABELS[c] for c in configs]

    # Chart 1: Micro-F1 across slices + feature dimensionality growth
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    series = [
        ("Public validation (729)", [data["slice_evaluations"][c]["public_validation"]["micro_f1"] for c in configs], "#0f766e", "o-"),
        ("v2 realistic (124)", [data["slice_evaluations"][c]["v2_realism"]["realistic"]["micro_f1"] for c in configs], "#0284c7", "s--"),
        ("v2 artificial (1716)", [data["slice_evaluations"][c]["v2_realism"]["artificial"]["micro_f1"] for c in configs], "#d97706", "^-."),
        ("v2 test full (3016)", [data["slice_evaluations"][c]["v2_test"]["micro_f1"] for c in configs], "#b91c1c", "D-"),
        ("v2 plausible_hard (1176)", [data["slice_evaluations"][c]["v2_realism"]["plausible_hard"]["micro_f1"] for c in configs], "#7c3aed", "v:"),
        ("2 списка статусов (1152)", [data["slice_evaluations"][c]["v2_problem_lists"]["both_status_lists_1152"]["micro_f1"] for c in configs], "#475569", "x-"),
    ]
    for label, vals, color, fmt in series:
        axes[0].plot(x, vals, fmt, label=label, color=color, linewidth=2, markersize=6)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, fontsize=8.5)
    axes[0].set_ylim(-0.04, 1.02)
    axes[0].set_ylabel("Micro-F1 (порог 0.5)")
    axes[0].set_title("Качество по срезам при расширении n-грамм")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8, loc="center left", bbox_to_anchor=(0.01, 0.57))

    # Panel 2: Stacked dimensions by n-gram length + char_wb
    bottom = np.zeros(len(configs))
    stack_specs = [
        ("char_wb (3,5)", [data["measurements"][c]["dimensions"]["char"] for c in configs], "#64748b"),
        ("word 1-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("1", 0) for c in configs], "#0f766e"),
        ("word 2-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("2", 0) for c in configs], "#0284c7"),
        ("word 3-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("3", 0) for c in configs], "#d97706"),
        ("word 4-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("4", 0) for c in configs], "#ea580c"),
        ("word 5-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("5", 0) for c in configs], "#dc2626"),
        ("word 6-gram", [data["measurements"][c]["dimensions"]["word_by_ngram_length"].get("6", 0) for c in configs], "#7f1d1d"),
    ]
    for label, vals, color in stack_specs:
        arr = np.asarray(vals) / 1000.0
        axes[1].bar(x, arr, bottom=bottom, label=label, color=color, width=0.58)
        bottom += arr
    for xi, tot in zip(x, bottom):
        axes[1].text(xi, tot + 6, f"{tot:.1f}k", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, fontsize=8.5)
    axes[1].set_ylim(0, 385)
    axes[1].set_ylabel("Количество признаков (тыс.)")
    axes[1].set_title("Рост размерности признаков (словари TF-IDF)")
    axes[1].legend(fontsize=8, loc="upper left", ncol=2)

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"01_metrics_and_dimensions.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Chart 2: Collisions vs Status-list F1 & Overfitting (Train vs Test + Weight Share)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    coll_src = [data["v2_collisions"][c]["conflicting_src_vec_groups"] for c in configs]
    coll_vec = [data["v2_collisions"][c]["vector_collision_groups_any"] for c in configs]
    status_f1 = [data["slice_evaluations"][c]["v2_problem_lists"]["both_status_lists_1152"]["micro_f1"] for c in configs]

    w = 0.32
    b1 = axes[0].bar(x - w / 2, coll_src, width=w, label="Конфликтные группы (mcId + вектор)", color="#b91c1c")
    b2 = axes[0].bar(x + w / 2, coll_vec, width=w, label="Векторные коллизии (только текст)", color="#d97706")
    axes[0].bar_label(b1, padding=3, fontsize=8.5)
    axes[0].bar_label(b2, padding=3, fontsize=8.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(xlabels, fontsize=8.5)
    axes[0].set_ylim(0, 270)
    axes[0].set_ylabel("Число групп коллизий в v2_test")
    axes[0].set_title("Коллизии TF-IDF (слева) vs F1 на 2 списках (справа)")

    ax_r = axes[0].twinx()
    ax_r.spines["top"].set_visible(False)
    ax_r.plot(x, status_f1, "o-", color="#0f766e", linewidth=2.5, label="Micro-F1 на 1152 списках")
    ax_r.set_ylim(-0.05, 0.55)
    ax_r.set_ylabel("Micro-F1 на status_confirmation + status_negated_denial", color="#0f766e")
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    # Panel 2: Train vs Test gap (overfitting on v2 templates)
    v2_train_f1 = [data["slice_evaluations"][c]["train_diagnostic"]["v2_train"]["micro_f1"] for c in configs]
    v2_test_f1 = [data["slice_evaluations"][c]["v2_test"]["micro_f1"] for c in configs]
    pub_train_f1 = [data["slice_evaluations"][c]["train_diagnostic"]["public_train"]["micro_f1"] for c in configs]
    pub_val_f1 = [data["slice_evaluations"][c]["public_validation"]["micro_f1"] for c in configs]

    axes[1].plot(x, v2_train_f1, "o-", color="#0284c7", linewidth=2, label="v2 train (8464) — фит")
    axes[1].plot(x, pub_train_f1, "s--", color="#0f766e", linewidth=2, label="public train (3398) — фит")
    axes[1].plot(x, pub_val_f1, "s-", color="#10b981", linewidth=2, label="public validation (729)")
    axes[1].plot(x, v2_test_f1, "D-", color="#b91c1c", linewidth=2, label="v2 test (3016)")
    for xi, vt, vs in zip(x, v2_train_f1, v2_test_f1):
        axes[1].annotate(f"{vt:.4f}", (xi, vt), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5, color="#0284c7")
        axes[1].annotate(f"{vs:.4f}", (xi, vs), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=7.5, color="#b91c1c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(xlabels, fontsize=8.5)
    axes[1].set_ylim(0.38, 1.03)
    axes[1].set_ylabel("Micro-F1")
    axes[1].set_title("Разрыв train / test при росте длины n-грамм")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8, loc="center right")

    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"02_collisions_and_overfitting.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pp_delta(val: float, ref: float) -> str:
    d = (val - ref) * 100.0
    rel = ((val - ref) / ref * 100.0) if ref else 0.0
    return f"{d:+.2f} п.п. ({rel:+.2f}%)"


def build_report(data: dict) -> str:
    configs = data["configs"]
    lines = [
        "# Эксперимент по расширению словесных n-грамм TF-IDF: (1,3) → (1,6)",
        "",
        "## 1. Конфигурация эксперимента",
        "",
        "Все 5 моделей обучены с нуля на идентичных смешанных данных `public (3398) + v2 (8464) = 11 862` строк, "
        "при фиксированном `SGDClassifier(loss='log_loss', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)` в `OneVsRestClassifier`, "
        "порогах `0.5` для всех 5 голов (`102..106`) и исключении исходного `mcId` при оценке. "
        "Символьный TF-IDF во всех вариантах зафиксирован: `char_wb (3,5)`, `min_df=3`, `max_features=28 000`, `sublinear_tf=True`, `float32`. "
        "Словесный TF-IDF: `min_df=2`, `sublinear_tf=True`, `float32`; в `Ref 46k` сохранён лимит `max_features=18 000`, в вариантах `A–D` ограничение `max_features` снято (`None`).",
        "",
        "Воспроизведение `Ref 46k` побитно совпало с зафиксированной ранее моделью `output/mixed-charwb-comparison/runs/char_wb` "
        "(`public_validation` Micro-F1 = `0.930016`, `v2_test` Micro-F1 = `0.447005`).",
        "",
        "## 2. Анализ коллизий TF-IDF (на фиксированном `v2_test`, 3 016 уникальных текстов)",
        "",
        "Векторизаторы обучались строго на обучающей выборке (`11 862` строк) без подмешивания тестовых текстов.",
        "",
        "| Конфигурация | Уникальных векторов (из 3016) | Векторных коллизий (только текст): групп / строк | Конфликтных групп по вектору (разные labels): групп / строк | Коллизий `(mcId, вектор)`: групп / строк | Конфликтных `(mcId, вектор)` групп / строк | Мин. неизбежных ошибок меток | Из 231 конфл. групп разрешено |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in configs:
        vc = data["v2_collisions"][c]
        tr = data["baseline_231_tracking"][c]
        lines.append(
            f"| {TABLE_NAMES[c]} | {vc['unique_combined_vectors']} | "
            f"{vc['vector_collision_groups_any']} / {vc['vector_collision_rows_any']} | "
            f"{vc['conflicting_vec_groups_offered_labels']} / {vc['conflicting_vec_rows_offered_labels']} | "
            f"{vc['src_vector_collision_groups_any']} / {vc['src_vector_collision_rows_any']} | "
            f"{vc['conflicting_src_vec_groups']} / {vc['conflicting_src_vec_rows']} | "
            f"{vc['minimum_unavoidable_label_errors']} | "
            f"{tr['resolved_no_conflict']} / {tr['total_baseline_groups']} |"
        )

    lines += [
        "",
        "### Ключевые наблюдения по коллизиям",
        "",
        "1. **Все 231 конфликтные группы `(mcId, вектор)` (642 строки: 153 группы `status_confirmation` + 78 групп `status_negated_denial`) перестали быть коллизиями уже в варианте A `(1,3)` без `max_features`.** "
        "Причина исходных 231 коллизий была не в длине n-грамм `(1,3)`, а в лимите `word max_features=18 000`: при жёсткой отсечке словаря в него не попадали редкие граничные биграммы и триграммы на стыке услуги и ответа.",
        "2. **Дальнейшее расширение `(1,3) → (1,4) → (1,5) → (1,6)` не разделило ни одного дополнительного текста в `v2_test`:** число уникальных word-векторов (`2 946`) и объединённых векторов (`2 952`) остаётся абсолютно неизменным от варианта `A` до `D`.",
        "3. **48 векторных коллизий (112 строк) сохраняются даже при `(1,6)`:** все 48 групп — это тексты `status_confirmation` с аугментацией `spelling` (опечатки в названиях услуг: `мозайки`, `пакраску`, `канолизации`, `разеток`, `выключатилей`, `штукотурку`). "
        "У этих пар **побитно одинаковые TF-IDF-векторы** даже при `(1,6)` и **разные правильные метки `offered_microcategories`**, просто в тестовой выборке столкнувшиеся строки случайно имеют разный `mcId` (поэтому `(mcId, вектор)` равно 0). "
        "Они не исчезают при `(1,6)`, так как `status_confirmation` отсутствует в `train` (`0` строк), и сочетания опечаточных слов с `да`/`нет` (например, `мозайки да`, `под пакраску нет`) имеют `df = 0 < min_df = 2` в обучающей выборке.",
        "4. **Какие n-граммы разделили 231 группу и почему списки «верно/неверно» остались сломанными:**",
        "   - В `status_confirmation` («Выполняем ли ...? Да/Нет») сработали 2–3-граммы конца строки с ответом (`под ключ нет/да`, `автоматами узо да`, `стен плиткой да`, `штукатурными составами да/нет`, `по маякам да/нет`), случайно встречавшиеся в других шаблонах `train` от 2 до 7 раз.",
        "   - В `status_negated_denial` («Верно/Неверно, что мы не выполняем ...») длина фразы от оператора истинности до названия услуги составляет 7–11 слов (например, `неверно что мы не выполняет сборка электрощита с автоматами и узо` — 9 словотокенов). Даже 6-грамма от `неверно` не дотягивается до ключевого слова услуги (`электрощита`, `узо`). Разделение 78 групп произошло за счёт **межстрочных граничных n-грамм** на переносе строки (`раковин унитазов неверно`, `раковин унитазов верно`), которые связывают услугу из конца строки $k$ со статусом следующей строки $k+1$ — то есть семантически неверную пару.",
        "",
        "## 3. Метрики качества, размерность и ресурсы",
        "",
        "### Общая сводка по срезам (Micro-F1 / Macro-F1 / Precision / Recall / FP / FN / Пустые)",
        "",
    ]

    def add_slice_table(title: str, extractor):
        lines.append(f"#### {title}")
        lines.append("")
        lines.append("| Конфигурация | Строк | Micro-F1 | Δ vs 46k | Macro-F1 | Precision | Recall | FP | FN | Пустых (при gold>0) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        ref_m = extractor(data["slice_evaluations"]["ref_46k"])
        for c in configs:
            m = extractor(data["slice_evaluations"][c])
            delta_str = "—" if c == "ref_46k" else pp_delta(m["micro_f1"], ref_m["micro_f1"])
            lines.append(
                f"| {TABLE_NAMES[c]} | {m['rows']} | {m['micro_f1']:.6f} | {delta_str} | "
                f"{m['macro_f1']:.6f} | {m['micro_precision']:.6f} | {m['micro_recall']:.6f} | "
                f"{m['fp']} | {m['fn']} | {m['empty_predictions']} ({m['empty_on_positive_rows']}) |"
            )
        lines.append("")

    add_slice_table("Public validation (729)", lambda s: s["public_validation"])
    add_slice_table("Полный v2 test (3 016)", lambda s: s["v2_test"])
    add_slice_table("v2 realistic (124)", lambda s: s["v2_realism"]["realistic"])
    add_slice_table("v2 plausible_hard (1 176)", lambda s: s["v2_realism"]["plausible_hard"])
    add_slice_table("v2 artificial (1 716)", lambda s: s["v2_realism"]["artificial"])
    add_slice_table("v2: status_confirmation (576)", lambda s: s["v2_problem_lists"]["status_confirmation"])
    add_slice_table("v2: status_negated_denial (576)", lambda s: s["v2_problem_lists"]["status_negated_denial"])
    add_slice_table("v2 без двух статусных списков (1 864)", lambda s: s["v2_problem_lists"]["v2_without_status_lists_1864"])
    add_slice_table("v2 challenge (40)", lambda s: s["challenge"])

    lines += [
        "### Размерность словаря, время обучения, RAM и размер артефакта",
        "",
        "| Конфигурация | Word dims (1 / 2 / 3 / 4 / 5 / 6) | Char dims | Всего признаков | Рост vs 46k | Fit wall, с | Fit CPU, с | Peak RSS (self / mon), MiB | Размер joblib, МБ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in configs:
        m = data["measurements"][c]
        dims = m["dimensions"]
        bl = dims["word_by_ngram_length"]
        breakdown = " / ".join(str(bl.get(str(k), 0)) for k in range(1, 7))
        ratio = dims["total"] / 46000.0
        lines.append(
            f"| {TABLE_NAMES[c]} | {dims['word']} ({breakdown}) | {dims['char']} | {dims['total']} | {ratio:.2f}× | "
            f"{m['fit_wall_seconds']:.2f} | {m['fit_cpu_seconds']:.2f} | "
            f"{m['fit_peak_process_rss_mib']:.1f} / {m.get('monitored_peak_rss_mib', 0):.1f} | "
            f"{m['artifact_bytes'] / (1024**2):.2f} |"
        )

    lines += [
        "",
        "## 4. Детальный анализ результатов",
        "",
        f"- **Корреляция снижения коллизий и F1:** число конфликтных групп `(mcId, вектор)` упало со `231` до `0` (`-100%`), однако Micro-F1 на `v2_test` не вырос, а снизился с `0.447005` до `0.441523–0.446170` (корреляция Пирсона `r = {data['correlation_collisions_vs_v2_f1']['pearson_r_groups_vs_v2_f1']:+.3f}`: меньшее число коллизий не привело к росту F1). На `642` строках, входивших в `231` группу, и на всех `1 152` строках двух списков Micro-F1 равен **`0.000000`** во всех пяти конфигурациях.",
        f"- **FN на списках услуг:** число FN на `status_confirmation` (`1 224`) и `status_negated_denial` (`1 224`) не уменьшилось ни на единицу (`2 448` FN и `1 152 / 1 152` пустых предсказаний во всех вариантах). В `D (1,6)` максимальная вероятность среди всех `576` строк `status_confirmation` достигает лишь `{data['status_resolved_vs_pred']['status_confirmation']['max_prob_max_D_1_6']:.4f}` (медиана `{data['status_resolved_vs_pred']['status_confirmation']['max_prob_p50_D_1_6']:.4f}`), а среди `status_negated_denial` — `{data['status_resolved_vs_pred']['status_negated_denial']['max_prob_max_D_1_6']:.4f}` (медиана `{data['status_resolved_vs_pred']['status_negated_denial']['max_prob_p50_D_1_6']:.4f}`).",
        "- **Качество на `public_validation` (729):** слегка снижается с `0.930016` до `0.927163` (`-0.29` п.п. при `(1,6)`), но парный bootstrap (2000 выборок) показывает, что 95% ДИ включает 0 (`[-0.69, +0.08]` п.п.). Зато на `v2_test` снижение при `(1,4)`, `(1,5)` и `(1,6)` статистически значимо (`-0.46` п.п., ДИ `[-0.72, -0.20]`; `-0.55` п.п., ДИ `[-0.85, -0.28]`; `-0.38` п.п., ДИ `[-0.67, -0.12]`).",
        "- **Рост размерности:** общее число признаков выросло в **7.38×** (`46 000 → 339 533`), словесных — в **17.31×** (`18 000 → 311 533`), размер файла модели на диске — в **10.2×** (`2.77 МБ → 28.23 МБ`).",
        "- **Переобучение на шаблонах синтетики (подтверждено):** на обучающей части `v2_train` (8 464 строк) Micro-F1 резко растёт при добавлении 4–6-грамм: `0.976426 (Ref 46k) → 0.977038 (A) → 0.987248 (B) → 0.988752 (C) → 0.989132 (D)` (FN на `v2_train` падают почти в 4 раза: `232 → 63`). При этом на `v2_test` качество **падает** (`0.4470 → 0.4432`). Доля квадрата L2-нормы весов классификатора на длинных 4–6-граммах в `D (1,6)` достигает **34.55%** (`11.32% + 11.68% + 11.55%`), тогда как доля весов 1–3-грамм сжимается с `58.74%` до `21.13%`.",
        "",
        "### Примеры, где расширение n-грамм исправило ошибку (`Ref 46k` → `D (1,6)`)",
        "",
        f"На `v2_test` все {data['fixed_examples_count']} полностью исправленных примеров принадлежат одному типу `concord_never` (при этом 25 других строк `concord_never` и 15 строк других типов сломались). На `public_validation` 8 строк исправлены, 12 сломаны. Ниже приведены 14 конкретных примеров:",
        "",
    ]

    for idx, ex in enumerate(data["fixed_examples"], 1):
        snippet = ex["text"].replace("\n", " ")[:240]
        ng_parts = []
        for item in ex["new_ngrams_explanation"]:
            top_terms = ", ".join(f"`{t['ngram']}` ({t['len']}-г, w={t['weight']:+.3f})" for t in item["top_new_ngrams"][:3])
            ng_parts.append(f"mcId {item['mcId']}: {top_terms}")
        lines.append(
            f"{idx}. **[{ex['dataset']} / {ex['type']}] `{ex['item_id']}`** (src={ex['source_mcId']}): "
            f"gold={ex['gold']}, Ref46k={ex['pred_ref_46k']} → D(1,6)={ex['pred_D_1_6']}. "
            f"Текст: «{snippet}{'...' if len(ex['text']) > 240 else ''}». Новые признаки: {'; '.join(ng_parts)}."
        )

    lines += [
        "",
        "### Примеры, где расширение n-грамм не помогло или ухудшило результат",
        "",
    ]
    for idx, ex in enumerate(data["failed_examples"], 1):
        if ex["kind"] == "still_colliding_vector_in_D_1_6":
            lines.append(
                f"{idx}. **Сохранившаяся побитная векторная коллизия при `(1,6)` (`status_confirmation`, `spelling`)**: "
                f"`{ex['item_a']}` (src={ex['source_mcId_a']}, offered={ex['offered_a']}, gold={ex['gold_a']}) и "
                f"`{ex['item_b']}` (src={ex['source_mcId_b']}, offered={ex['offered_b']}, gold={ex['gold_b']}). "
                f"Оба дают одинаковые вероятности `{ex['prob_D_a']}` и пустой ответ `[]`."
            )
        else:
            snippet = ex["text"].replace("\n", " ")[:220]
            lines.append(
                f"{idx}. **[{ex['kind']} / {ex['type']}] `{ex['item_id']}`** (src={ex['source_mcId']}): "
                f"gold={ex['gold']}, Ref46k={ex['pred_ref_46k']} (`{ex['prob_ref_46k']}`) → "
                f"D(1,6)={ex['pred_D_1_6']} (`{ex['prob_D_1_6']}`). Текст: «{snippet}»."
            )

    lines += [
        "",
        "## 5. Итоговый вывод",
        "",
        "1. **Устраняет ли расширение n-грамм коллизии между объявлениями с разными правильными метками?** "
        "Частично, но не за счёт длины `4–6`: все 231 конфликтные группы `(mcId, вектор)` исчезают уже при `(1,3)` после снятия лимита `max_features=18 000`. "
        "Однако 48 групп побитно одинаковых текстовых векторов (`112` строк `status_confirmation` с опечатками) сохраняются даже при `(1,6)`, а в списках «верно/неверно» (`status_negated_denial`) даже 6-граммы не связывают оператор с услугой внутри одной строки (связь произошла лишь случайно через межстрочный перенос).",
        "2. **Улучшает ли это Micro-F1 на v2 и насколько?** "
        "Нет — качество не выросло, а незначительно снизилось: с `0.447005` (`Ref 46k`) до `0.446170` (`A`), `0.442408` (`B`), `0.441523` (`C`), `0.443223` (`D`, `-0.38` п.п.). На двух проблемных типах списков (`1 152` строки) Micro-F1 остался равным `0.000000`.",
        "3. **Как влияет на качество public validation?** "
        "Наблюдается небольшое снижение с `0.930016` до `0.927163–0.928513` (`-0.15…-0.29` п.п., в пределах 95% доверительного интервала bootstrap).",
        "4. **Оправдывает ли прирост качества увеличение количества признаков и вычислительных затрат?** "
        "Нет. Прироста качества нет ни на `v2_test`, ни на `public_validation`, тогда как размерность выросла в `7.38×` (`46k → 339.5k`), размер модели — в `10.2×`, а длинные n-граммы усилили запоминание шаблонов обучающей синтетики (`v2_train` `0.9764 → 0.9891` при падении на `v2_test`).",
        "",
        "**Главный вывод:** ограниченная длина n-грамм `(1,3)` **не является** первопричиной провала на синтетическом `v2` и на 6-позиционных списках. "
        "Даже без коллизий линейный bag-of-ngrams классификатор поверх всего текста документа не может выучить привязку отдельных «да/нет» и «верно/неверно» к конкретным услугам без сегментации/структурного представления и при отсутствии таких конструкций в `train`.",
    ]

    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_notebook() -> None:
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        md("# Эксперимент: расширение словесных n-грамм TF-IDF `(1,3) → (1,6)`\n\n"
           "Воспроизводимый разбор результатов эксперимента без необходимости в `.joblib`: "
           "метрики, коллизии TF-IDF, переобучение на шаблонах и примеры исправлений/ошибок."),
        code(
            "from pathlib import Path\n"
            "import json\n"
            "from IPython.display import display, Markdown, Image\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'avito_compare').is_dir() and (p / 'output').is_dir())\n"
            "OUT = ROOT / 'output/ngram-expansion-experiment'\n"
            "data = json.loads((OUT / 'analysis.json').read_text(encoding='utf-8'))\n"
            "configs = data['configs']\n"
            "def show_table(headers, rows):\n"
            "    md_txt = '| ' + ' | '.join(headers) + ' |\\n| ' + ' | '.join(['---']*len(headers)) + ' |\\n'\n"
            "    md_txt += '\\n'.join('| ' + ' | '.join(map(str, r)) + ' |' for r in rows)\n"
            "    display(Markdown(md_txt))\n"
            "rows = []\n"
            "for c in configs:\n"
            "    m = data['measurements'][c]\n"
            "    vc = data['v2_collisions'][c]\n"
            "    se = data['slice_evaluations'][c]\n"
            "    rows.append([\n"
            "        c, m['dimensions']['word'], m['dimensions']['total'],\n"
            "        f\"{vc['conflicting_src_vec_groups']} ({vc['vector_collision_groups_any']})\",\n"
            "        f\"{se['public_validation']['micro_f1']:.5f}\",\n"
            "        f\"{se['v2_test']['micro_f1']:.5f}\",\n"
            "        f\"{se['v2_problem_lists']['both_status_lists_1152']['micro_f1']:.5f}\",\n"
            "        f\"{se['train_diagnostic']['v2_train']['micro_f1']:.5f}\",\n"
            "        f\"{m['fit_wall_seconds']:.1f}s\", f\"{m['fit_peak_process_rss_mib']:.0f} MiB\"\n"
            "    ])\n"
            "show_table(['Конфиг', 'Word dims', 'Total dims', 'Коллизии mcId+vec (vec)', 'Public F1', 'v2_test F1', 'Списки 1152 F1', 'v2_train F1', 'Fit', 'RSS'], rows)"
        ),
        code(
            "display(Image(filename=str(OUT / '01_metrics_and_dimensions.png')))\n"
            "display(Image(filename=str(OUT / '02_collisions_and_overfitting.png')))"
        ),
        code(
            "print('Корреляция числа конфликтных групп и v2_test Micro-F1:', data['correlation_collisions_vs_v2_f1'])\n"
            "print('Доля L2^2 веса модели по длинам n-грамм:')\n"
            "for c in configs:\n"
            "    print(c, {k: round(v, 4) for k, v in data['weight_analysis'][c]['l2_sq_share'].items()})"
        ),
    ])
    with tempfile.TemporaryDirectory(prefix="ngram-exp-nb-") as tmp:
        spec = Path(tmp) / "kernels/ngram-exp"
        spec.mkdir(parents=True)
        (spec / "kernel.json").write_text(
            json.dumps({"argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "Ngram exp", "language": "python"}),
            encoding="utf-8",
        )
        os.environ.update({
            "JUPYTER_PATH": tmp,
            "IPYTHONDIR": str(Path(tmp) / "ipython"),
            "JUPYTER_RUNTIME_DIR": str(Path(tmp) / "runtime"),
            "MPLCONFIGDIR": str(Path(tmp) / "matplotlib"),
        })
        NotebookClient(nb, kernel_name="ngram-exp", timeout=120, resources={"metadata": {"path": str(ROOT)}}).execute()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, OUT / "ngram_expansion.ipynb")


def main() -> None:
    data = json.loads((OUT / "analysis.json").read_text(encoding="utf-8"))
    make_charts(data)
    build_report(data)
    build_notebook()
    print("Generated charts, report.md, and ngram_expansion.ipynb in", OUT)


if __name__ == "__main__":
    main()
