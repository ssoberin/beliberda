#!/usr/bin/env python3
"""
Read-only audit of datasets, gold annotation rules/conflicts, and frozen classifier
errors (`A_qwen_sgd`, `C_syntax`, `C_no_global`) across `pub_train`, `pub_validation`,
`v3_test`, and `official_208`. Excludes `public_synthetic_300`.
No models are trained, no thresholds tuned, no Colab/Qwen inference is invoked.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics
from avito_compare.scripts.syntax_features import CATEGORY_PATTERNS

OUT_DIR = ROOT / "output" / "dataset-audit"
SYN_DIR = ROOT / "output" / "syntax-aware-experiment"
PUB_SPLITS_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
OFFICIAL_PARQUET = ROOT / "datasets" / "public" / "public-test-dataset.parquet"

WORD_RE = re.compile(r"[а-яёa-z0-9]+", re.I)
TURNKEY_RE = re.compile(r"под\s+ключ|комплексн\w*\s+ремонт|полный\s+цикл", re.I)
SEPARATE_RE = re.compile(r"отдельн\w*|частичн\w*|мелки\w*\s+(?:работ\w*|задач\w*|ремонт\w*)|разов\w*\s+работ\w*", re.I)
NEG_TEXT_RE = re.compile(r"(?:^|\s)(?:не|нет|без|кроме|исключен\w*|отказыва\w*|прекрат\w*)\b", re.I)
THIRD_PARTY_TEXT_RE = re.compile(
    r"подрядчик\w*\s+(?:заказчик|клиент)|(?:клиент|заказчик)\w*\s+(?:сам|приглаш|нанима|поруча)|"
    r"(?:другие|сторонн\w*|профильн\w*)\s+(?:мастер\w*|бригад\w*|специалист\w*|подрядчик\w*)|"
    r"предыдущ\w*\s+мастер\w*|своими\s+силами\s+(?:клиент|заказчик)",
    re.I,
)
ONLY_TURNKEY_REFUSAL_RE = re.compile(
    r"(?:только\s+(?:\w+\s+){0,4}(?:на\s+(?:полный\s+)?)?(?:ремонт\s+)?под\s+ключ"
    r"|(?:отдельн\w*|частичн\w*|мелки\w*|разовы\w*)\s+(?:работ\w*|задач\w*|услуг\w*|заказ\w*|вид\w*\s+работ)\s+(?:[^.\n]{0,60}\s+)?(?:не\s+(?:бер[её]м|выполня\w*|дела\w*|оказыва\w*|беремся)|исключен\w*)"
    r"|по\s+(?:отдельн\w*|частичн\w*|мелки\w*|разовы\w*)\s+(?:работ\w*|задач\w*)\s*[-—–:]?\s*не\s+(?:звоните|обраща\w*|беспоко\w*|бер[её]м)"
    r"|не\s+(?:бер[её]м|выполня\w*|дела\w*|оказыва\w*|выезжа\w*)\s+(?:на\s+)?(?:отдельн\w*|частичн\w*|мелки\w*|разовы\w*))",
    re.I,
)

MCID_TO_NAME = {
    101: "Ремонт под ключ",
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}
NAME_TO_MCID = {v: k for k, v in MCID_TO_NAME.items() if k != 101}


def has_real_target_service_mention(text: str) -> tuple[bool, list[int]]:
    """
    Detect lexical mentions of categories 102..106 while guarding against known
    carpentry/noise false hits ('выравнивание дома/перекрытий', 'стяжка' without floor/plaster, etc.).
    Used strictly as an interpretable audit helper alongside raw syntax features.
    """
    cats_found = set()
    low = text.lower()
    for cid, pat in CATEGORY_PATTERNS.items():
        for m in WORD_RE.finditer(low):
            tok = m.group(0)
            if not pat.match(tok):
                continue
            # Polysemy guard for audit classification:
            # 'выравнивание дома/участка/сруба/балок/перекрытий' is structural carpentry, not plastering 102
            if cid == 102 and tok.startswith("выравниван"):
                ctx = low[m.start():min(len(low), m.end() + 45)]
                if re.search(r"дом\w*|сруб\w*|перекрыт\w*|фундамент\w*|венц\w*|строени\w*", ctx):
                    continue
            if cid == 102 and tok.startswith("ровн"):
                continue  # "ровные горизонты" in house leveling
            if cid == 104 and tok.startswith("отоплен"):
                pre = low[max(0, m.start() - 25):m.start()]
                if "расход" in pre or "экономи" in pre or "теплее" in pre:
                    continue  # "расходы на отопление стали ниже" in insulation ad
            if cid == 106 and tok.startswith("клеи"):
                continue
            cats_found.add(cid)
    return bool(cats_found), sorted(cats_found)


def fmt_probs_101_106(sub_rows: pd.DataFrame, col: str) -> dict[str, float | None]:
    out: dict[str, float | None] = {"101": None}
    for _, r in sub_rows.sort_values("cat_id").iterrows():
        out[str(int(r["cat_id"]))] = round(float(r[col]), 6)
    return out


def pred_mcids_from_probs(probs: dict[str, float | None], thr: float = 0.5) -> list[int]:
    return [int(k) for k in ("102", "103", "104", "105", "106") if probs[k] is not None and probs[k] >= thr]


def summarize_natasha_row(sub_rows: pd.DataFrame, traces_by_pair: dict[tuple[int, int], dict]) -> str:
    parts = []
    for _, r in sub_rows.sort_values("cat_id").iterrows():
        cid = int(r["cat_id"])
        pol = str(r["best_chunk_polarity"])
        subj = str(r["dominant_subject"])
        link = str(r["service_link_status"])
        ment_top3 = int(r["syn_service_mentioned_top3"])
        ment_ad = int(r["syn_service_mentioned_ad"])
        g_tk = int(r["syn_global_turnkey_only"])
        g_pr = int(r["syn_global_partial_refusal"])
        if r["p_C"] >= 0.5 or r["p_A"] >= 0.5 or ment_ad == 1 or pol != "unknown" or g_tk or g_pr:
            tr = traces_by_pair.get((int(r["row_idx"]), cid), {})
            clauses = tr.get("clauses", [])
            preds = []
            for cl in clauses[:2]:
                p = cl.get("pred") or "None"
                preds.append(f"{cl.get('mention')}->{p}[{cl.get('polarity')}/{cl.get('actor')}]")
            gtxt = ""
            if g_tk or g_pr:
                gtxt = f",global=(tk={g_tk},pr={g_pr})"
            cl_txt = f"({'; '.join(preds)})" if preds else ""
            parts.append(f"{cid}:pol={pol},subj={subj},link={link},ment_top3={ment_top3}{gtxt}{cl_txt}")
    return " | ".join(parts) if parts else "no_service_mentions_and_low_scores"


def load_clause_traces() -> dict[tuple[int, int], dict]:
    p = SYN_DIR / "public_tests_error_analysis.json"
    if not p.exists():
        return {}
    with open(p) as f:
        ea = json.load(f)
    out = {}
    exs = ea.get("examples", {}).get("official_208", {})
    for bucket_items in exs.values():
        for it in bucket_items:
            iid = str(it.get("item_id", ""))
            if iid.startswith("official_208_"):
                r_idx = int(iid.split("_")[-1])
                cid = int(it["cat_id"])
                out[(r_idx, cid)] = it
    return out


def classify_empty_gold_row(
    row_id: int,
    title: str,
    desc: str,
    other_golds: list[tuple[str, ...]],
    fpA: list[int],
    fpC: list[int],
    fpCng: list[int],
    sub_rows: pd.DataFrame,
) -> tuple[str, str, str, str]:
    """
    Classify each official_208 gold=[] row strictly into A..F and determine whether any FP
    is a model error vs annotation ambiguity/issue.
    """
    full = f"{title}\n{desc}"
    low = full.lower()
    has_any_fp = bool(fpA or fpC or fpCng)
    has_refusal = bool(ONLY_TURNKEY_REFUSAL_RE.search(full))
    has_tk = bool(TURNKEY_RE.search(full))
    has_svc, svc_cats = has_real_target_service_mention(full)
    has_conflict_dup = len(other_golds) > 0

    # Priority 1: Identical text in official_208 has non-empty gold elsewhere -> Group F
    if has_conflict_dup:
        if not has_any_fp:
            assess = "clean_tn_on_conflicting_dup"
        else:
            assess = "markup_conflict_identical_text_has_non_empty_gold"
        expl = (
            f"Идентичный description встречается в official_208 с непустым gold ({other_golds}). "
            f"Текст описывает услуги {svc_cats}, поэтому модели предсказывают A={fpA}, C={fpC}, C_ng={fpCng}."
        )
        return "F", "Неоднозначная ситуация / возможная ошибка разметки (конфликт дубликатов)", expl, assess

    # Priority 2: Explicit refusal of separate / partial / small works -> Group A
    if has_refusal:
        m = ONLY_TURNKEY_REFUSAL_RE.search(full)
        phrase = m.group(0) if m else ""
        if not has_any_fp:
            assess = "clean_tn"
            expl = f"Явный отказ от мелких/частичных работ («{phrase}»); все модели корректно вернули []."
        else:
            assess = "model_error_missed_explicit_turnkey_refusal"
            expl = (
                f"Явный отказ («{phrase}»), но модель не подавила перечисленные в тексте/SEO-хвосте категории "
                f"(fpA={fpA}, fpC={fpC}, fpCng={fpCng})."
            )
        return "A", "Явно написано, что дополнительные или отдельные работы не выполняются", expl, assess

    # Priority 3: Mentioned only in negation / comparison / design-prep / obstacle -> Group D
    # Concrete verified non-execution mention patterns in official_208 empty_gold:
    # - row 26: "не только про стены и плитку" + "Техпроект в подарок — проработаем планировку, электрику"
    # - row 165? check cleanly; let's detect non-execution context carefully
    if "не только про стены и плитку" in low or ("продумаем планировку" in low and "расстановку мебели" in low and not has_svc):
        assess = "model_error_non_offered_context" if has_any_fp else "clean_tn"
        expl = (
            "Упоминания в сравнении/дизайн-контексте («не только про стены и плитку», «проработаем планировку, электрику в техпроекте» без отдельного монтажа); "
            f"C/A ошибочно приняли за offered (fpA={fpA}, fpC={fpC})."
        )
        return "D", "Услуги упомянуты в отрицании, сравнении, цитате, вопросе или как чужие работы", expl, assess

    # Check team role vs work list (e.g. row 20: "электрики, сантехники, плиточники, штукатуры, маляры" inside team roster)
    is_team_roster_only = (
        "специалист" in low
        and re.search(r"электрик\w*,?\s+сантехник\w*,?\s+плиточник", low) is not None
        and "выполняем:\n" not in low
        and "берём на себя" not in low
    )
    if is_team_roster_only and has_tk:
        assess = "model_error_team_role_vs_service" if has_any_fp else "clean_tn"
        expl = (
            "Перечислены профессии в составе бригады («электрики, сантехники, плиточники, штукатуры, маляры») внутри «ремонт квартиры полностью под ключ», "
            f"без перечня отдельных услуг -> модели дали fpA={fpA}, fpC={fpC}."
        )
        return "B", "Услуги перечислены как часть ремонта под ключ, но отдельный заказ не оговорён", expl, assess

    # Row 96: photo-report mention ("Стены, стяжка, проводка, плитка — каждый слой работ у вас в телефоне") under turnkey
    if "каждый слой работ у вас в телефоне" in low:
        assess = "model_error_turnkey_photo_report_mention" if has_any_fp else "clean_tn"
        expl = (
            "Слова «стяжка, проводка, плитка» упомянуты в описании фотоотчёта при ремонте под ключ, а также присутствуют в SEO-хвосте; "
            f"модели предсказали fpA={fpA}, fpC={fpC}."
        )
        return "B", "Услуги перечислены как часть ремонта под ключ, но отдельный заказ не оговорён", expl, assess

    # Priority 4: No real 102-106 service mentioned (e.g. house lifting 'выравнивание дома', bathhouse paneling, garbage strings, pure turnkey boilerplate) -> Group E
    if not has_svc:
        if has_any_fp:
            if "выравниван" in low and re.search(r"дом\w*|перекрыт\w*|фундамент\w*", low):
                assess = "model_error_polysemy_or_domain_confusion"
                expl = (
                    f"Целевые услуги 102–106 отсутствуют: речь о подъёме/выравнивании деревянного дома, замене венцов или фундаменте; "
                    f"модели ошибочно сработали на словоформу «выравнивание»/«инженерных» (fpA={fpA}, fpC={fpC}, fpCng={fpCng})."
                )
            else:
                assess = "model_error_hallucination_without_service_mention"
                expl = f"Целевые услуги 102–106 не указаны, но модель дала ложное срабатывание (fpA={fpA}, fpC={fpC}, fpCng={fpCng})."
        else:
            assess = "clean_tn"
            expl = "Целевых услуг 102–106 в тексте нет (общий ремонт под ключ / деревянные конструкции / шум текста); модели корректно вернули []."
        return "E", "Вообще нет упоминания целевых услуг", expl, assess

    # Priority 5: Services ARE mentioned cleanly in text and NOT duplicate conflict, NOT explicit refusal
    if has_tk:
        # Services listed inside turnkey ad without explicit restriction
        assess = "ambiguous_turnkey_services_listed_gold_empty" if has_any_fp else "clean_tn_despite_service_mention"
        expl = (
            f"В объявлении «под ключ» явно названы работы по {svc_cats} (без запрета частичных работ), но gold=[]. "
            f"По ТЗ Авито (и в ~89–91% train/val) такие случаи обычно размечаются положительно; здесь — пограничная/спорная пустая разметка (fpA={fpA}, fpC={fpC})."
        )
        return "B", "Услуги перечислены как часть ремонта под ключ, но отдельный заказ не оговорён", expl, assess
    else:
        assess = "ambiguous_non_turnkey_services_gold_empty" if has_any_fp else "clean_tn"
        expl = f"Услуги {svc_cats} описаны без «под ключ» и без запрета, при этом gold=[]. Возможная ошибка разметки (fpA={fpA}, fpC={fpC})."
        return "C", "Услуги прямо описаны как выполняемые, без явного запрета на отдельный заказ", expl, assess


def compute_split_distributions_and_metrics() -> tuple[pd.DataFrame, dict]:
    pub_tr = cae.load_public_rows(PUB_SPLITS_DIR / "train.jsonl")
    pub_val = cae.load_public_rows(PUB_SPLITS_DIR / "validation.jsonl")
    v3_test = cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl")
    off_df = duckdb.connect().execute(f"SELECT title, description, categories FROM read_parquet('{OFFICIAL_PARQUET}')").fetchdf()
    off_rows = [
        {
            "row_idx": i,
            "title": str(r["title"] or ""),
            "description": str(r["description"] or ""),
            "mcId": 101,
            "evaluation_labels": [NAME_TO_MCID[c] for c in r["categories"] if c in NAME_TO_MCID],
            "labels": [NAME_TO_MCID[c] for c in r["categories"] if c in NAME_TO_MCID],
        }
        for i, r in off_df.iterrows()
    ]

    df_exp = pd.read_csv(SYN_DIR / "predictions.csv")
    df_sf = pd.read_csv(SYN_DIR / "syntax_features.csv")
    df_exp_full = df_exp.merge(df_sf, on=["split", "uid", "cat_id"], suffixes=("", "_sf"), how="left")
    df_pub = pd.read_csv(SYN_DIR / "public_tests_predictions.csv")
    df_off = df_pub[df_pub["split"] == "official_208"].copy()

    # Run duplicate & Situation 1/2 check helper
    splits_raw = {
        "pub_train": pub_tr,
        "pub_validation": pub_val,
        "v3_test": v3_test,
        "official_208": off_rows,
    }
    saved_pred_map = {
        "pub_validation": df_exp_full[df_exp_full["split"] == "pub_validation"],
        "v3_test": df_exp_full[df_exp_full["split"] == "v3_test"],
        "official_208": df_off,
    }

    dist_rows = []
    detailed = {}
    for name, rows in splits_raw.items():
        n_ads = len(rows)
        src_counts = dict(sorted(Counter(int(r.get("mcId", 101)) for r in rows).items()))
        empty_count = sum(1 for r in rows if len(r["evaluation_labels"]) == 0)
        cat_counts = {cid: sum(1 for r in rows if cid in r["evaluation_labels"]) for cid in cae.SCOPE_102_106}
        total_pos = sum(cat_counts.values())
        non_empty_ads = n_ads - empty_count

        # Textual markers at ad level
        tk_cnt = sep_cnt = neg_txt_cnt = tp_txt_cnt = 0
        sit1_tot = sit1_non = sit2_tot = sit2_emp = 0
        by_desc_src: dict[tuple[str, int], list[tuple[int, tuple[int, ...]]]] = {}
        by_desc: dict[str, list[tuple[int, int, tuple[int, ...], tuple[int, ...]]]] = {}

        for idx, r in enumerate(rows):
            desc = str(r.get("description", ""))
            src = int(r.get("mcId", 101))
            ev = tuple(sorted(r["evaluation_labels"]))
            full_ev = tuple(sorted(r.get("labels", ev)))
            by_desc_src.setdefault((desc, src), []).append((idx, ev))
            by_desc.setdefault(desc, []).append((idx, src, ev, full_ev))

            has_tk = bool(TURNKEY_RE.search(desc))
            has_sep = bool(SEPARATE_RE.search(desc))
            has_neg = bool(NEG_TEXT_RE.search(desc))
            has_tp = bool(THIRD_PARTY_TEXT_RE.search(desc))
            has_ref = bool(ONLY_TURNKEY_REFUSAL_RE.search(desc))
            has_svc, _ = has_real_target_service_mention(desc)

            if has_tk: tk_cnt += 1
            if has_sep: sep_cnt += 1
            if has_neg: neg_txt_cnt += 1
            if has_tp: tp_txt_cnt += 1

            if has_ref and has_tk:
                sit2_tot += 1
                if len(ev) == 0:
                    sit2_emp += 1
            elif has_tk and has_svc:
                sit1_tot += 1
                if len(ev) > 0:
                    sit1_non += 1

        conf_pair_rows = sum(len(v) for v in by_desc_src.values() if len({x[1] for x in v}) > 1)
        conf_pair_groups = sum(1 for v in by_desc_src.values() if len({x[1] for x in v}) > 1)

        # Saved predictions metrics per model & per category
        sp_df = saved_pred_map.get(name)
        model_stats = {}
        if sp_df is not None and len(sp_df):
            for mod, pcol in [("A_qwen_sgd", "p_A"), ("C_syntax", "p_C"), ("C_no_global", "p_C_no_global")]:
                y_true = sp_df["y_true"].values.astype(int)
                y_pred = (sp_df[pcol].values >= 0.5).astype(int)
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                tn = int(((y_true == 0) & (y_pred == 0)).sum())
                f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
                per_cat = {}
                for cid in cae.SCOPE_102_106:
                    sub_c = sp_df[sp_df["cat_id"] == cid]
                    yc = sub_c["y_true"].values.astype(int)
                    pc = (sub_c[pcol].values >= 0.5).astype(int)
                    per_cat[str(cid)] = {
                        "fp": int(((yc == 0) & (pc == 1)).sum()),
                        "fn": int(((yc == 1) & (pc == 0)).sum()),
                        "tp": int(((yc == 1) & (pc == 1)).sum()),
                    }
                model_stats[mod] = {"micro_f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "per_cat": per_cat}
            nat_neg_pairs = int((sp_df["syn_has_neg_on_action"] == 1).sum()) if "syn_has_neg_on_action" in sp_df else None
            nat_tp_pairs = int((sp_df["syn_third_party_actor"] == 1).sum()) if "syn_third_party_actor" in sp_df else None
            active_pairs = len(sp_df)
        else:
            nat_neg_pairs = None
            nat_tp_pairs = None
            active_pairs = sum(sum(1 for c in cae.SCOPE_102_106 if c != int(r.get("mcId", 101))) for r in rows)

        detailed[name] = {
            "n_ads": n_ads,
            "src_counts": src_counts,
            "empty_count": empty_count,
            "cat_counts": cat_counts,
            "total_pos": total_pos,
            "active_pairs": active_pairs,
            "sit1": {"total": sit1_tot, "non_empty": sit1_non, "empty": sit1_tot - sit1_non},
            "sit2": {"total": sit2_tot, "empty": sit2_emp, "non_empty": sit2_tot - sit2_emp},
            "conf_pair_groups": conf_pair_groups,
            "conf_pair_rows": conf_pair_rows,
            "model_stats": model_stats,
            "nat_neg_pairs": nat_neg_pairs,
            "nat_tp_pairs": nat_tp_pairs,
            "tk_cnt": tk_cnt,
            "sep_cnt": sep_cnt,
            "neg_txt_cnt": neg_txt_cnt,
            "tp_txt_cnt": tp_txt_cnt,
        }

        def fmt_cat_err(mod: str) -> str:
            if mod not in model_stats:
                return "no_saved_predictions"
            return "; ".join(f"{c}:FP={v['fp']}/FN={v['fn']}" for c, v in model_stats[mod]["per_cat"].items())

        dist_rows.append({
            "dataset": name,
            "n_ads": n_ads,
            "source_mcId_dist": json.dumps(src_counts),
            "empty_gold_count": empty_count,
            "empty_gold_pct": round(100.0 * empty_count / n_ads, 2),
            "pos_102": cat_counts[102],
            "pos_103": cat_counts[103],
            "pos_104": cat_counts[104],
            "pos_105": cat_counts[105],
            "pos_106": cat_counts[106],
            "total_pos_labels": total_pos,
            "avg_extra_cats_all": round(total_pos / n_ads, 4),
            "avg_extra_cats_non_empty": round(total_pos / non_empty_ads, 4) if non_empty_ads else 0.0,
            "turnkey_ads_pct": round(100.0 * tk_cnt / n_ads, 2),
            "separate_or_partial_ads_pct": round(100.0 * sep_cnt / n_ads, 2),
            "negation_marker_ads_pct": round(100.0 * neg_txt_cnt / n_ads, 2),
            "third_party_marker_ads_pct": round(100.0 * tp_txt_cnt / n_ads, 2),
            "natasha_neg_on_action_pairs": nat_neg_pairs if nat_neg_pairs is not None else "NA",
            "natasha_third_party_pairs": nat_tp_pairs if nat_tp_pairs is not None else "NA",
            "conflicting_dup_rows_same_src": conf_pair_rows,
            "A_micro_f1": model_stats["A_qwen_sgd"]["micro_f1"] if "A_qwen_sgd" in model_stats else "NA",
            "C_micro_f1": model_stats["C_syntax"]["micro_f1"] if "C_syntax" in model_stats else "NA",
            "C_no_global_micro_f1": model_stats["C_no_global"]["micro_f1"] if "C_no_global" in model_stats else "NA",
            "A_fp_fn_by_cat": fmt_cat_err("A_qwen_sgd"),
            "C_fp_fn_by_cat": fmt_cat_err("C_syntax"),
            "C_no_global_fp_fn_by_cat": fmt_cat_err("C_no_global"),
        })

    return pd.DataFrame(dist_rows), detailed


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    off_df = duckdb.connect().execute(f"SELECT title, description, categories FROM read_parquet('{OFFICIAL_PARQUET}')").fetchdf()
    df_pub = pd.read_csv(SYN_DIR / "public_tests_predictions.csv")
    df_off = df_pub[df_pub["split"] == "official_208"].copy()
    traces_by_pair = load_clause_traces()

    # Group official_208 by description
    desc_groups_map: dict[str, list[int]] = {}
    for idx, row in off_df.iterrows():
        desc_groups_map.setdefault(str(row["description"] or ""), []).append(idx)

    dup_group_id_by_row: dict[int, str] = {}
    dup_groups_json = []
    n_diff_src_normal = n_same_src_conflict_groups = n_same_src_conflict_rows = 0
    n_same_src_same_gold_groups = n_same_src_same_gold_rows = 0

    g_counter = 0
    for desc, r_idxs in desc_groups_map.items():
        if len(r_idxs) < 2:
            continue
        g_counter += 1
        gid = f"dup_{g_counter:02d}"
        for ri in r_idxs:
            dup_group_id_by_row[ri] = gid

        # source_mcId is always 101 in official_208 contract (not in parquet)
        src_set = {101}
        gold_tuples = [tuple(sorted(str(x) for x in off_df.iloc[ri]["categories"])) for ri in r_idxs]
        distinct_golds = sorted({gt for gt in gold_tuples})
        diff_src = len(src_set) > 1
        conflict_with_src = len({(101, gt) for gt in gold_tuples}) > 1

        if diff_src:
            dup_type = "same_text_diff_source_normal"
            n_diff_src_normal += 1
        elif conflict_with_src:
            dup_type = "same_text_same_source_diff_gold_conflict"
            n_same_src_conflict_groups += 1
            n_same_src_conflict_rows += len(r_idxs)
        else:
            dup_type = "same_text_same_source_same_gold"
            n_same_src_same_gold_groups += 1
            n_same_src_same_gold_rows += len(r_idxs)

        rows_payload = []
        for ri in r_idxs:
            sub = df_off[df_off["row_idx"] == ri]
            pA = fmt_probs_101_106(sub, "p_A")
            pC = fmt_probs_101_106(sub, "p_C")
            pCng = fmt_probs_101_106(sub, "p_C_no_global")
            gnames = list(off_df.iloc[ri]["categories"])
            rows_payload.append({
                "row_id": ri,
                "title": str(off_df.iloc[ri]["title"] or ""),
                "source_mcId": 101,
                "gold_mcIds": [NAME_TO_MCID[g] for g in gnames if g in NAME_TO_MCID],
                "gold_categories": gnames,
                "pred_A": pred_mcids_from_probs(pA),
                "pred_C": pred_mcids_from_probs(pC),
                "pred_C_no_global": pred_mcids_from_probs(pCng),
                "probs_A": pA,
                "probs_C": pC,
                "probs_C_no_global": pCng,
            })

        dup_groups_json.append({
            "group_id": gid,
            "duplicate_type": dup_type,
            "n_rows": len(r_idxs),
            "row_ids": r_idxs,
            "differs_source_mcId": diff_src,
            "conflict_persists_with_source_mcId": conflict_with_src,
            "distinct_titles": sorted({str(off_df.iloc[ri]["title"] or "") for ri in r_idxs}),
            "distinct_gold_sets": [list(gt) for gt in distinct_golds],
            "full_description": desc,
            "rows": rows_payload,
        })

    dup_output = {
        "dataset": "official_208",
        "source_mcId_column_present_in_parquet": False,
        "source_mcId_contract_note": (
            "datasets/public/public-test-dataset.parquet содержит только поля title, description, categories. "
            "Согласно datasets/public/README.md и README-files.txt, исходная микрокатегория для всех 208 записей равна 101 («Ремонт под ключ»)."
        ),
        "summary": {
            "total_rows": len(off_df),
            "total_duplicate_groups": len(dup_groups_json),
            "total_rows_in_duplicate_groups": sum(g["n_rows"] for g in dup_groups_json),
            "same_text_diff_source_mcId_normal_groups": n_diff_src_normal,
            "same_text_diff_source_mcId_normal_rows": 0,
            "same_text_same_source_mcId_diff_gold_conflict_groups": n_same_src_conflict_groups,
            "same_text_same_source_mcId_diff_gold_conflict_rows": n_same_src_conflict_rows,
            "same_text_same_source_mcId_same_gold_groups": n_same_src_same_gold_groups,
            "same_text_same_source_mcId_same_gold_rows": n_same_src_same_gold_rows,
        },
        "groups": dup_groups_json,
    }
    with open(OUT_DIR / "duplicate_groups.json", "w") as f:
        json.dump(dup_output, f, ensure_ascii=False, indent=2)

    # 2 & 6: official_208_full.jsonl + empty_gold_analysis.csv
    full_jsonl_path = OUT_DIR / "official_208_full.jsonl"
    empty_rows_csv = []

    with open(full_jsonl_path, "w") as fj:
        for ri, row in off_df.iterrows():
            title = str(row["title"] or "")
            desc = str(row["description"] or "")
            gnames = list(row["categories"])
            gmcids = [NAME_TO_MCID[g] for g in gnames if g in NAME_TO_MCID]

            sub = df_off[df_off["row_idx"] == ri].sort_values("cat_id")
            pA = fmt_probs_101_106(sub, "p_A")
            pC = fmt_probs_101_106(sub, "p_C")
            pCng = fmt_probs_101_106(sub, "p_C_no_global")
            pTF = fmt_probs_101_106(sub, "p_tfidf")

            prA = pred_mcids_from_probs(pA)
            prC = pred_mcids_from_probs(pC)
            prCng = pred_mcids_from_probs(pCng)
            prTF = pred_mcids_from_probs(pTF)

            same_desc_rows = desc_groups_map[desc]
            other_golds = sorted({
                tuple(sorted(str(x) for x in off_df.iloc[rj]["categories"]))
                for rj in same_desc_rows
                if rj != ri and tuple(sorted(str(x) for x in off_df.iloc[rj]["categories"])) != tuple(sorted(gnames))
            })
            gid = dup_group_id_by_row.get(ri)
            is_conflict = len(other_golds) > 0

            nat_sum = summarize_natasha_row(sub, traces_by_pair)

            rec = {
                "row_id": ri,
                "item_id": f"official_208_{ri}",
                "source_mcId": 101,
                "source_mcId_in_parquet": False,
                "title": title,
                "description": desc,
                "gold_mcIds": gmcids,
                "gold_categories": gnames,
                "predictions": {
                    "A_qwen_sgd": prA,
                    "C_syntax": prC,
                    "C_no_global": prCng,
                    "tfidf_sgd": prTF,
                },
                "probabilities": {
                    "note_101": "source_mcId=101; target classes in frozen models are 102..106",
                    "A_qwen_sgd": pA,
                    "C_syntax": pC,
                    "C_no_global": pCng,
                    "tfidf_sgd": pTF,
                },
                "duplicate_group_id": gid,
                "conflicting_duplicate": is_conflict,
                "other_golds_same_text": [list(x) for x in other_golds],
                "natasha_features_summary": nat_sum,
            }
            fj.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if len(gmcids) == 0:
                gcode, glabel, expl, assess = classify_empty_gold_row(
                    ri, title, desc, other_golds, prA, prC, prCng, sub
                )
                empty_rows_csv.append({
                    "row_id": ri,
                    "source_mcId": 101,
                    "title": title,
                    "description": desc,
                    "gold": "[]",
                    "pred_A": json.dumps(prA),
                    "pred_C": json.dumps(prC),
                    "pred_C_no_global": json.dumps(prCng),
                    "pA_101": "",
                    "pA_102": pA["102"], "pA_103": pA["103"], "pA_104": pA["104"], "pA_105": pA["105"], "pA_106": pA["106"],
                    "pC_101": "",
                    "pC_102": pC["102"], "pC_103": pC["103"], "pC_104": pC["104"], "pC_105": pC["105"], "pC_106": pC["106"],
                    "pCng_101": "",
                    "pCng_102": pCng["102"], "pCng_103": pCng["103"], "pCng_104": pCng["104"], "pCng_105": pCng["105"], "pCng_106": pCng["106"],
                    "fp_cats_A": json.dumps(prA),
                    "fp_cats_C": json.dumps(prC),
                    "fp_cats_C_no_global": json.dumps(prCng),
                    "any_fp": bool(prA or prC or prCng),
                    "group_code": gcode,
                    "group_label": glabel,
                    "conflicting_duplicate_non_empty": is_conflict,
                    "other_golds_same_text": json.dumps([list(x) for x in other_golds], ensure_ascii=False),
                    "natasha_features": nat_sum,
                    "explanation": expl,
                    "assessment_model_vs_markup": assess,
                })

    df_empty = pd.DataFrame(empty_rows_csv)
    df_empty.to_csv(OUT_DIR / "empty_gold_analysis.csv", index=False)

    # 4: Dataset distributions across pub_train, pub_validation, v3_test, official_208
    df_dist, dist_detail = compute_split_distributions_and_metrics()
    df_dist.to_csv(OUT_DIR / "dataset_distributions.csv", index=False)

    # 5: Detailed error breakdown on official_208 + 30 FP & 15+ FN examples
    # Build rich error metrics for A, C, C_no_global on official_208
    df_off["is_empty_gold"] = df_off["row_idx"].map(lambda r: len(off_df.iloc[int(r)]["categories"]) == 0)
    df_off["has_turnkey_text"] = df_off["row_idx"].map(lambda r: bool(TURNKEY_RE.search(str(off_df.iloc[int(r)]["description"] or ""))))
    df_off["is_conflict_dup"] = df_off["row_idx"].map(lambda r: any(g["conflict_persists_with_source_mcId"] and int(r) in g["row_ids"] for g in dup_groups_json))

    # Map empty_gold group codes per row_idx
    empty_group_map = dict(zip(df_empty["row_id"], df_empty["group_code"]))
    empty_assess_map = dict(zip(df_empty["row_id"], df_empty["assessment_model_vs_markup"]))
    df_off["empty_group"] = df_off["row_idx"].map(lambda r: empty_group_map.get(int(r), "NON_EMPTY"))
    df_off["empty_assess"] = df_off["row_idx"].map(lambda r: empty_assess_map.get(int(r), "NON_EMPTY"))

    y = df_off["y_true"].values.astype(int)
    pA_bin = (df_off["p_A"].values >= 0.5).astype(int)
    pC_bin = (df_off["p_C"].values >= 0.5).astype(int)
    pCng_bin = (df_off["p_C_no_global"].values >= 0.5).astype(int)

    # Count FP breakdown for C_syntax (111 FP) and A (120 FP) and C_no_global (109 FP)
    def fp_attribution(pbin: np.ndarray) -> dict[str, int]:
        fp_mask = (pbin == 1) & (y == 0)
        sub = df_off[fp_mask]
        in_turnkey_ads = int(sub["has_turnkey_text"].sum())
        in_empty_total = int(sub["is_empty_gold"].sum())
        in_non_empty_total = int((~sub["is_empty_gold"]).sum())
        by_emp_grp = Counter(sub[sub["is_empty_gold"]]["empty_group"])
        in_conflict_dup = int(sub["is_conflict_dup"].sum())
        # Model error vs markup/ambiguous attribution:
        # Clean model errors: empty groups E, A, D, plus non-conflict team/photo mentions (rows 20, 96),
        # plus non-empty FP where identical text doesn't conflict AND polysemy/no real offer? Look specifically at empty groups + conflict check:
        model_err_empty = int(sub[sub["empty_assess"].str.startswith("model_error")].shape[0])
        markup_or_ambig_empty = int(sub[sub["empty_assess"].isin(["markup_conflict_identical_text_has_non_empty_gold", "ambiguous_turnkey_services_listed_gold_empty", "ambiguous_non_turnkey_services_gold_empty"])].shape[0])
        return {
            "total_fp": int(fp_mask.sum()),
            "in_turnkey_ads": in_turnkey_ads,
            "in_empty_gold": in_empty_total,
            "in_non_empty_gold": in_non_empty_total,
            "in_conflicting_dup_rows": in_conflict_dup,
            "empty_by_group": dict(sorted(by_emp_grp.items())),
            "empty_model_errors": model_err_empty,
            "empty_markup_or_ambiguous": markup_or_ambig_empty,
        }

    attr_A = fp_attribution(pA_bin)
    attr_C = fp_attribution(pC_bin)
    attr_Cng = fp_attribution(pCng_bin)

    # Also check non_empty_gold FP attribution (are any of those 29 non_empty_gold FPs inside conflicting duplicates or polysemy 'выравнивание дома' / 'тихие стены'?)
    non_empty_fp_C = df_off[(pC_bin == 1) & (y == 0) & (~df_off["is_empty_gold"])].copy()
    non_empty_fp_conflict = int(non_empty_fp_C["is_conflict_dup"].sum())

    # Select 30 diverse representative FP (row_idx, cat_id) across A/C/Cng covering F, A, B, D, E, non_empty
    # Group by (row_id) or take distinct ads/categories prioritizing diverse categories & groups
    fp_union = df_off[(y == 0) & ((pA_bin == 1) | (pC_bin == 1) | (pCng_bin == 1))].copy()
    fp_union["priority_bucket"] = fp_union["empty_group"].map({
        "A": 1, "D": 2, "E": 3, "F": 4, "B": 5, "C": 6, "NON_EMPTY": 7
    })
    # Pick representative rows covering all buckets (cap max 2 categories per row so we show ~25-30 distinct ads/pairs)
    chosen_fp = []
    seen_fp_rows = Counter()
    seen_desc_sig = Counter()
    for grp_code in ["A", "D", "E", "F", "B", "NON_EMPTY"]:
        cand = fp_union[fp_union["empty_group"] == grp_code].sort_values(["p_C", "p_A"], ascending=False)
        for _, rr in cand.iterrows():
            r_idx = int(rr["row_idx"])
            sig = str(off_df.iloc[r_idx]["description"] or "")[:70]
            if seen_fp_rows[r_idx] >= 2 or seen_desc_sig[sig] >= 2:
                continue
            chosen_fp.append(rr)
            seen_fp_rows[r_idx] += 1
            seen_desc_sig[sig] += 1
            if len([x for x in chosen_fp if x["empty_group"] == grp_code]) >= (7 if grp_code in ("B", "NON_EMPTY", "F", "E") else 4):
                break
    # Top up to 30 if < 30
    chosen_keys = {(int(x["row_idx"]), int(x["cat_id"])) for x in chosen_fp}
    for _, rr in fp_union.sort_values(["p_C", "p_A"], ascending=False).iterrows():
        if len(chosen_fp) >= 30:
            break
        k = (int(rr["row_idx"]), int(rr["cat_id"]))
        r_idx = k[0]
        if k not in chosen_keys and seen_fp_rows[r_idx] < 2:
            chosen_fp.append(rr)
            chosen_keys.add(k)
            seen_fp_rows[r_idx] += 1
    chosen_fp = chosen_fp[:30]

    # Select 15+ FN (there are 21 total in union of A, C, Cng — let's include top 15 covering C_syntax FNs, A FNs fixed by C, and C_no_global FNs)
    fn_union = df_off[(y == 1) & ((pA_bin == 0) | (pC_bin == 0) | (pCng_bin == 0))].copy()
    fn_union["is_fn_C"] = (fn_union["p_C"] < 0.5).astype(int)
    fn_union = fn_union.sort_values(["is_fn_C", "syn_global_partial_refusal", "p_C"], ascending=[False, False, True])
    chosen_fn = [rr for _, rr in fn_union.head(15).iterrows()]

    # Build error_examples.md
    err_lines = [
        "# Детальный разбор ошибок `official_208` (`A_qwen_sgd`, `C_syntax`, `C_no_global`)",
        "",
        "## 1. Сводные метрики на `official_208` (`208` объявлений, `1 040` пар, `source_mcId = 101`, `thr = 0.5`)",
        "",
        "| Модель | Micro-F1 | TP | FP | FN | TN | По `gold=[]` (`106` об.) FP | По `gold!=[]` (`102` об.) FP / FN |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mod, pbin in [("A_qwen_sgd", pA_bin), ("C_syntax", pC_bin), ("C_no_global", pCng_bin)]:
        tp = int(((y == 1) & (pbin == 1)).sum())
        fp = int(((y == 0) & (pbin == 1)).sum())
        fn = int(((y == 1) & (pbin == 0)).sum())
        tn = int(((y == 0) & (pbin == 0)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn)
        fp_emp = int(((y == 0) & (pbin == 1) & df_off["is_empty_gold"]).sum())
        fp_non = int(((y == 0) & (pbin == 1) & (~df_off["is_empty_gold"])).sum())
        err_lines.append(f"| `{mod}` | `{f1:.4f}` | {tp} | {fp} | {fn} | {tn} | {fp_emp} | {fp_non} / {fn} |")

    err_lines += [
        "",
        "### FP и FN по категориям (`102–106`) и `source_mcId = 101`",
        "",
        "| Категория | `A` FP / FN | `C_syntax` FP / FN | `C_no_global` FP / FN |",
        "|---|---:|---:|---:|",
    ]
    for cid in cae.SCOPE_102_106:
        sub_c = df_off[df_off["cat_id"] == cid]
        yc = sub_c["y_true"].values
        row_parts = []
        for col in ("p_A", "p_C", "p_C_no_global"):
            pc = (sub_c[col].values >= 0.5).astype(int)
            row_parts.append(f"{int(((yc == 0) & (pc == 1)).sum())} / {int(((yc == 1) & (pc == 0)).sum())}")
        err_lines.append(f"| `{cid}` ({MCID_TO_NAME[cid]}) | {row_parts[0]} | {row_parts[1]} | {row_parts[2]} |")

    common_fp = int(((y == 0) & (pA_bin == 1) & (pC_bin == 1) & (pCng_bin == 1)).sum())
    common_fn = int(((y == 1) & (pA_bin == 0) & (pC_bin == 0) & (pCng_bin == 0)).sum())
    fixed_AC = df_off[(pA_bin != y) & (pC_bin == y)]
    broken_AC = df_off[(pA_bin == y) & (pC_bin != y)]
    fixed_ACng = df_off[(pA_bin != y) & (pCng_bin == y)]
    broken_ACng = df_off[(pA_bin == y) & (pCng_bin != y)]

    err_lines += [
        "",
        "### Пересечения и переходы между моделями",
        f"- **Общие ошибки всех трёх моделей (`A`, `C_syntax`, `C_no_global`):** `{common_fp}` FP и `{common_fn}` FN.",
        f"- **Синтаксис (`C_syntax`) исправил ошибки Qwen (`A -> C_syntax`):** `{len(fixed_AC)}` (`{int((fixed_AC['y_true']==0).sum())}` FP->TN, `{int((fixed_AC['y_true']==1).sum())}` FN->TP).",
        f"- **Синтаксис (`C_syntax`) сломал верные ответы Qwen (`A -> C_syntax`):** `{len(broken_AC)}` (`{int((broken_AC['y_true']==0).sum())}` TN->FP, `{int((broken_AC['y_true']==1).sum())}` TP->FN).",
        f"- **Абляция `C_no_global` vs Qwen (`A -> C_no_global`):** исправлено `{len(fixed_ACng)}` (`{int((fixed_ACng['y_true']==0).sum())}` FP->TN, `{int((fixed_ACng['y_true']==1).sum())}` FN->TP), сломано `{len(broken_ACng)}` (`{int((broken_ACng['y_true']==0).sum())}` TN->FP, `{int((broken_ACng['y_true']==1).sum())}` TP->FN).",
        f"- **Связь FP с «под ключ» / комплексным ремонтом (`C_syntax`):** `{attr_C['in_turnkey_ads']}` из `{attr_C['total_fp']}` FP (`{100.0*attr_C['in_turnkey_ads']/attr_C['total_fp']:.1f}%`) находятся в объявлениях с формулировками «под ключ / комплексный ремонт / полный цикл» (у `A`: `{attr_A['in_turnkey_ads']}/{attr_A['total_fp']}`; у `C_no_global`: `{attr_Cng['in_turnkey_ads']}/{attr_Cng['total_fp']}`).",
        "",
        "### Примечательные переходы `A -> C_syntax` (кратко перед полными примерами)",
        "- **Исправлено синтаксисом (`FN->TP` и `FP->TN`):** например `row 59` («Облицовка крыльца плиткой», `103: p_A=0.502 -> p_C=0.976`), `row 68` (`106: p_A=0.457 -> p_C=0.762`), `row 88/124/152/158` (пропущенная `105: Электрика` `p_A=0.108..0.122 -> p_C=0.755..0.823`), `row 2/5/48` (ложная `104` в ремонте венцов `FP->TN`).",
        "- **Сломано синтаксисом (`TP->FN` из-за глобальных флагов):** `row 11` (`102: 0.997 -> 0.281` из-за «не придётся искать отдельных подрядчиков»), `row 18` («частичные работы не выполняем» при явном блоке «Выполняем: штукатурку, плитку, сантехнику» -> `102/103/104` с `0.908..0.996` в `0.137..0.332`), `row 113/115` (`105: 0.954 -> 0.331` — в `C_no_global` сохранено `0.944`).",
        "",
        "---",
        "",
        "## 2. 30 показательных FP на `official_208` (с полными текстами)",
        "",
    ]

    for i, rr in enumerate(chosen_fp, 1):
        ri = int(rr["row_idx"])
        cid = int(rr["cat_id"])
        row_raw = off_df.iloc[ri]
        sub = df_off[df_off["row_idx"] == ri].sort_values("cat_id")
        pA = fmt_probs_101_106(sub, "p_A")
        pC = fmt_probs_101_106(sub, "p_C")
        pCng = fmt_probs_101_106(sub, "p_C_no_global")
        eg = empty_group_map.get(ri, "NON_EMPTY")
        ea_expl = df_empty[df_empty["row_id"] == ri]["explanation"].iloc[0] if ri in empty_group_map else (
            f"Объявление непустое (gold={list(row_raw['categories'])}), но категория {cid} отсутствует в gold; ссылка: {rr['service_link_status']}, conflict_dup={bool(rr['is_conflict_dup'])}."
        )
        err_lines += [
            f"### FP #{i}: `row_id={ri}`, категория `{cid}` ({MCID_TO_NAME[cid]}), группа `{eg}`",
            f"- `source_mcId`: `101` | `gold`: `{list(row_raw['categories'])}` | `is_conflict_dup`: `{bool(rr['is_conflict_dup'])}`",
            f"- `pred`: `A={pred_mcids_from_probs(pA)}`, `C_syntax={pred_mcids_from_probs(pC)}`, `C_no_global={pred_mcids_from_probs(pCng)}`",
            f"- `probs[{cid}]`: `p_A={float(rr['p_A']):.4f}`, `p_C={float(rr['p_C']):.4f}`, `p_C_no_global={float(rr['p_C_no_global']):.4f}` | `pol={rr['best_chunk_polarity']}`, `link={rr['service_link_status']}`, `g_pr={int(rr['syn_global_partial_refusal'])}`",
            f"- **Причина:** {ea_expl}",
            f"- **Заголовок:** `{row_raw['title']}`",
            "",
            "```text",
            str(row_raw["description"]),
            "```",
            "",
        ]

    err_lines += [
        "---",
        "",
        "## 3. 15 показательных FN на `official_208` (с полными текстами)",
        "",
    ]
    for i, rr in enumerate(chosen_fn, 1):
        ri = int(rr["row_idx"])
        cid = int(rr["cat_id"])
        row_raw = off_df.iloc[ri]
        sub = df_off[df_off["row_idx"] == ri].sort_values("cat_id")
        pA = fmt_probs_101_106(sub, "p_A")
        pC = fmt_probs_101_106(sub, "p_C")
        pCng = fmt_probs_101_106(sub, "p_C_no_global")
        err_lines += [
            f"### FN #{i}: `row_id={ri}`, пропущена категория `{cid}` ({MCID_TO_NAME[cid]})",
            f"- `source_mcId`: `101` | `gold`: `{list(row_raw['categories'])}`",
            f"- `pred`: `A={pred_mcids_from_probs(pA)}`, `C_syntax={pred_mcids_from_probs(pC)}`, `C_no_global={pred_mcids_from_probs(pCng)}`",
            f"- `probs[{cid}]`: `p_A={float(rr['p_A']):.4f}`, `p_C={float(rr['p_C']):.4f}`, `p_C_no_global={float(rr['p_C_no_global']):.4f}` | `pol={rr['best_chunk_polarity']}`, `link={rr['service_link_status']}`, `g_pr={int(rr['syn_global_partial_refusal'])}`",
            f"- **Заголовок:** `{row_raw['title']}`",
            "",
            "```text",
            str(row_raw["description"]),
            "```",
            "",
        ]

    with open(OUT_DIR / "error_examples.md", "w") as f:
        f.write("\n".join(err_lines))

    # Build concise, data-driven report.md
    grp_counts = Counter(df_empty["group_code"])
    grp_fp_counts = Counter(df_empty[df_empty["any_fp"]]["group_code"])
    assess_fp_counts = Counter(df_empty[df_empty["any_fp"]]["assessment_model_vs_markup"])

    d_tr = dist_detail["pub_train"]
    d_va = dist_detail["pub_validation"]
    d_v3 = dist_detail["v3_test"]
    d_208 = dist_detail["official_208"]

    b_df = df_empty[df_empty["group_code"] == "B"]
    b_ambig_ids = [int(x) for x in b_df[b_df["assessment_model_vs_markup"] == "ambiguous_turnkey_services_listed_gold_empty"]["row_id"].tolist()]
    b_err_ids = [int(x) for x in b_df[b_df["assessment_model_vs_markup"] == "model_error_turnkey_photo_report_mention"]["row_id"].tolist()]
    b_clean_ids = [int(x) for x in b_df[b_df["assessment_model_vs_markup"] == "clean_tn_despite_service_mention"]["row_id"].tolist()]
    emp_fp_C = df_off[(pC_bin == 1) & (y == 0) & df_off["is_empty_gold"]]
    b_ambig_c_fp = int((emp_fp_C["empty_assess"] == "ambiguous_turnkey_services_listed_gold_empty").sum())
    b_err_c_fp = int((emp_fp_C["empty_assess"] == "model_error_turnkey_photo_report_mention").sum())

    # Compute exact pair-level non-empty FP attribution for C_syntax
    non_empty_fp_C_rows = df_off[(pC_bin == 1) & (y == 0) & (~df_off["is_empty_gold"])]
    non_empty_fp_non_conf = non_empty_fp_C_rows[~non_empty_fp_C_rows["is_conflict_dup"]]
    # Rows among non-conflicting non-empty FPs where gold is plausibly incomplete/borderline (explicitly states tile/wall-leveling/putty)
    borderline_non_empty_ids = {21, 24, 91, 99, 119, 134, 150, 188}
    n_ne_fp_borderline = int(non_empty_fp_non_conf["row_idx"].isin(borderline_non_empty_ids).sum())
    n_ne_fp_model = int((~non_empty_fp_non_conf["row_idx"].isin(borderline_non_empty_ids)).sum())

    report_lines = [
        "# Аудит датасетов, эталонной разметки и ошибок классификатора",
        "",
        "> **Режим аудита:** только чтение и анализ исходных данных и сохранённых предсказаний (`output/syntax-aware-experiment/`); модели не переобучались, пороги (`0.5`) не подбирались, Qwen/Colab не запускались, `public_synthetic_300` полностью исключён.",
        "",
        "---",
        "",
        "## 1. Проверка противоречивых дубликатов `official_208` с учётом `(description, source_mcId)`",
        "",
        f"- **Схема файла:** [public-test-dataset.parquet](../../datasets/public/public-test-dataset.parquet) содержит только столбцы `title`, `description`, `categories` (`source_mcId` в parquet **отсутствует**).",
        f"- **Контракт `source_mcId`:** по [datasets/public/README.md](../../datasets/public/README.md) и `README-files.txt`, исходная микрокатегория для **всех 208 записей** равна **`101` («Ремонт под ключ»)**, поэтому пара `(description, source_mcId)` для `official_208` не разделяет одинаковые тексты.",
        f"- **Всего групп совпадающих `description`:** `26` групп (`95` строк из `208`):",
        f"  1. **Совпадающий текст, но разные `source_mcId` (нормальная ситуация):** **`{n_diff_src_normal}` групп (`0` строк)**.",
        f"  2. **Совпадающий текст и одинаковый `source_mcId = 101`, но разные `gold` (настоящие противоречия):** **`{n_same_src_conflict_groups}` групп (`{n_same_src_conflict_rows}` строк)** — например, `row 13` (`['Плиточные работы', 'Сантехника', 'Электрика']`) vs `row 58` (`[]`), `row 40` (`[]`) vs `row 160` (все 5 категорий), `row 106` (`['Электрика']`) vs `rows 143/171/194` (`[]`). Даже при проверке пары `(title, description, source_mcId=101)` все **`13` групп (`56` строк)** остаются противоречивыми.",
        f"  3. **Идентичные строки с одинаковыми `gold`:** **`{n_same_src_same_gold_groups}` групп (`{n_same_src_same_gold_rows}` строк)**.",
        f"- Для сравнения: в `pub_train` (`3 398`), `pub_validation` (`729`) и `v3_test` (`600`) внутри каждого сплита число конфликтов при одинаковом `(description, source_mcId)` равно **`0`** (дубликаты по тексту предварительно вычищены).",
        f"- Полная выгрузка групп, предсказаний и вероятностей: [duplicate_groups.json](duplicate_groups.json).",
        "",
        "---",
        "",
        "## 2. Разбор всех `106` объявлений `official_208` с `gold = []`",
        "",
        "| Код | Группа | Всего строк | Строк с любым FP (`A`/`C`/`C_ng`) | Природа оставшихся FP (на уровне строк) |",
        "|---|---|---:|---:|---|",
        f"| `A` | Явный отказ от дополнительных/отдельных работ | {grp_counts['A']} | {grp_fp_counts['A']} | `row 103` (`Отдельные мелкие задачи не берём...`, ошибка модели); остальные 4 (`22, 74, 104, 176`) чисто подавлены (`[]`) |",
        f"| `B` | Услуги перечислены как часть «под ключ», без оговорки об отдельном заказе | {grp_counts['B']} | {grp_fp_counts['B']} | `{len(b_ambig_ids)}` спорных разметок с FP (`rows {b_ambig_ids}` -> `{b_ambig_c_fp}` парных FP в `C`), `{len(b_err_ids)}` ошибка модели на фотоотчёте (`rows {b_err_ids}` -> `{b_err_c_fp}` парных FP в `C`, всего `{attr_C['empty_by_group'].get('B', 0)}`), `{len(b_clean_ids)}` строки без FP (`rows {b_clean_ids}`) |",
        f"| `C` | Услуги прямо описаны без «под ключ» и без запрета | {grp_counts['C']} | {grp_fp_counts['C']} | `0` строк (после фильтрации фразы `расходы на отопление` в утеплении фасадов, отнесённой к `E`) |",
        f"| `D` | Услуги упомянуты в отрицании, сравнении, цитате, вопросе или как чужие работы | {grp_counts['D']} | {grp_fp_counts['D']} | `row 26` (`«не только про стены и плитку»` + техпроект в подарок — ошибка модели: 2 парных FP в `C`) |",
        f"| `E` | Нет упоминания целевых услуг `102–106` | {grp_counts['E']} | {grp_fp_counts['E']} | `8` строк с FP: `7` строк полисемии `«выравнивание дома/перекрытий» -> 102` (`2, 5, 39, 48, 95, 148, 157`) и `1` галлюцинация только у `A` в `row 17` («отделка парной в бане» -> `104`, исправлено в `C`) |",
        f"| `F` | Неоднозначная ситуация / конфликт дубликатов (`gold=[]` здесь, но `gold!=[]` у копии) | {grp_counts['F']} | {grp_fp_counts['F']} | Все `13` строк имеют идентичный `description` с непустым `gold` в том же `official_208` (`28, 29, 33, 35, 40, 46, 57, 58, 60, 66, 143, 171, 194`) |",
        "",
        "- Полная таблица по всем `106` объявлениям (включая вероятности `101–106`, предсказания и признаки Natasha): [empty_gold_analysis.csv](empty_gold_analysis.csv).",
        "",
        "---",
        "",
        "## 3. Проверка интерпретации требований Авито (ТЗ vs реальные данные)",
        "",
        "### Что явно следует из [Innoglobalhack-Avito_2026.md](../../mvp/innoglobalhack_avito/innoglobalhack_avito/Innoglobalhack-Avito_2026.md) и [microcategories-guide.md](../../mvp/innoglobalhack_avito/innoglobalhack_avito/microcategories-guide.md):",
        "- В разделе **`## Задача`** требуется: (1) выделить услуги, которые исполнитель **предлагает в тексте**, (2) определить их микрокатегории (`102–106`, кроме исходной), (3) вернуть `[]`, если таких услуг нет, и не учитывать отрицания, сравнения и чужие услуги.",
        "- В справочнике (`microcategories-guide.md`, секция 101): **слово «под ключ» само по себе не отменяет и не задаёт остальные категории** — нельзя вывести услуги лишь из общей фразы «ремонт под ключ», но конкретные упомянутые услуги остаются основанием, если исполнитель их предлагает.",
        "- Запрет **«Нельзя утверждать, что услугу можно заказать отдельно, если это не следует из исходного текста»** находится в разделе **`## Дополнительная задача — генерация текстов`** и регулирует **текст черновика**, а не требование выкидывать услуги из классификации в Ситуации 1.",
        "",
        "### Как две ситуации реально размечены в данных:",
        f"| Ситуация | `pub_train` (`3 398`) | `pub_validation` (`729`) | `official_208` (`208`) |",
        "|---|---:|---:|---:|",
        f"| **1. «Под ключ» + явные услуги `102–106`, без отказа от частичных** | `{d_tr['sit1']['total']}` (`{d_tr['sit1']['non_empty']}` непустых = **`{100*d_tr['sit1']['non_empty']/max(d_tr['sit1']['total'],1):.1f}%`**, `{d_tr['sit1']['empty']}` пустых) | `{d_va['sit1']['total']}` (`{d_va['sit1']['non_empty']}` непустых = **`{100*d_va['sit1']['non_empty']/max(d_va['sit1']['total'],1):.1f}%`**, `{d_va['sit1']['empty']}` пустых) | `{d_208['sit1']['total']}` (`{d_208['sit1']['non_empty']}` непустых = **`{100*d_208['sit1']['non_empty']/max(d_208['sit1']['total'],1):.1f}%`**, `{d_208['sit1']['empty']}` пустых) |",
        f"| **2. «Только под ключ» / «частичные работы не берём»** | `{d_tr['sit2']['total']}` (`{d_tr['sit2']['empty']}` пустых = **`{100*d_tr['sit2']['empty']/max(d_tr['sit2']['total'],1):.1f}%`**, `{d_tr['sit2']['non_empty']}` непустых) | `{d_va['sit2']['total']}` (`{d_va['sit2']['empty']}` пустых = **`{100*d_va['sit2']['empty']/max(d_va['sit2']['total'],1):.1f}%`**, `{d_va['sit2']['non_empty']}` непустых) | `{d_208['sit2']['total']}` (`{d_208['sit2']['empty']}` пустых = **`{100*d_208['sit2']['empty']/max(d_208['sit2']['total'],1):.1f}%`**, `{d_208['sit2']['non_empty']}` непустая `row 18`) |",
        "",
        "- **Что остаётся неоднозначным в разметке:** Ситуация 1 в `pub_train`/`pub_validation` размечается как **положительная в ~89–91% случаев**, однако в `official_208` пустая доля для неё достигает **`22.2%` (`20/90`)**, включая прямые противоречия дубликатов (`row 40` vs `160`, `row 29/57` и др.). Аналогично в Ситуации 2 часть объявлений с «НЕ ВЫПОЛНЯЕМ МЕЛКИЙ БЫТОВОЙ РЕМОНТ» (`pub_val 141, 261`) или «частичные работы не выполняем» при списке работ (`official_208 row 18`) всё равно размечены положительно, тогда как `rows 22, 74, 103, 104, 176` — как `[]`.",
        "",
        "---",
        "",
        "## 4. Сравнение распределений (`pub_train`, `pub_validation`, `v3_test`, `official_208`)",
        "",
        "- Полная таблица: [dataset_distributions.csv](dataset_distributions.csv).",
        f"- **`source_mcId`:** в `pub_train`, `pub_validation` и `official_208` везде `101`; в `v3_test` (`600`) равномерно по `100` строк на каждую из `101..106`.",
        f"- **Контраст по доле `gold = []`:** `pub_train` — `{100*d_tr['empty_count']/d_tr['n_ads']:.1f}%` (`{d_tr['empty_count']}`), `pub_validation` — `{100*d_va['empty_count']/d_va['n_ads']:.1f}%` (`{d_va['empty_count']}`), `v3_test` — **`{100*d_v3['empty_count']/d_v3['n_ads']:.1f}%`** (`{d_v3['empty_count']}`), `official_208` — **`{100*d_208['empty_count']/d_208['n_ads']:.1f}%`** (`{d_208['empty_count']}`).",
        f"- **Плотность лингвистических ловушек (доля пар с сигналом Natasha):**",
        f"  - `syn_has_neg_on_action`: в `v3_test` **`{d_v3['nat_neg_pairs']}`** пар (`{100*d_v3['nat_neg_pairs']/d_v3['active_pairs']:.2f}%`), против **`{d_va['nat_neg_pairs']}`** (`{100*d_va['nat_neg_pairs']/d_va['active_pairs']:.2f}%`) в `pub_validation` и **`{d_208['nat_neg_pairs']}`** (`{100*d_208['nat_neg_pairs']/d_208['active_pairs']:.2f}%`) в `official_208`.",
        f"  - `syn_third_party_actor`: в `v3_test` **`{d_v3['nat_tp_pairs']}`** пар (`{100*d_v3['nat_tp_pairs']/d_v3['active_pairs']:.2f}%`), против **`{d_va['nat_tp_pairs']}`** (`{100*d_va['nat_tp_pairs']/d_va['active_pairs']:.2f}%`) в `pub_validation` и **`{d_208['nat_tp_pairs']}`** (`{100*d_208['nat_tp_pairs']/d_208['active_pairs']:.2f}%`) в `official_208`.",
        "- **Вывод по V3:** да, `v3_test` существенно сдвинут к сложным отрицаниям и сторонним исполнителям (**в ~24–28 раз чаще на пару по отрицаниям и почти бесконечно чаще по стороннему исполнителю**, чем в `pub_validation` и `official_208`) и почти не содержит длинных «под ключ» объявлений с пустым `gold` (`12.0%` против `51.0%`). Прирост `+10.9 п.п.` на `v3_test` подтверждает эффективность синтаксиса на таких конструкциях, но в `official_208` они почти не встречаются.",
        "",
        "---",
        "",
        "## 5. Итоговая атрибуция FP на `official_208` (`A`, `C_syntax`, `C_no_global`)",
        "",
        f"- **Всего FP (`thr = 0.5`):** `A_qwen_sgd = {attr_A['total_fp']}`, `C_syntax = {attr_C['total_fp']}`, `C_no_global = {attr_Cng['total_fp']}`.",
        f"- **Сколько FP в объявлениях с формулировками «под ключ / комплексный ремонт / полный цикл»:** у `A` — **`{attr_A['in_turnkey_ads']}/{attr_A['total_fp']}`** (`{100*attr_A['in_turnkey_ads']/attr_A['total_fp']:.1f}%`), у `C_syntax` — **`{attr_C['in_turnkey_ads']}/{attr_C['total_fp']}`** (`{100*attr_C['in_turnkey_ads']/attr_C['total_fp']:.1f}%`), у `C_no_global` — **`{attr_Cng['in_turnkey_ads']}/{attr_Cng['total_fp']}`** (`{100*attr_Cng['in_turnkey_ads']/attr_Cng['total_fp']:.1f}%`).",
        f"- **Разделение `{attr_C['total_fp']}` парных FP у `C_syntax` (ошибки модели vs проблемы/неоднозначность разметки):**",
        f"  - **Внутри `gold = []` (`{attr_C['in_empty_gold']}` FP):** **`{attr_C['empty_model_errors']}`** — чистые ошибки модели (`7` полисемия «выравнивание дома/перекрытий -> 102», `3` на фотоотчёте `row 96`, `2` на сравнении/техпроекте `row 26`, `1` на явном отказе `row 103`); **`{attr_C['empty_markup_or_ambiguous']}`** — проблемы/неоднозначность разметки (`{attr_C['empty_by_group'].get('F',0)}` на конфликтующих дубликатах `F` и `26` на явных перечнях услуг «под ключ» в группе `B`, где нет запрета на отдельные заказы).",
        f"  - **Внутри `gold != []` (`{attr_C['in_non_empty_gold']}` FP):** **`{non_empty_fp_conflict}`** — в конфликтующих дубликатах; **`{n_ne_fp_borderline}`** — пограничные/вероятные пропуски в `gold` (`rows 21, 24, 99, 119` прямо пишут про «укладку плитки», `91, 134, 150, 188` — «выравнивание стен / шпатлёвка стен»); **`{n_ne_fp_model}`** — ошибки модели (`rows 4, 18, 130, 173, 174, 190`).",
        f"  - **Суммарно по `C_syntax` (`{attr_C['total_fp']}` FP):** **`{attr_C['empty_model_errors'] + n_ne_fp_model}`** явных ошибок модели (плюс `{n_ne_fp_borderline}` пограничных на непустых строках -> до **`{attr_C['empty_model_errors'] + n_ne_fp_model + n_ne_fp_borderline}`**) против **`{attr_C['empty_markup_or_ambiguous'] + non_empty_fp_conflict}`** (`57` на конфликтах дубликатов + `26` на «под ключ» `gold=[]` при перечислении услуг; с учётом `{n_ne_fp_borderline}` пограничных — до **`{attr_C['empty_markup_or_ambiguous'] + non_empty_fp_conflict + n_ne_fp_borderline}`**) на проблемах и неоднозначности разметки.",
        "- Все 208 записей, 30 FP и 15 FN с полными текстами сохранены в [official_208_full.jsonl](official_208_full.jsonl) и [error_examples.md](error_examples.md).",
    ]
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(report_lines))

    # Also print exact pair-level vs row-level empty_gold FP counts cleanly
    emp_fp_C = df_off[(pC_bin == 1) & (y == 0) & df_off["is_empty_gold"]]
    print("\n=== PAIR-LEVEL EMPTY_GOLD FP COUNTS (C_syntax / A / C_no_global) ===")
    for name, pb in [("A", pA_bin), ("C_syntax", pC_bin), ("C_no_global", pCng_bin)]:
        sub_fp = df_off[(pb == 1) & (y == 0)]
        sub_emp = sub_fp[sub_fp["is_empty_gold"]]
        sub_ne = sub_fp[~sub_fp["is_empty_gold"]]
        print(f"  {name:12s}: total_fp={len(sub_fp)} | empty_fp={len(sub_emp)} (by_assess={dict(Counter(sub_emp['empty_assess']))}) | non_empty_fp={len(sub_ne)} (in_conflict_dup={int(sub_ne['is_conflict_dup'].sum())})")

    print("=== AUDIT SUMMARY COUNTS ===")
    print("Duplicate groups official_208:", dup_output["summary"])
    print("Empty gold (106 rows) distribution by A..F:")
    print(df_empty.groupby(["group_code", "assessment_model_vs_markup"]).size().to_string())
    print("\nFP Attribution A / C_syntax / C_no_global:")
    print("  A:", attr_A)
    print("  C_syntax:", attr_C, "non_empty_fp_in_conflict_dup:", non_empty_fp_conflict)
    print("  C_no_global:", attr_Cng)
    print("\nDistributions detail:")
    for k, v in dist_detail.items():
        print(f"  {k}: n={v['n_ads']} empty={v['empty_count']} sit1={v['sit1']} sit2={v['sit2']} neg_pairs={v['nat_neg_pairs']} tp_pairs={v['nat_tp_pairs']} conf_rows={v['conf_pair_rows']}")


if __name__ == "__main__":
    main()
