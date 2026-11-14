#!/usr/bin/env python3
"""
Detailed post-hoc error analysis comparing A (Qwen+SGD), B (Meta CatBoost),
and C (Syntax+Meta CatBoost) on `pub_validation` and `v3_test`.

Classifies regressions (`A_right -> C_wrong`) and remaining errors (`both_wrong`)
into the 5 root-cause buckets (parser errors, extraction rule misses, CatBoost
non-use/overweighting, Top-3 retrieval misses, and simple-case regressions) and
emits concrete real examples with UD parse traces into `error_analysis.json`.
"""
from __future__ import annotations

import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.syntax_features import SyntaxAnalyzer, extract_pair_features

PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
OUT_DIR = ROOT / "output" / "syntax-aware-experiment"


def trace_row_example(
    analyzer: SyntaxAnalyzer,
    row_raw: dict,
    pred_row: pd.Series,
    feat_row: pd.Series,
) -> dict:
    desc = row_raw["description"]
    chunks = chunk_ad_fixed(desc)
    top3_pos = [int(x) for x in str(pred_row["top3_positions"]).split("|") if x != ""]
    top3_sims = [float(feat_row["sim_slot0"]), float(feat_row["sim_slot1"]), float(feat_row["sim_slot2"])][: len(top3_pos)]
    cat_id = int(pred_row["cat_id"])
    dbg = extract_pair_features(
        analyzer, chunks, top3_pos, top3_sims, cat_id, len(desc), include_debug=True
    )
    return {
        "split": str(pred_row["split"]),
        "item_id": str(pred_row["item_id"]),
        "construction": str(pred_row["construction"]),
        "cat_id": cat_id,
        "y_true": int(pred_row["y_true"]),
        "p_A": round(float(pred_row["p_A"]), 4),
        "p_B": round(float(pred_row["p_B"]), 4),
        "p_C": round(float(pred_row["p_C"]), 4),
        "pred_A": int(pred_row["pred_A_05"]),
        "pred_B": int(pred_row["pred_B_05"]),
        "pred_C": int(pred_row["pred_C_05"]),
        "n_chunks": len(chunks),
        "top3_positions": top3_pos,
        "top3_chunks": [chunks[p] for p in top3_pos if p < len(chunks)],
        "best_chunk_polarity": str(pred_row["best_chunk_polarity"]),
        "dominant_subject": str(pred_row["dominant_subject"]),
        "temporal_state": str(pred_row["temporal_state"]),
        "service_link_status": str(pred_row["service_link_status"]),
        "syn_affirmative_offer": int(pred_row["syn_affirmative_offer"]),
        "syn_has_neg_on_action": int(pred_row["syn_has_neg_on_action"]),
        "syn_global_turnkey_only": int(pred_row["syn_global_turnkey_only"]),
        "syn_global_partial_refusal": int(pred_row["syn_global_partial_refusal"]),
        "syn_global_outside_top3": int(pred_row["syn_global_outside_top3"]),
        "syn_service_mentioned_top3": int(feat_row["syn_service_mentioned_top3"]),
        "syn_service_mentioned_ad": int(feat_row["syn_service_mentioned_ad"]),
        "clauses": dbg.get("_debug_clauses", []),
        "global_traces": dbg.get("_debug_global_traces", []),
    }


def classify_error_bucket(r: pd.Series) -> str:
    """Assign an error pair (`pred_C != y_true`) to one of 5 primary diagnostic buckets."""
    yt = int(r["y_true"])
    pa = int(r["pred_A_05"])
    pol = str(r["best_chunk_polarity"])
    in_top3 = int(r["syn_service_mentioned_top3"])
    in_ad = int(r["syn_service_mentioned_ad"])

    # Bucket 5: Regression where A was right and C flipped wrong
    if pa == yt:
        if yt == 1 and pol in ("negated", "third_party") or (yt == 0 and pol in ("offered", "double_neg_positive")):
            return "5b_regression_syntax_misled_catboost"
        if in_top3 == 0 or pol == "unknown":
            return "5a_regression_simple_or_calibration_shift"
        return "5c_regression_other_feature_shift"

    # Both A and C are wrong (`pred_A != y_true` and `pred_C != y_true`):
    # Bucket 4: Mention exists in ad, missed by Top-3 retrieval (outside Top-3)
    if in_ad == 1 and in_top3 == 0:
        return "4_retrieval_outside_top3"
    if in_ad == 0 and in_top3 == 0:
        if int(r["syn_global_outside_top3"]) == 1:
            return "4_global_restriction_outside_top3_or_implicit"
        return "4_no_lexical_mention_in_ad"

    # Mention IS in Top-3 (`in_top3 == 1`):
    # Bucket 1/2a: Parser or extraction failed to link (`unknown`)
    if pol == "unknown":
        if str(r["service_link_status"]) == "orphan_mention":
            return "1_parser_broken_arc_or_orphan_mention"
        return "2a_extractor_returned_unknown"

    # Bucket 2b: Polarity disagrees with gold label (`extractor wrong polarity`)
    if (yt == 0 and pol in ("offered", "double_neg_positive")) or (yt == 1 and pol in ("negated", "third_party")):
        return "2b_extractor_or_parser_wrong_polarity"

    # Bucket 3: Syntax features got the polarity/restriction direction RIGHT, yet CatBoost failed to flip A!
    return "3_catboost_underweighted_correct_syntax_vs_psgd"


def main() -> None:
    preds = pd.read_csv(OUT_DIR / "predictions.csv")
    feats = pd.read_csv(OUT_DIR / "syntax_features.csv")
    merged = pd.concat(
        [
            preds.reset_index(drop=True),
            feats[
                [
                    "ad_char_len", "n_chunks", "n_padded_slots",
                    "sim_slot0", "sim_slot1", "sim_slot2", "sim_max", "sim_mean",
                    "syn_service_mentioned_top3", "syn_service_mentioned_ad",
                ]
            ].reset_index(drop=True),
        ],
        axis=1,
    )

    raw_splits = {
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }
    analyzer = SyntaxAnalyzer()

    summary = {}
    examples = {}

    for sp in ("pub_validation", "v3_test"):
        sub = merged[merged["split"] == sp].copy()
        rows_raw = raw_splits[sp]

        fixed_AC = sub[(sub["pred_A_05"] != sub["y_true"]) & (sub["pred_C_05"] == sub["y_true"])]
        reg_AC = sub[(sub["pred_A_05"] == sub["y_true"]) & (sub["pred_C_05"] != sub["y_true"])]
        fixed_BC = sub[(sub["pred_B_05"] != sub["y_true"]) & (sub["pred_C_05"] == sub["y_true"])]
        reg_BC = sub[(sub["pred_B_05"] == sub["y_true"]) & (sub["pred_C_05"] != sub["y_true"])]
        both_AC_wrong = sub[(sub["pred_A_05"] != sub["y_true"]) & (sub["pred_C_05"] != sub["y_true"])]
        all_C_wrong = sub[sub["pred_C_05"] != sub["y_true"]]

        buckets = Counter(classify_error_bucket(r) for _, r in all_C_wrong.iterrows())
        fixed_constructions = Counter(fixed_AC["construction"].fillna("none").tolist())
        reg_constructions = Counter(reg_AC["construction"].fillna("none").tolist())
        both_wrong_constructions = Counter(both_AC_wrong["construction"].fillna("none").tolist())

        # Also check error breakdown specifically by gold polarity (`FP` vs `FN`)
        c_fp = int(((sub["pred_C_05"] == 1) & (sub["y_true"] == 0)).sum())
        c_fn = int(((sub["pred_C_05"] == 0) & (sub["y_true"] == 1)).sum())
        a_fp = int(((sub["pred_A_05"] == 1) & (sub["y_true"] == 0)).sum())
        a_fn = int(((sub["pred_A_05"] == 0) & (sub["y_true"] == 1)).sum())
        b_fp = int(((sub["pred_B_05"] == 1) & (sub["y_true"] == 0)).sum())
        b_fn = int(((sub["pred_B_05"] == 0) & (sub["y_true"] == 1)).sum())

        summary[sp] = {
            "active_pairs": int(len(sub)),
            "errors_A": {"total": a_fp + a_fn, "fp": a_fp, "fn": a_fn},
            "errors_B": {"total": b_fp + b_fn, "fp": b_fp, "fn": b_fn},
            "errors_C": {"total": c_fp + c_fn, "fp": c_fp, "fn": c_fn},
            "A_to_C": {"fixed": int(len(fixed_AC)), "broken": int(len(reg_AC)), "both_wrong": int(len(both_AC_wrong))},
            "B_to_C": {"fixed": int(len(fixed_BC)), "broken": int(len(reg_BC))},
            "C_error_root_cause_buckets": dict(buckets),
            "fixed_AC_by_construction": dict(fixed_constructions),
            "broken_AC_by_construction": dict(reg_constructions),
            "both_wrong_by_construction": dict(both_wrong_constructions),
        }

        def pick_traces(df_slice: pd.DataFrame, limit: int = 8) -> list[dict]:
            out = []
            for idx, r in df_slice.head(limit).iterrows():
                ri = int(r["row_idx"])
                out.append(trace_row_example(analyzer, rows_raw[ri], r, r))
            return out

        examples[sp] = {
            "A_wrong_C_right_rescued": pick_traces(
                fixed_AC.assign(_gap=(fixed_AC["p_A"] - fixed_AC["p_C"]).abs()).sort_values("_gap", ascending=False),
                limit=8,
            ),
            "B_wrong_C_right_syntax_specific": pick_traces(
                fixed_BC.assign(_gap=(fixed_BC["p_B"] - fixed_BC["p_C"]).abs()).sort_values("_gap", ascending=False),
                limit=8,
            ),
            "A_right_C_wrong_regressions": pick_traces(
                reg_AC.assign(_gap=(reg_AC["p_A"] - reg_AC["p_C"]).abs()).sort_values("_gap", ascending=False),
                limit=8,
            ),
            "both_wrong_unresolved": pick_traces(
                both_AC_wrong.assign(_conf=(both_AC_wrong["p_C"] - 0.5).abs()).sort_values("_conf", ascending=False),
                limit=8,
            ),
        }

    out = {"summary": summary, "examples": examples}
    (OUT_DIR / "error_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved -> {OUT_DIR / 'error_analysis.json'}")


if __name__ == "__main__":
    main()
