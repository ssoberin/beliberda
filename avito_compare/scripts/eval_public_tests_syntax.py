#!/usr/bin/env python3
"""
Inference-only evaluation of frozen models (`tfidf_sgd`, `A_qwen_sgd`, `B0_score_cat`,
`B_meta`, `C_syntax`, `C_no_global`, `C_syntax_only_no_meta`) on:
  - `official_208`: datasets/public/public-test-dataset.parquet
  - `public_synthetic_300`: datasets/public/public-test-synthetic-300.parquet

Uses zero retraining and zero threshold tuning on tests.
Chunk embeddings are assembled by reusing `output/cat-aware-v2/cache/chunks_embeddings.npz`
plus the pre-encoded Colab T4 GPU delta embeddings (`public_tests_delta_embs.npz`).
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.analyze_syntax_errors import classify_error_bucket, trace_row_example
from avito_compare.scripts.audit_errors_cat_aware import PAST_OR_REFUSAL_RE
from avito_compare.scripts.eval_fixed_cat_aware import GLOBAL_TURNKEY_RE, compute_detailed_metrics
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.qwen3_embedding_experiment import load_official_parquet, normalize
from avito_compare.scripts.run_syntax_experiment import (
    bootstrap_f1_delta,
    build_pair_dataframe,
    error_transition_counts,
    probs_to_eval_matrix,
)
from avito_compare.scripts.syntax_features import (
    GLOBAL_CANDIDATE_RE,
    NON_SYNTAX_META_FEATURES,
    SYNTAX_CAT_FEATURES,
    SYNTAX_ONLY_FEATURES,
    SyntaxAnalyzer,
    detect_chunk_global_restriction,
    extract_chunk_syntax_signals,
)

PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
V2_CACHE_NPZ = ROOT / "output" / "cat-aware-v2" / "cache" / "chunks_embeddings.npz"
EXP_DIR = ROOT / "output" / "syntax-aware-experiment"
EXP_CACHE_DIR = EXP_DIR / "cache"
MODELS_DIR = EXP_DIR / "models"

OFFICIAL_208_PQ = ROOT / "datasets" / "public" / "public-test-dataset.parquet"
SYNTHETIC_300_PQ = ROOT / "datasets" / "public" / "public-test-synthetic-300.parquet"
DELTA_CHUNKS_JSON = EXP_CACHE_DIR / "public_tests_delta_chunks.json"
DELTA_EMBS_NPZ = EXP_CACHE_DIR / "public_tests_delta_embs.npz"


def reconstruct_v2_chunk_map() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Reconstruct exact chunk texts -> embeddings lookup from frozen v2 cache."""
    v2_splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }
    uniq_ads: list[str] = []
    ad_to_uid: dict[str, int] = {}
    for rows in v2_splits.values():
        for r in rows:
            txt = r["description"]
            if txt not in ad_to_uid:
                ad_to_uid[txt] = len(uniq_ads)
                uniq_ads.append(txt)
    uniq_chunks: list[str] = []
    chunk_to_cid: dict[str, int] = {}
    for txt in uniq_ads:
        for ch in chunk_ad_fixed(txt):
            if ch not in chunk_to_cid:
                chunk_to_cid[ch] = len(uniq_chunks)
                uniq_chunks.append(ch)
    npz = np.load(V2_CACHE_NPZ)
    chunk_emb = npz["chunk_emb"]
    query_emb = npz["query_emb"].astype(np.float32)
    assert len(uniq_chunks) == len(chunk_emb), f"Cache mismatch: {len(uniq_chunks)} vs {len(chunk_emb)}"
    return uniq_chunks, chunk_emb, query_emb


def load_test_splits() -> dict[str, list[dict]]:
    out = {}
    for sname, pq in [("official_208", OFFICIAL_208_PQ), ("public_synthetic_300", SYNTHETIC_300_PQ)]:
        _, _, rows = load_official_parquet(pq)
        for idx, r in enumerate(rows):
            r["item_id"] = f"{sname}_{idx}"
            r["source_mcId"] = 101
            r["labels"] = r["evaluation_labels"]
            r["construction"] = ""
        out[sname] = rows
    return out


def prepare_delta_chunks(v2_chunks: list[str], test_splits: dict[str, list[dict]]) -> list[str]:
    v2_set = set(v2_chunks)
    delta_set: set[str] = set()
    for rows in test_splits.values():
        for r in rows:
            for ch in chunk_ad_fixed(r["description"]):
                if ch not in v2_set:
                    delta_set.add(ch)
    # Sort by length ascending to minimize transformer padding on GPU
    delta_sorted = sorted(delta_set, key=lambda s: (len(s), s))
    EXP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DELTA_CHUNKS_JSON.write_text(json.dumps(delta_sorted, ensure_ascii=False) + "\n", encoding="utf-8")
    return delta_sorted


def build_public_test_slices(
    sname: str,
    rows: list[dict],
    n_chunks_arr: np.ndarray,
    train_norm_texts: set[str],
) -> dict[str, np.ndarray]:
    descs = [r["description"] for r in rows]
    slices = {
        "all": np.ones(len(rows), dtype=bool),
        "short_lt_300": np.asarray([len(d) < 300 for d in descs], dtype=bool),
        "n_chunks_1": n_chunks_arr == 1,
        "n_chunks_ge_3": n_chunks_arr >= 3,
        "negations": np.asarray([bool(cae.NEG_CUE_RE.search(d)) for d in descs], dtype=bool),
        "third_party_or_contractor": np.asarray([bool(cae.CONTRACTOR_RE.search(d)) for d in descs], dtype=bool),
        "global_turnkey_only": np.asarray([bool(GLOBAL_TURNKEY_RE.search(d)) for d in descs], dtype=bool),
        "past_or_refusal": np.asarray([bool(PAST_OR_REFUSAL_RE.search(d)) for d in descs], dtype=bool),
        "empty_gold": np.asarray([len(r["evaluation_labels"]) == 0 for r in rows], dtype=bool),
        "non_empty_gold": np.asarray([len(r["evaluation_labels"]) > 0 for r in rows], dtype=bool),
    }
    if sname == "official_208":
        norms = [normalize(d) for d in descs]
        slices["no_train_overlap"] = np.asarray([nm not in train_norm_texts for nm in norms], dtype=bool)
        lbls_by_norm: dict[str, set[tuple[int, ...]]] = defaultdict(set)
        for r, nm in zip(rows, norms):
            lbls_by_norm[nm].add(tuple(sorted(r["evaluation_labels"])))
        conflicting_norms = {nm for nm, lsets in lbls_by_norm.items() if len(lsets) > 1}
        slices["non_conflicting_groups"] = np.asarray([nm not in conflicting_norms for nm in norms], dtype=bool)
        slices["clean_no_overlap_non_conflicting"] = slices["no_train_overlap"] & slices["non_conflicting_groups"]
    return slices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-delta-only", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    print("1. Loading frozen v2 chunk dictionary & public test parquets...")
    v2_chunks, v2_chunk_emb, query_emb_f32 = reconstruct_v2_chunk_map()
    test_splits = load_test_splits()
    for sname, rows in test_splits.items():
        print(f"   {sname:22s}: {len(rows)} ads")

    delta_sorted = prepare_delta_chunks(v2_chunks, test_splits)
    print(f"   Prepared {len(delta_sorted)} delta chunks -> {DELTA_CHUNKS_JSON}")
    if args.prepare_delta_only:
        return

    assert DELTA_EMBS_NPZ.exists(), f"Missing {DELTA_EMBS_NPZ}. Run Colab T4 delta encoding first!"
    ck = np.load(DELTA_EMBS_NPZ, allow_pickle=False)
    delta_emb = ck["embs"]
    assert len(delta_emb) == len(delta_sorted), f"Delta count mismatch: {len(delta_emb)} vs {len(delta_sorted)}"
    norms = np.linalg.norm(delta_emb.astype(np.float32), axis=1)
    assert np.all((norms > 0.98) & (norms < 1.02)), f"Bad delta norms: min={norms.min()}, max={norms.max()}"
    print(f"   Verified Colab delta embeddings: shape={delta_emb.shape}, norms=[{norms.min():.4f}, {norms.max():.4f}]")

    # Assemble combined chunk pool for the public test splits
    chunk_emb_map = {ch: v2_chunk_emb[i] for i, ch in enumerate(v2_chunks)}
    for i, ch in enumerate(delta_sorted):
        chunk_emb_map[ch] = delta_emb[i]

    uniq_chunks: list[str] = []
    chunk_to_cid: dict[str, int] = {}
    offsets = [0]
    cindices: list[int] = []
    split_uids: dict[str, np.ndarray] = {}
    split_nchunks: dict[str, np.ndarray] = {}

    uid_counter = 0
    for sname, rows in test_splits.items():
        uids = []
        nch = []
        for r in rows:
            cs = chunk_ad_fixed(r["description"])
            for ch in cs:
                if ch not in chunk_to_cid:
                    chunk_to_cid[ch] = len(uniq_chunks)
                    uniq_chunks.append(ch)
                cindices.append(chunk_to_cid[ch])
            offsets.append(len(cindices))
            uids.append(uid_counter)
            nch.append(len(cs))
            uid_counter += 1
        split_uids[sname] = np.asarray(uids, dtype=np.int32)
        split_nchunks[sname] = np.asarray(nch, dtype=np.int32)

    offsets_arr = np.asarray(offsets, dtype=np.int64)
    cindices_arr = np.asarray(cindices, dtype=np.int32)
    chunk_emb_f32 = np.vstack([chunk_emb_map[ch].astype(np.float32) for ch in uniq_chunks])
    raw_sims = chunk_emb_f32 @ query_emb_f32.T
    chunk_cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    print("2. Loading frozen models & building 3078-d + TF-IDF predictions...")
    sgd_bundle = joblib.load(MODELS_DIR / "sgd_full_heads.joblib")
    sgd_heads = sgd_bundle["heads"]
    tfidf_clf = joblib.load(MODELS_DIR / "tfidf_baseline.joblib")
    prev_metrics = json.loads((EXP_DIR / "metrics.json").read_text(encoding="utf-8"))

    variants_spec = {
        "B0_score_cat": (["p_sgd", "cat_id"], ["cat_id"]),
        "B_meta": (["p_sgd"] + NON_SYNTAX_META_FEATURES, ["cat_id"]),
        "C_syntax": (["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES, SYNTAX_CAT_FEATURES),
        "C_no_global": (
            ["p_sgd"] + NON_SYNTAX_META_FEATURES + [
                c for c in SYNTAX_ONLY_FEATURES
                if c not in ("syn_global_turnkey_only", "syn_global_partial_refusal", "syn_global_outside_top3")
            ],
            SYNTAX_CAT_FEATURES,
        ),
        "C_syntax_only_no_meta": (["p_sgd", "cat_id"] + SYNTAX_ONLY_FEATURES, SYNTAX_CAT_FEATURES),
    }
    cb_models = {}
    for vname in variants_spec:
        cb = CatBoostClassifier()
        cb.load_model(str(MODELS_DIR / f"catboost_{vname}.cbm"))
        cb_models[vname] = cb

    split_feats = {}
    probs_A = {}
    probs_tf = {}
    y_eval_by_split = {}
    src_by_split = {}
    t_sgd0 = time.perf_counter()
    for sname, rows in test_splits.items():
        y_ev_5, src_ev = cae._rows_xy_for_ids(rows, cae.SCOPE_102_106, use_eval=True)
        y_eval_by_split[sname] = y_ev_5
        src_by_split[sname] = src_ev
        feats, _ = cae.build_category_features_for_uids(
            split_uids[sname], offsets_arr, cindices_arr, chunk_emb_f32, chunk_cat_sims, return_selection=False
        )
        split_feats[sname] = feats
        probs_A[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])
        probs_tf[sname] = tfidf_clf.predict_proba([r["description"] for r in rows])
    sgd_inf_sec = time.perf_counter() - t_sgd0

    print("3. Parsing unique test chunks via Natasha UD...")
    needed_chunk_cat_keys: set[tuple[str, int]] = set()
    needed_global_chunks: set[str] = set()
    for sname, rows in test_splits.items():
        for r_idx, uid in enumerate(split_uids[sname]):
            start, end = int(offsets_arr[int(uid)]), int(offsets_arr[int(uid) + 1])
            ad_cids = cindices_arr[start:end]
            sims_mat = chunk_cat_sims[ad_cids]
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch_txt = uniq_chunks[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch_txt):
                    needed_global_chunks.add(ch_txt)
            for cid in cae.SCOPE_102_106:
                col_101 = cae.ALL_IDS.index(cid)
                order = np.argsort(-sims_mat[:, col_101], kind="stable")[:k_take]
                for p in order:
                    needed_chunk_cat_keys.add((uniq_chunks[int(ad_cids[p])], cid))

    analyzer = SyntaxAnalyzer()
    t_nat0 = time.perf_counter()
    for ch in uniq_chunks:
        analyzer.parse_text(ch)
    natasha_parse_sec = time.perf_counter() - t_nat0

    t_ext0 = time.perf_counter()
    chunk_sig_cache = {k: extract_chunk_syntax_signals(analyzer, k[0], k[1]) for k in needed_chunk_cat_keys}
    chunk_global_cache = {ch: detect_chunk_global_restriction(analyzer, ch) for ch in needed_global_chunks}
    feat_extract_sec = time.perf_counter() - t_ext0
    print(
        f"   Parsed {len(uniq_chunks)} unique chunks in {natasha_parse_sec:.2f}s "
        f"({1000.0 * natasha_parse_sec / len(uniq_chunks):.2f} ms/chunk)"
    )

    print("4. Building pair features & running frozen CatBoost inference...")
    dfs_by_split = {}
    all_feat_cols = ["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES
    num_cols = [c for c in all_feat_cols if c not in SYNTAX_CAT_FEATURES]
    for sname, rows in test_splits.items():
        df_s = build_pair_dataframe(
            sname, rows, split_uids[sname], offsets_arr, cindices_arr, chunk_cat_sims, uniq_chunks,
            y_eval_by_split[sname], src_by_split[sname], probs_A[sname], chunk_sig_cache, chunk_global_cache,
        )
        assert not df_s[all_feat_cols].isna().any().any(), f"NaN in {sname}"
        assert np.all(np.isfinite(df_s[num_cols].to_numpy(dtype=np.float64))), f"Inf in {sname}"
        dfs_by_split[sname] = df_s

    train_norm_texts = {
        normalize(r["description"])
        for r in (
            cae.load_public_rows(PUBLIC_DIR / "train.jsonl")
            + cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl")
        )
    }

    th05 = np.full(5, 0.5)
    metrics_out = {
        "timing": {
            "unique_chunks_total": len(uniq_chunks),
            "reused_from_v2": len(set(uniq_chunks) & set(v2_chunks)),
            "delta_encoded_colab_t4": len(delta_sorted),
            "natasha_parse_sec": round(natasha_parse_sec, 3),
            "feat_extract_sec": round(feat_extract_sec, 3),
            "sgd_and_tfidf_sec": round(sgd_inf_sec, 3),
        },
        "splits": {},
        "transitions_and_bootstrap": {},
    }

    pred_dfs = []
    ea_summary = {}
    ea_examples = {}

    for sname, rows in test_splits.items():
        df_s = dfs_by_split[sname].copy()
        y_s = y_eval_by_split[sname]
        src_s = src_by_split[sname]

        probs_map = {
            "tfidf_sgd": probs_tf[sname],
            "A_qwen_sgd": probs_A[sname],
        }
        t_cb0 = time.perf_counter()
        for vname, (fcols, cat_cols) in variants_spec.items():
            p_vec = cb_models[vname].predict_proba(Pool(df_s[fcols], cat_features=cat_cols))[:, 1]
            probs_map[vname] = probs_to_eval_matrix(df_s, p_vec, len(rows), src_s, cae.SCOPE_102_106)
        cb_inf_sec = time.perf_counter() - t_cb0

        metrics_out["splits"][sname] = {"thr_05": {}, "thr_cal": {}, "slices_05": {}, "cb_infer_sec": round(cb_inf_sec, 4)}
        preds_05 = {}
        for vname, pr_mat in probs_map.items():
            m05, yp05 = compute_detailed_metrics(y_s, pr_mat, th05, src_s, cae.SCOPE_102_106)
            preds_05[vname] = yp05
            metrics_out["splits"][sname]["thr_05"][vname] = m05

            cal_ths = np.asarray(prev_metrics["splits"]["pub_validation"]["thr_cal"][vname]["thresholds"], dtype=float)
            mcal, _ = compute_detailed_metrics(y_s, pr_mat, cal_ths, src_s, cae.SCOPE_102_106)
            metrics_out["splits"][sname]["thr_cal"][vname] = {"thresholds": cal_ths.tolist(), **mcal}

        slices = build_public_test_slices(sname, rows, split_nchunks[sname], train_norm_texts)
        for sl_name, mk in slices.items():
            if int(mk.sum()) == 0:
                continue
            sl_block = {"rows": int(mk.sum())}
            for vname in ("tfidf_sgd", "A_qwen_sgd", "B_meta", "C_syntax", "C_no_global"):
                ms, _ = compute_detailed_metrics(y_s[mk], probs_map[vname][mk], th05, src_s[mk], cae.SCOPE_102_106)
                sl_block[vname] = ms
            metrics_out["splits"][sname]["slices_05"][sl_name] = sl_block

        ypA, ypB, ypC = preds_05["A_qwen_sgd"], preds_05["B_meta"], preds_05["C_syntax"]
        metrics_out["transitions_and_bootstrap"][sname] = {
            "A_to_B": error_transition_counts(y_s, ypA, ypB, src_s, cae.SCOPE_102_106),
            "A_to_C": error_transition_counts(y_s, ypA, ypC, src_s, cae.SCOPE_102_106),
            "B_to_C": error_transition_counts(y_s, ypB, ypC, src_s, cae.SCOPE_102_106),
            "boot_C_minus_A": bootstrap_f1_delta(y_s, ypA, ypC, src_s, cae.SCOPE_102_106),
            "boot_C_minus_B": bootstrap_f1_delta(y_s, ypB, ypC, src_s, cae.SCOPE_102_106),
        }

        cat_to_col = {cid: j for j, cid in enumerate(cae.SCOPE_102_106)}
        for vcol, vkey in [
            ("tfidf", "tfidf_sgd"),
            ("A", "A_qwen_sgd"),
            ("B", "B_meta"),
            ("C", "C_syntax"),
            ("C_no_global", "C_no_global"),
        ]:
            df_s[f"p_{vcol}"] = [round(float(probs_map[vkey][int(r["row_idx"]), cat_to_col[int(r["cat_id"])]]), 6) for _, r in df_s.iterrows()]
            df_s[f"pred_{vcol}_05"] = [int(preds_05[vkey][int(r["row_idx"]), cat_to_col[int(r["cat_id"])]]) for _, r in df_s.iterrows()]
        pred_dfs.append(df_s)

        # Error analysis buckets and real UD traces
        all_C_wrong = df_s[df_s["pred_C_05"] != df_s["y_true"]]
        buckets = Counter(classify_error_bucket(r) for _, r in all_C_wrong.iterrows())
        fixed_AC = df_s[(df_s["pred_A_05"] != df_s["y_true"]) & (df_s["pred_C_05"] == df_s["y_true"])]
        reg_AC = df_s[(df_s["pred_A_05"] == df_s["y_true"]) & (df_s["pred_C_05"] != df_s["y_true"])]
        both_wrong = df_s[(df_s["pred_A_05"] != df_s["y_true"]) & (df_s["pred_C_05"] != df_s["y_true"])]
        ea_summary[sname] = {
            "active_pairs": int(len(df_s)),
            "A_to_C": {"fixed": int(len(fixed_AC)), "broken": int(len(reg_AC)), "both_wrong": int(len(both_wrong))},
            "C_error_root_cause_buckets": dict(buckets),
        }

        def pick(df_sl, lim=8):
            out_tr = []
            for _, r in df_sl.head(lim).iterrows():
                out_tr.append(trace_row_example(analyzer, rows[int(r["row_idx"])], r, r))
            return out_tr

        ea_examples[sname] = {
            "A_wrong_C_right_rescued": pick(fixed_AC.assign(_g=(fixed_AC["p_A"] - fixed_AC["p_C"]).abs()).sort_values("_g", ascending=False)),
            "A_right_C_wrong_regressions": pick(reg_AC.assign(_g=(reg_AC["p_A"] - reg_AC["p_C"]).abs()).sort_values("_g", ascending=False)),
            "both_wrong_unresolved": pick(both_wrong.assign(_c=(both_wrong["p_C"] - 0.5).abs()).sort_values("_c", ascending=False)),
        }

    metrics_out["timing"]["total_eval_seconds"] = round(time.perf_counter() - t0, 2)
    pd.concat(pred_dfs, ignore_index=True).to_csv(EXP_DIR / "public_tests_predictions.csv", index=False)
    (EXP_DIR / "public_tests_metrics.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (EXP_DIR / "public_tests_error_analysis.json").write_text(
        json.dumps({"summary": ea_summary, "examples": ea_examples}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== PUBLIC TESTS SUMMARY (thr = 0.5) ===")
    for sname in ("official_208", "public_synthetic_300"):
        print(f"\nSplit: {sname}")
        for vname in ("tfidf_sgd", "A_qwen_sgd", "B0_score_cat", "B_meta", "C_syntax", "C_no_global", "C_syntax_only_no_meta"):
            m = metrics_out["splits"][sname]["thr_05"][vname]
            c = m["confusion"]
            print(
                f"  {vname:22s}: MicroF1={m['micro_f1']:.4f} MacroF1={m['macro_f1']:.4f} "
                f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
                f"FP={c['fp']:3d} FN={c['fn']:3d} Exact={m['exact_match']:.4f} AP={m['micro_ap']:.4f}"
            )
    print(f"\nSaved -> {EXP_DIR / 'public_tests_metrics.json'}")


if __name__ == "__main__":
    main()
