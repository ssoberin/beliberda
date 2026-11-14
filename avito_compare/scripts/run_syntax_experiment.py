#!/usr/bin/env python3
"""
Leakage-free experiment testing whether Russian UD dependency-tree syntax features
(Natasha/Slovnet) improve Category-Aware Qwen3-Embedding-0.6B + SGDClassifier
in the strict `public_plus_v3` mode.

Outputs are written exclusively to `output/syntax-aware-experiment/`.
Existing baselines and caches in `output/cat-aware-v2/` are read-only and untouched.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect

if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import json
import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.audit_errors_cat_aware import PAST_OR_REFUSAL_RE
from avito_compare.scripts.eval_fixed_cat_aware import (
    GLOBAL_TURNKEY_RE,
    build_split_slices,
    compute_detailed_metrics,
)
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.qwen3_v3_experiment import build_tfidf_pipeline
from avito_compare.scripts.syntax_features import (
    GLOBAL_CANDIDATE_RE,
    NON_SYNTAX_META_FEATURES,
    SYNTAX_CAT_FEATURES,
    SYNTAX_ONLY_FEATURES,
    SyntaxAnalyzer,
    chunk_mentions_category,
    detect_chunk_global_restriction,
    extract_chunk_syntax_signals,
    extract_pair_features,
)

SEED = 42
PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
CACHE_NPZ = ROOT / "output" / "cat-aware-v2" / "cache" / "chunks_embeddings.npz"
OUT_DIR = ROOT / "output" / "syntax-aware-experiment"
EXP_CACHE_DIR = OUT_DIR / "cache"
MODELS_DIR = OUT_DIR / "models"

CATBOOST_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "Logloss",
    "iterations": 1200,
    "depth": 4,
    "learning_rate": 0.05,
    "l2_leaf_reg": 5.0,
    "random_seed": SEED,
    "early_stopping_rounds": 50,
    "thread_count": 4,
    "verbose": 200,
}


def compute_dup_groups(descriptions: list[str], sim_threshold: float = 0.85) -> tuple[np.ndarray, dict]:
    """
    Group near-duplicate training ads via TF-IDF (1,2) cosine similarity >= 0.85
    connected components so StratifiedGroupKFold never leaks template clones across folds.
    """
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, dtype=np.float32)
    X = normalize(vec.fit_transform(descriptions))
    sim = (X @ X.T).tocoo()
    mask = (sim.row < sim.col) & (sim.data >= sim_threshold)
    from scipy.sparse import csr_matrix

    n = len(descriptions)
    adj = csr_matrix(
        (np.ones(int(mask.sum()), dtype=np.int8), (sim.row[mask], sim.col[mask])),
        shape=(n, n),
    )
    adj = adj + adj.T
    n_comp, labels = connected_components(adj, directed=False)
    sizes = Counter(labels.tolist())
    multi_groups = sum(1 for v in sizes.values() if v > 1)
    max_size = max(sizes.values())
    diag = {
        "n_ads": n,
        "n_components": int(n_comp),
        "multi_item_groups": int(multi_groups),
        "max_group_size": int(max_size),
        "sim_threshold": sim_threshold,
    }
    return labels, diag


def fit_leakage_free_oof_sgd(
    X_by_cat: dict[int, np.ndarray],
    y_tr_5: np.ndarray,
    dup_groups: np.ndarray,
) -> tuple[np.ndarray, dict[int, list[float]]]:
    """
    Produce strictly leakage-free 5-fold StratifiedGroupKFold OOF SGD probabilities on train.
    Alpha is chosen inside each outer fold via inner 3-fold StratifiedGroupKFold on tr_idx only.
    """
    n = len(y_tr_5)
    oof_probs = np.zeros((n, len(cae.SCOPE_102_106)), dtype=np.float64)
    chosen_alphas: dict[int, list[float]] = {}
    outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)

    for col, cid in enumerate(cae.SCOPE_102_106):
        X = X_by_cat[cid]
        y = y_tr_5[:, col]
        chosen_alphas[cid] = []
        for fold, (tr_idx, va_idx) in enumerate(outer_cv.split(X, y, dup_groups)):
            X_tr_f, y_tr_f, grp_tr_f = X[tr_idx], y[tr_idx], dup_groups[tr_idx]
            best_alpha = cae.ALPHA_GRID[0]
            best_inner_f1 = -1.0
            for alpha in cae.ALPHA_GRID:
                in_oof = np.zeros(len(tr_idx), dtype=np.float64)
                for itr, iva in inner_cv.split(X_tr_f, y_tr_f, grp_tr_f):
                    clf_in = SGDClassifier(**{**cae.SGD_PARAMS, "alpha": alpha}).fit(X_tr_f[itr], y_tr_f[itr])
                    in_oof[iva] = clf_in.predict_proba(X_tr_f[iva])[:, 1]
                f1_in = float(f1_score(y_tr_f, in_oof >= 0.5, zero_division=0))
                if f1_in > best_inner_f1:
                    best_inner_f1 = f1_in
                    best_alpha = alpha
            chosen_alphas[cid].append(float(best_alpha))
            clf_out = SGDClassifier(**{**cae.SGD_PARAMS, "alpha": best_alpha}).fit(X_tr_f, y_tr_f)
            oof_probs[va_idx, col] = clf_out.predict_proba(X[va_idx])[:, 1]
    return oof_probs, chosen_alphas


def probs_to_eval_matrix(
    df_pairs: pd.DataFrame,
    probs_col: np.ndarray,
    n_rows: int,
    sources: np.ndarray,
    ids_tuple: tuple[int, ...],
) -> np.ndarray:
    """Pivot active-pair probability vector back to (n_rows, len(ids_tuple)) matrix for split scoring."""
    mat = np.zeros((n_rows, len(ids_tuple)), dtype=np.float64)
    cat_to_col = {cid: j for j, cid in enumerate(ids_tuple)}
    ridxs = df_pairs["row_idx"].to_numpy()
    cols = df_pairs["cat_id"].astype(int).map(cat_to_col).to_numpy()
    mat[ridxs, cols] = probs_col
    for i, src in enumerate(sources):
        if int(src) in cat_to_col:
            mat[i, cat_to_col[int(src)]] = 0.0
    return mat


def calibrate_thresholds_per_cat(
    y_cal_5: np.ndarray,
    probs_cal: np.ndarray,
    src_cal: np.ndarray,
    ids_tuple: tuple[int, ...],
) -> np.ndarray:
    """Tune per-category threshold strictly on calibration split."""
    grid = np.arange(0.20, 0.81, 0.025)
    ths = []
    for j, cid in enumerate(ids_tuple):
        mask = src_cal != cid
        best_th = float(max(grid, key=lambda t: f1_score(y_cal_5[mask, j], probs_cal[mask, j] >= t, zero_division=0)))
        ths.append(round(best_th, 4))
    return np.asarray(ths, dtype=float)


def bootstrap_f1_delta(
    y_true: np.ndarray,
    yp_base: np.ndarray,
    yp_cand: np.ndarray,
    sources: np.ndarray,
    ids_tuple: tuple[int, ...],
    n_boot: int = 1000,
) -> dict:
    """Ad-level paired bootstrap 95% CI for Micro-F1 delta (candidate - baseline)."""
    rng = np.random.default_rng(SEED)
    n = len(y_true)
    active = np.ones_like(y_true, dtype=bool)
    for i, src in enumerate(sources):
        if int(src) in ids_tuple:
            active[i, ids_tuple.index(int(src))] = False

    tp_b = ((yp_base == 1) & (y_true == 1) & active).sum(axis=1)
    fp_b = ((yp_base == 1) & (y_true == 0) & active).sum(axis=1)
    fn_b = ((yp_base == 0) & (y_true == 1) & active).sum(axis=1)

    tp_c = ((yp_cand == 1) & (y_true == 1) & active).sum(axis=1)
    fp_c = ((yp_cand == 1) & (y_true == 0) & active).sum(axis=1)
    fn_c = ((yp_cand == 0) & (y_true == 1) & active).sum(axis=1)

    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1_b = 2 * tp_b[idx].sum() / max(1, 2 * tp_b[idx].sum() + fp_b[idx].sum() + fn_b[idx].sum())
        f1_c = 2 * tp_c[idx].sum() / max(1, 2 * tp_c[idx].sum() + fp_c[idx].sum() + fn_c[idx].sum())
        deltas[b] = f1_c - f1_b
    return {
        "delta_mean": round(float(deltas.mean()), 6),
        "ci95_low": round(float(np.percentile(deltas, 2.5)), 6),
        "ci95_high": round(float(np.percentile(deltas, 97.5)), 6),
        "p_positive": round(float(np.mean(deltas > 0)), 4),
    }


def error_transition_counts(
    y_true: np.ndarray,
    yp_base: np.ndarray,
    yp_cand: np.ndarray,
    sources: np.ndarray,
    ids_tuple: tuple[int, ...],
) -> dict:
    """Count exact fixed/broken error transitions on active (ad, cat) pairs."""
    both_ok = fixed = broken = both_err = 0
    fn_fixed = fp_fixed = fn_broken = fp_broken = 0
    for i in range(len(y_true)):
        for j, cid in enumerate(ids_tuple):
            if int(sources[i]) == cid:
                continue
            yt, pb, pc = int(y_true[i, j]), int(yp_base[i, j]), int(yp_cand[i, j])
            if pb == yt and pc == yt:
                both_ok += 1
            elif pb != yt and pc == yt:
                fixed += 1
                if yt == 1:
                    fn_fixed += 1
                else:
                    fp_fixed += 1
            elif pb == yt and pc != yt:
                broken += 1
                if yt == 0:
                    fp_broken += 1
                else:
                    fn_broken += 1
            else:
                both_err += 1
    return {
        "both_correct": both_ok,
        "base_wrong_cand_right_fixed": fixed,
        "base_right_cand_wrong_broken": broken,
        "both_wrong": both_err,
        "net_gain": fixed - broken,
        "fixed_breakdown": {"fn_to_tp": fn_fixed, "fp_to_tn": fp_fixed},
        "broken_breakdown": {"tp_to_fn": fn_broken, "tn_to_fp": fp_broken},
    }


def compute_extended_slices(sname: str, rows: list[dict], n_chunks_arr: np.ndarray) -> dict[str, np.ndarray]:
    base = build_split_slices(sname, rows, n_chunks_arr)
    descs = [r["description"] for r in rows]
    base["third_party_or_contractor"] = np.asarray([bool(cae.CONTRACTOR_RE.search(d)) for d in descs], dtype=bool)
    base["double_neg_or_not_only"] = np.asarray(
        [bool(re.search(r"не\s+только|больше\s+не\s+отказ|не\s+отказываем", d, re.I)) for d in descs],
        dtype=bool,
    )
    base["price_negation"] = np.asarray([bool(re.search(r"бесплатно|по\s+смете|за\s+счёт", d, re.I)) for d in descs], dtype=bool)
    if sname == "pub_validation":
        base["past_or_refusal"] = np.asarray([bool(PAST_OR_REFUSAL_RE.search(d)) for d in descs], dtype=bool)
    return base


def build_pair_dataframe(
    split_name: str,
    rows: list[dict],
    uids: np.ndarray,
    offsets: np.ndarray,
    cindices: np.ndarray,
    chunk_cat_sims: np.ndarray,
    uniq_chunks: list[str],
    y_eval_5: np.ndarray,
    src_arr: np.ndarray,
    probs_sgd_5: np.ndarray,
    chunk_sig_cache: dict[tuple[str, int], dict],
    chunk_global_cache: dict[str, tuple[int, int, list[str]]],
    dup_groups: np.ndarray | None = None,
) -> pd.DataFrame:
    records = []
    for r_idx, row in enumerate(rows):
        uid = int(uids[r_idx])
        src = int(src_arr[r_idx])
        start, end = int(offsets[uid]), int(offsets[uid + 1])
        ad_cids = cindices[start:end]
        full_chunks = [uniq_chunks[int(c)] for c in ad_cids]
        sims_mat = chunk_cat_sims[ad_cids]
        ad_len = len(row["description"])
        dgrp = int(dup_groups[r_idx]) if dup_groups is not None else -1

        for j_cat, cid in enumerate(cae.SCOPE_102_106):
            if src == cid:
                continue
            col_101 = cae.ALL_IDS.index(cid)
            k_take = min(cae.TOP_K, len(ad_cids))
            order = np.argsort(-sims_mat[:, col_101], kind="stable")[:k_take]
            top3_pos = order.tolist()
            top3_sims = [float(sims_mat[p, col_101]) for p in top3_pos]
            mentioned_anywhere = int(any(chunk_mentions_category(ch, cid) for ch in full_chunks))

            feat = extract_pair_features(
                analyzer=None,
                full_chunks=full_chunks,
                top3_positions=top3_pos,
                top3_sims=top3_sims,
                cat_id=cid,
                ad_char_len=ad_len,
                include_debug=False,
                precomputed_chunk_signals=chunk_sig_cache,
                precomputed_chunk_globals=chunk_global_cache,
                precomputed_mentioned_anywhere=mentioned_anywhere,
            )
            feat["split"] = split_name
            feat["row_idx"] = r_idx
            feat["uid"] = uid
            feat["item_id"] = str(row.get("item_id", f"{split_name}_{r_idx}"))
            feat["source_mcId"] = src
            feat["y_true"] = int(y_eval_5[r_idx, j_cat])
            feat["p_sgd"] = float(probs_sgd_5[r_idx, j_cat])
            feat["dup_group_id"] = dgrp
            feat["top3_positions"] = "|".join(map(str, top3_pos))
            feat["top3_cids"] = "|".join(str(int(ad_cids[p])) for p in top3_pos)
            feat["construction"] = str(row.get("construction") or "")
            records.append(feat)
    return pd.DataFrame.from_records(records)


def main() -> None:
    t_total = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Loading dataset splits & Qwen cache (read-only)...")
    splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }
    for k, v in splits.items():
        print(f"   {k:16s}: {len(v)} ads")

    npz = np.load(CACHE_NPZ)
    chunk_emb_f32 = npz["chunk_emb"].astype(np.float32)
    query_emb_f32 = npz["query_emb"].astype(np.float32)
    offsets = npz["ad_chunk_offsets"]
    cindices = npz["ad_chunk_indices"]
    raw_sims = chunk_emb_f32 @ query_emb_f32.T
    chunk_cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    # Reconstruct exact uniq_chunks lookup matching output/cat-aware-v2/cache/chunks_embeddings.npz
    uniq_ads: list[str] = []
    ad_to_uid: dict[str, int] = {}
    for rows in splits.values():
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
    assert len(uniq_chunks) == len(chunk_emb_f32), f"Cache mismatch: {len(uniq_chunks)} vs {len(chunk_emb_f32)}"
    print(f"   Verified chunks_embeddings.npz alignment: {len(uniq_chunks)} unique chunks.")

    split_feats = {}
    split_nchunks = {}
    split_uids = {}
    for sname in splits:
        uids = npz[f"uid_{sname}"]
        split_uids[sname] = uids
        feats, _ = cae.build_category_features_for_uids(
            uids, offsets, cindices, chunk_emb_f32, chunk_cat_sims, return_selection=False
        )
        split_feats[sname] = feats
        split_nchunks[sname] = np.asarray([int(offsets[u + 1] - offsets[u]) for u in uids], dtype=np.int32)

    tr_names = ["pub_train", "v3_train"]
    cal_names = ["pub_calibration", "v3_validation"]
    tr_rows = [r for s in tr_names for r in splits[s]]
    cal_rows = [r for s in cal_names for r in splits[s]]

    y_tr_5, src_tr = cae._rows_xy_for_ids(tr_rows, cae.SCOPE_102_106, use_eval=False)
    y_tr_eval_5, _ = cae._rows_xy_for_ids(tr_rows, cae.SCOPE_102_106, use_eval=True)
    y_cal_5, src_cal = cae._rows_xy_for_ids(cal_rows, cae.SCOPE_102_106, use_eval=True)
    y_pub_val_5, src_pub_val = cae._rows_xy_for_ids(splits["pub_validation"], cae.SCOPE_102_106, use_eval=True)
    y_v3_test_5, src_v3_test = cae._rows_xy_for_ids(splits["v3_test"], cae.SCOPE_102_106, use_eval=True)

    print("2. Computing near-duplicate groups on train (TF-IDF >= 0.85 connected components)...")
    dup_groups_tr, dup_diag = compute_dup_groups([r["description"] for r in tr_rows], sim_threshold=0.85)
    print(f"   {dup_diag}")

    print("3. Reproducing Baseline 1 (Variant A: Category-Aware Qwen + Fixed Segmentation + SGD)...")
    t_sgd0 = time.perf_counter()
    sgd_heads = {}
    sgd_head_diag = {}
    X_tr_by_cat = {}
    X_cal_by_cat = {}
    for j, cid in enumerate(cae.SCOPE_102_106):
        X_tr_c = np.vstack([split_feats[s][cid] for s in tr_names])
        X_cal_c = np.vstack([split_feats[s][cid] for s in cal_names])
        X_tr_by_cat[cid] = X_tr_c
        X_cal_by_cat[cid] = X_cal_c
        clf_c, best_alpha, tr_f1, alpha_diag = cae._fit_cv_binary_head(X_tr_c, y_tr_5[:, j])
        sgd_heads[cid] = clf_c
        sgd_head_diag[str(cid)] = {"alpha": best_alpha, "train_f1_05": round(tr_f1, 6), "cv_oof_f1": alpha_diag[str(best_alpha)]["oof_f1"]}
    sgd_fit_sec = time.perf_counter() - t_sgd0

    t_sgd_inf0 = time.perf_counter()
    probs_A = {}
    for sname in ("pub_validation", "v3_test", "pub_calibration", "v3_validation"):
        probs_A[sname] = np.column_stack([sgd_heads[cid].predict_proba(split_feats[sname][cid])[:, 1] for cid in cae.SCOPE_102_106])
    sgd_eval_inf_sec = time.perf_counter() - t_sgd_inf0
    probs_A_cal = np.vstack([probs_A["pub_calibration"], probs_A["v3_validation"]])
    joblib.dump({"heads": sgd_heads, "ids": cae.SCOPE_102_106, "diag": sgd_head_diag}, MODELS_DIR / "sgd_full_heads.joblib")

    th05 = np.full(5, 0.5)
    mA_pub05, ypA_pub05 = compute_detailed_metrics(y_pub_val_5, probs_A["pub_validation"], th05, src_pub_val, cae.SCOPE_102_106)
    mA_v305, ypA_v305 = compute_detailed_metrics(y_v3_test_5, probs_A["v3_test"], th05, src_v3_test, cae.SCOPE_102_106)
    print(f"   Baseline A reproduced (thr=0.5): pub_validation Micro-F1={mA_pub05['micro_f1']:.4f} (ref 0.9314), v3_test Micro-F1={mA_v305['micro_f1']:.4f} (ref 0.8069)")
    assert abs(mA_pub05["micro_f1"] - 0.9314) < 0.002 and abs(mA_v305["micro_f1"] - 0.8069) < 0.002, (
        f"Baseline A did not reproduce! pub={mA_pub05['micro_f1']}, v3={mA_v305['micro_f1']}"
    )

    print("4. Generating strictly leakage-free StratifiedGroupKFold OOF SGD probabilities on train...")
    t_oof0 = time.perf_counter()
    p_sgd_oof_tr, oof_alphas = fit_leakage_free_oof_sgd(X_tr_by_cat, y_tr_5, dup_groups_tr)
    oof_sec = time.perf_counter() - t_oof0
    mA_oof_tr, _ = compute_detailed_metrics(y_tr_eval_5, p_sgd_oof_tr, th05, src_tr, cae.SCOPE_102_106)
    print(f"   OOF Train SGD Micro-F1={mA_oof_tr['micro_f1']:.4f} in {oof_sec:.1f}s")

    print("5. Training & evaluating Baseline 2: TF-IDF word (1,3) + char_wb (3,5) + SGD...")
    t_tf0 = time.perf_counter()
    tfidf_clf = build_tfidf_pipeline()
    tr_texts = [r["description"] for r in tr_rows]
    tfidf_clf.fit(tr_texts, y_tr_5)
    tfidf_fit_sec = time.perf_counter() - t_tf0
    joblib.dump(tfidf_clf, MODELS_DIR / "tfidf_baseline.joblib")

    probs_tf_cal = tfidf_clf.predict_proba([r["description"] for r in cal_rows])
    th_tf_cal = calibrate_thresholds_per_cat(y_cal_5, probs_tf_cal, src_cal, cae.SCOPE_102_106)
    probs_tf_pub = tfidf_clf.predict_proba([r["description"] for r in splits["pub_validation"]])
    probs_tf_v3 = tfidf_clf.predict_proba([r["description"] for r in splits["v3_test"]])
    mTF_pub05, ypTF_pub05 = compute_detailed_metrics(y_pub_val_5, probs_tf_pub, th05, src_pub_val, cae.SCOPE_102_106)
    mTF_v305, ypTF_v305 = compute_detailed_metrics(y_v3_test_5, probs_tf_v3, th05, src_v3_test, cae.SCOPE_102_106)
    mTF_pub_cal, _ = compute_detailed_metrics(y_pub_val_5, probs_tf_pub, th_tf_cal, src_pub_val, cae.SCOPE_102_106)
    mTF_v3_cal, _ = compute_detailed_metrics(y_v3_test_5, probs_tf_v3, th_tf_cal, src_v3_test, cae.SCOPE_102_106)
    print(f"   TF-IDF Baseline (thr=0.5): pub_validation Micro-F1={mTF_pub05['micro_f1']:.4f}, v3_test Micro-F1={mTF_v305['micro_f1']:.4f}")

    print("6. Collecting unique Top-3 chunks & global candidates across splits for Natasha UD parsing...")
    needed_chunk_cat_keys: set[tuple[str, int]] = set()
    needed_global_chunks: set[str] = set()
    eval_unique_chunks: set[str] = set()

    for sname, rows in splits.items():
        uids = split_uids[sname]
        for r_idx, uid in enumerate(uids):
            start, end = int(offsets[int(uid)]), int(offsets[int(uid) + 1])
            ad_cids = cindices[start:end]
            sims_mat = chunk_cat_sims[ad_cids]
            src = int(rows[r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch_txt = uniq_chunks[int(c_int)]
                if sname in ("pub_validation", "v3_test"):
                    eval_unique_chunks.add(ch_txt)
                if GLOBAL_CANDIDATE_RE.search(ch_txt):
                    needed_global_chunks.add(ch_txt)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col_101 = cae.ALL_IDS.index(cid)
                order = np.argsort(-sims_mat[:, col_101], kind="stable")[:k_take]
                for p in order:
                    needed_chunk_cat_keys.add((uniq_chunks[int(ad_cids[p])], cid))

    print(f"   Unique (chunk, cat) pairs to extract: {len(needed_chunk_cat_keys)}; global candidate chunks: {len(needed_global_chunks)}")
    syntax_cache_file = EXP_CACHE_DIR / "chunk_syntax_cache.pkl"
    if syntax_cache_file.exists():
        with syntax_cache_file.open("rb") as f:
            cached_syn = pickle.load(f)
        chunk_sig_cache = cached_syn["chunk_sig_cache"]
        chunk_global_cache = cached_syn["chunk_global_cache"]
        parse_timing = cached_syn["parse_timing"]
        print(f"   Loaded cached Natasha chunk parses from {syntax_cache_file.name}.")
    else:
        analyzer = SyntaxAnalyzer()
        unique_chunks_to_parse = sorted({ch for (ch, _) in needed_chunk_cat_keys} | needed_global_chunks)
        t_p0 = time.perf_counter()
        for idx_ch, ch in enumerate(unique_chunks_to_parse):
            analyzer.parse_text(ch)
            if (idx_ch + 1) % 5000 == 0:
                print(f"     parsed {idx_ch + 1}/{len(unique_chunks_to_parse)} unique chunks ({time.perf_counter() - t_p0:.1f}s)...")
        parse_only_sec = time.perf_counter() - t_p0

        # Measure exact uncached CPU cost on eval splits separately
        eval_sa = SyntaxAnalyzer()
        t_eval_p0 = time.perf_counter()
        for ch in sorted(eval_unique_chunks):
            eval_sa.parse_text(ch)
        eval_parse_sec = time.perf_counter() - t_eval_p0

        t_ext0 = time.perf_counter()
        chunk_sig_cache = {}
        for ch, cid in needed_chunk_cat_keys:
            chunk_sig_cache[(ch, cid)] = extract_chunk_syntax_signals(analyzer, ch, cid)
        chunk_global_cache = {}
        for ch in needed_global_chunks:
            chunk_global_cache[ch] = detect_chunk_global_restriction(analyzer, ch)
        ext_sec = time.perf_counter() - t_ext0

        parse_timing = {
            "unique_chunks_parsed": len(unique_chunks_to_parse),
            "natasha_parse_all_sec": round(parse_only_sec, 3),
            "feature_extract_unique_sec": round(ext_sec, 3),
            "eval_unique_chunks": len(eval_unique_chunks),
            "eval_natasha_parse_sec": round(eval_parse_sec, 3),
            "eval_ads_count": len(splits["pub_validation"]) + len(splits["v3_test"]),
            "ms_per_eval_chunk": round(1000.0 * eval_parse_sec / max(1, len(eval_unique_chunks)), 3),
            "ms_per_eval_ad_parse": round(1000.0 * eval_parse_sec / (len(splits["pub_validation"]) + len(splits["v3_test"])), 3),
        }
        with syntax_cache_file.open("wb") as f:
            pickle.dump(
                {"chunk_sig_cache": chunk_sig_cache, "chunk_global_cache": chunk_global_cache, "parse_timing": parse_timing},
                f,
            )
    print(f"   Natasha timing: {parse_timing}")

    print("7. Assembling (ad, cat) DataFrames for Train (OOF), Calibration, pub_validation, v3_test...")
    t_df0 = time.perf_counter()
    # Map dup_group_id cleanly to pub_train + v3_train slice
    n_pub_tr = len(splits["pub_train"])
    df_tr_pub = build_pair_dataframe(
        "pub_train", splits["pub_train"], split_uids["pub_train"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_tr_eval_5[:n_pub_tr], src_tr[:n_pub_tr], p_sgd_oof_tr[:n_pub_tr],
        chunk_sig_cache, chunk_global_cache, dup_groups=dup_groups_tr[:n_pub_tr],
    )
    df_tr_v3 = build_pair_dataframe(
        "v3_train", splits["v3_train"], split_uids["v3_train"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_tr_eval_5[n_pub_tr:], src_tr[n_pub_tr:], p_sgd_oof_tr[n_pub_tr:],
        chunk_sig_cache, chunk_global_cache, dup_groups=dup_groups_tr[n_pub_tr:],
    )
    df_tr = pd.concat([df_tr_pub, df_tr_v3], ignore_index=True)

    n_pub_cal = len(splits["pub_calibration"])
    df_cal_pub = build_pair_dataframe(
        "pub_calibration", splits["pub_calibration"], split_uids["pub_calibration"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A["pub_calibration"], chunk_sig_cache, chunk_global_cache,
    )
    df_cal_v3 = build_pair_dataframe(
        "v3_validation", splits["v3_validation"], split_uids["v3_validation"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A["v3_validation"], chunk_sig_cache, chunk_global_cache,
    )
    df_cal = pd.concat([df_cal_pub, df_cal_v3], ignore_index=True)

    df_pub_val = build_pair_dataframe(
        "pub_validation", splits["pub_validation"], split_uids["pub_validation"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_pub_val_5, src_pub_val, probs_A["pub_validation"], chunk_sig_cache, chunk_global_cache,
    )
    df_v3_test = build_pair_dataframe(
        "v3_test", splits["v3_test"], split_uids["v3_test"], offsets, cindices, chunk_cat_sims, uniq_chunks,
        y_v3_test_5, src_v3_test, probs_A["v3_test"], chunk_sig_cache, chunk_global_cache,
    )
    pair_assembly_sec = time.perf_counter() - t_df0
    print(
        f"   Pairs built in {pair_assembly_sec:.2f}s: train={len(df_tr)}, cal={len(df_cal)}, "
        f"pub_val={len(df_pub_val)}, v3_test={len(df_v3_test)}"
    )

    # Assert zero NaN / Inf across numeric columns and p_sgd
    all_feat_cols = ["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES
    num_cols = [c for c in all_feat_cols if c not in SYNTAX_CAT_FEATURES]
    for name_df, df_x in [("train", df_tr), ("cal", df_cal), ("pub_val", df_pub_val), ("v3_test", df_v3_test)]:
        assert not df_x[all_feat_cols].isna().any().any(), f"NaN detected in {name_df}!"
        vals = df_x[num_cols].to_numpy(dtype=np.float64)
        assert np.all(np.isfinite(vals)), f"Inf detected in {name_df}!"

    # Save syntax_features.csv for pub_validation + v3_test
    df_eval_all = pd.concat([df_pub_val, df_v3_test], ignore_index=True)
    df_eval_all.to_csv(OUT_DIR / "syntax_features.csv", index=False)

    print("8. Training CatBoost B (meta control), C (+ syntax), and ablations...")
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

    trained_models = {}
    model_meta = {}
    eval_probs_by_variant = {"pub_validation": {}, "v3_test": {}, "cal": {}}

    for vname, (fcols, cat_cols) in variants_spec.items():
        tr_pool = Pool(df_tr[fcols], df_tr["y_true"], cat_features=cat_cols)
        cal_pool = Pool(df_cal[fcols], df_cal["y_true"], cat_features=cat_cols)
        pub_pool = Pool(df_pub_val[fcols], cat_features=cat_cols)
        v3_pool = Pool(df_v3_test[fcols], cat_features=cat_cols)

        t_m0 = time.perf_counter()
        cb = CatBoostClassifier(**CATBOOST_PARAMS)
        cb.fit(tr_pool, eval_set=cal_pool, use_best_model=True)
        fit_s = time.perf_counter() - t_m0
        cb.save_model(str(MODELS_DIR / f"catboost_{vname}.cbm"))
        trained_models[vname] = cb

        t_inf0 = time.perf_counter()
        p_pub = cb.predict_proba(pub_pool)[:, 1]
        p_v3 = cb.predict_proba(v3_pool)[:, 1]
        inf_s = time.perf_counter() - t_inf0
        p_cal = cb.predict_proba(cal_pool)[:, 1]

        mat_cal = probs_to_eval_matrix(df_cal, p_cal, len(cal_rows), src_cal, cae.SCOPE_102_106)
        mat_pub = probs_to_eval_matrix(df_pub_val, p_pub, len(splits["pub_validation"]), src_pub_val, cae.SCOPE_102_106)
        mat_v3 = probs_to_eval_matrix(df_v3_test, p_v3, len(splits["v3_test"]), src_v3_test, cae.SCOPE_102_106)
        eval_probs_by_variant["cal"][vname] = mat_cal
        eval_probs_by_variant["pub_validation"][vname] = mat_pub
        eval_probs_by_variant["v3_test"][vname] = mat_v3

        p_tr_cb = cb.predict_proba(tr_pool)[:, 1]
        imp_pvc = cb.get_feature_importance(cal_pool, type="PredictionValuesChange")
        imp_lfc = cb.get_feature_importance(cal_pool, type="LossFunctionChange")
        model_meta[vname] = {
            "n_features": len(fcols),
            "best_iteration": int(cb.get_best_iteration()),
            "tree_count": int(cb.tree_count_),
            "fit_seconds": round(fit_s, 3),
            "eval_predict_seconds": round(inf_s, 4),
            "train_logloss": round(float(log_loss(df_tr["y_true"], p_tr_cb)), 6),
            "cal_logloss": round(float(log_loss(df_cal["y_true"], p_cal)), 6),
            "feature_importance_pvc_cal": [
                {"feature": fc, "importance": round(float(iv), 4)}
                for fc, iv in sorted(zip(fcols, imp_pvc), key=lambda x: -x[1])
            ],
            "feature_importance_lfc_cal": [
                {"feature": fc, "importance": round(float(iv), 6)}
                for fc, iv in sorted(zip(fcols, imp_lfc), key=lambda x: -x[1])
            ],
        }
        print(
            f"   [{vname:21s}] trees={cb.tree_count_:3d} tr_loss={model_meta[vname]['train_logloss']:.4f} "
            f"cal_loss={model_meta[vname]['cal_logloss']:.4f} fit={fit_s:.2f}s"
        )

    print("9. Evaluating all variants cleanly at thr=0.5 (and calibration-tuned thresholds separately)...")
    thA_cal = calibrate_thresholds_per_cat(y_cal_5, probs_A_cal, src_cal, cae.SCOPE_102_106)
    variants_all_probs = {
        "tfidf_sgd": {"pub_validation": probs_tf_pub, "v3_test": probs_tf_v3, "cal": probs_tf_cal, "cal_th": th_tf_cal},
        "A_qwen_sgd": {"pub_validation": probs_A["pub_validation"], "v3_test": probs_A["v3_test"], "cal": probs_A_cal, "cal_th": thA_cal},
    }
    for vname in variants_spec:
        th_v_cal = calibrate_thresholds_per_cat(y_cal_5, eval_probs_by_variant["cal"][vname], src_cal, cae.SCOPE_102_106)
        variants_all_probs[vname] = {
            "pub_validation": eval_probs_by_variant["pub_validation"][vname],
            "v3_test": eval_probs_by_variant["v3_test"][vname],
            "cal": eval_probs_by_variant["cal"][vname],
            "cal_th": th_v_cal,
        }

    metrics_out = {
        "dup_groups_train": dup_diag,
        "sgd_head_diag": sgd_head_diag,
        "oof_train_sgd_05": mA_oof_tr,
        "oof_alphas": {str(k): v for k, v in oof_alphas.items()},
        "timing": {
            "sgd_fit_seconds": round(sgd_fit_sec, 3),
            "sgd_eval_infer_seconds": round(sgd_eval_inf_sec, 4),
            "oof_sgd_seconds": round(oof_sec, 2),
            "tfidf_fit_seconds": round(tfidf_fit_sec, 2),
            "pair_assembly_seconds": round(pair_assembly_sec, 3),
            **parse_timing,
        },
        "catboost_models": model_meta,
        "splits": {},
        "transitions_and_bootstrap": {},
    }

    preds_cache = {}
    for sname, y_s, src_s in [
        ("pub_validation", y_pub_val_5, src_pub_val),
        ("v3_test", y_v3_test_5, src_v3_test),
    ]:
        metrics_out["splits"][sname] = {"thr_05": {}, "thr_cal": {}, "slices_05": {}}
        preds_cache[sname] = {}
        slices = compute_extended_slices(sname, splits[sname], split_nchunks[sname])

        for vname, pdict in variants_all_probs.items():
            pr = pdict[sname]
            m05, yp05 = compute_detailed_metrics(y_s, pr, th05, src_s, cae.SCOPE_102_106)
            mcal, ypcal = compute_detailed_metrics(y_s, pr, pdict["cal_th"], src_s, cae.SCOPE_102_106)
            metrics_out["splits"][sname]["thr_05"][vname] = m05
            metrics_out["splits"][sname]["thr_cal"][vname] = {"thresholds": pdict["cal_th"].tolist(), **mcal}
            preds_cache[sname][vname] = (pr, yp05)

        for sl_name, mk in slices.items():
            if int(mk.sum()) == 0:
                continue
            sl_block = {"rows": int(mk.sum())}
            for vname in ("tfidf_sgd", "A_qwen_sgd", "B_meta", "C_syntax", "C_no_global"):
                pr = variants_all_probs[vname][sname]
                ms, _ = compute_detailed_metrics(y_s[mk], pr[mk], th05, src_s[mk], cae.SCOPE_102_106)
                sl_block[vname] = ms
            metrics_out["splits"][sname]["slices_05"][sl_name] = sl_block

        ypA = preds_cache[sname]["A_qwen_sgd"][1]
        ypB = preds_cache[sname]["B_meta"][1]
        ypC = preds_cache[sname]["C_syntax"][1]
        metrics_out["transitions_and_bootstrap"][sname] = {
            "A_to_B": error_transition_counts(y_s, ypA, ypB, src_s, cae.SCOPE_102_106),
            "A_to_C": error_transition_counts(y_s, ypA, ypC, src_s, cae.SCOPE_102_106),
            "B_to_C": error_transition_counts(y_s, ypB, ypC, src_s, cae.SCOPE_102_106),
            "boot_C_minus_A": bootstrap_f1_delta(y_s, ypA, ypC, src_s, cae.SCOPE_102_106),
            "boot_C_minus_B": bootstrap_f1_delta(y_s, ypB, ypC, src_s, cae.SCOPE_102_106),
            "boot_B_minus_A": bootstrap_f1_delta(y_s, ypA, ypB, src_s, cae.SCOPE_102_106),
        }

    # Save predictions.csv aligned row-for-row with df_eval_all (pub_validation + v3_test active pairs)
    cat_to_col = {cid: j for j, cid in enumerate(cae.SCOPE_102_106)}
    for vcol, vkey in [
        ("tfidf", "tfidf_sgd"),
        ("A", "A_qwen_sgd"),
        ("B", "B_meta"),
        ("C", "C_syntax"),
        ("B0", "B0_score_cat"),
        ("C_no_global", "C_no_global"),
        ("C_syntax_only", "C_syntax_only_no_meta"),
    ]:
        p_list = []
        y_list = []
        for _, r in df_eval_all.iterrows():
            sp = r["split"]
            ri = int(r["row_idx"])
            cj = cat_to_col[int(r["cat_id"])]
            pr_mat, yp_mat = preds_cache[sp][vkey]
            p_list.append(round(float(pr_mat[ri, cj]), 6))
            y_list.append(int(yp_mat[ri, cj]))
        df_eval_all[f"p_{vcol}"] = p_list
        df_eval_all[f"pred_{vcol}_05"] = y_list

    pred_cols = [
        "split", "row_idx", "uid", "item_id", "source_mcId", "cat_id", "construction", "y_true",
        "p_tfidf", "pred_tfidf_05", "p_A", "pred_A_05", "p_B", "pred_B_05", "p_C", "pred_C_05",
        "p_B0", "pred_B0_05", "p_C_no_global", "pred_C_no_global_05",
        "best_chunk_polarity", "dominant_subject", "temporal_state", "service_link_status",
        "syn_has_neg_on_action", "syn_has_not_only", "syn_double_neg_cancel", "syn_affirmative_offer",
        "syn_third_party_actor", "syn_seller_actor", "syn_price_condition_only",
        "syn_global_turnkey_only", "syn_global_partial_refusal", "syn_global_outside_top3",
        "syn_has_conflicting_claims", "top3_positions", "top3_cids",
    ]
    df_eval_all[pred_cols].to_csv(OUT_DIR / "predictions.csv", index=False)

    metrics_out["timing"]["total_script_seconds"] = round(time.perf_counter() - t_total, 2)
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cfg = {
        "mode": "public_plus_v3",
        "seed": SEED,
        "train_ads": len(tr_rows),
        "calibration_ads": len(cal_rows),
        "pub_validation_ads": len(splits["pub_validation"]),
        "v3_test_ads": len(splits["v3_test"]),
        "catboost_params": CATBOOST_PARAMS,
        "non_syntax_meta_features": NON_SYNTAX_META_FEATURES,
        "syntax_only_features": SYNTAX_ONLY_FEATURES,
        "syntax_cat_features": SYNTAX_CAT_FEATURES,
        "tfidf_params": {"word_ngram": [1, 3], "word_min_df": 2, "word_max_features": 18000, "char_wb_ngram": [3, 5], "char_min_df": 3, "char_max_features": 28000, "sgd_alpha": 0.0001},
        "sgd_params": cae.SGD_PARAMS,
        "alpha_grid": cae.ALPHA_GRID,
    }
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== SUMMARY (thr = 0.5) ===")
    for sname in ("pub_validation", "v3_test"):
        print(f"\nSplit: {sname}")
        for vname in ("tfidf_sgd", "A_qwen_sgd", "B0_score_cat", "B_meta", "C_syntax", "C_no_global", "C_syntax_only_no_meta"):
            m = metrics_out["splits"][sname]["thr_05"][vname]
            c = m["confusion"]
            print(
                f"  {vname:22s}: MicroF1={m['micro_f1']:.4f} MacroF1={m['macro_f1']:.4f} "
                f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} "
                f"FP={c['fp']:3d} FN={c['fn']:3d} Exact={m['exact_match']:.4f} AP={m['micro_ap']:.4f}"
            )
    print(f"\nDone in {time.perf_counter() - t_total:.1f}s -> {OUT_DIR}")


if __name__ == "__main__":
    main()
