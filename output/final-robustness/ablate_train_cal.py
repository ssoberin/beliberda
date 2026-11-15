#!/usr/bin/env python3
"""Train & compare isolated robustness candidates (H3_only vs H1+H2+H3) strictly on train + calibration + robustness_benchmark."""
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
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics
from avito_compare.scripts.eval_public_tests_syntax import reconstruct_v2_chunk_map
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.run_syntax_experiment import (
    CATBOOST_PARAMS,
    build_pair_dataframe,
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
from avito_compare.scripts.text_preprocess import normalize_ad_text, strip_prompt_injections
from robustness_benchmark import RobustPipelineRunner, build_cases, evaluate_suite

PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
V2_CACHE_NPZ = ROOT / "output" / "cat-aware-v2" / "cache" / "chunks_embeddings.npz"
EXP_MODELS_DIR = ROOT / "output" / "syntax-aware-experiment" / "models"
OPT_DIR = ROOT / "output" / "catboost-optimization"
OUT_DIR = ROOT / "output" / "final-robustness"
OUT_CACHE = OUT_DIR / "cache"
OUT_MODELS = OUT_DIR / "models"


def score_cal(df_cal: pd.DataFrame, model: CatBoostClassifier, cols: list[str], cal_rows: list[dict],
              n_pub_cal: int, y_cal_5: np.ndarray, src_cal: np.ndarray) -> tuple[dict, np.ndarray]:
    probs = model.predict_proba(Pool(df_cal[cols], cat_features=SYNTAX_CAT_FEATURES))[:, 1]
    if "split" in df_cal.columns and set(df_cal["split"].unique()) == {"pub_calibration", "v3_validation"}:
        m_pub = df_cal["split"].to_numpy() == "pub_calibration"
        m_v3 = df_cal["split"].to_numpy() == "v3_validation"
        p_pub = probs_to_eval_matrix(df_cal[m_pub], probs[m_pub], n_pub_cal, src_cal[:n_pub_cal], cae.SCOPE_102_106)
        p_v3 = probs_to_eval_matrix(df_cal[m_v3], probs[m_v3], len(cal_rows) - n_pub_cal, src_cal[n_pub_cal:], cae.SCOPE_102_106)
        p_mat = np.vstack([p_pub, p_v3])
    else:
        p_mat = probs_to_eval_matrix(df_cal, probs, len(cal_rows), src_cal, cae.SCOPE_102_106)
    m, yp = compute_detailed_metrics(y_cal_5, p_mat, np.full(5, 0.5), src_cal, cae.SCOPE_102_106)
    return m, yp


def main() -> None:
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    print("1. Loading train + calibration data & frozen Qwen/SGD caches...", flush=True)
    v2_chunks, v2_chunk_emb, query_emb_f32 = reconstruct_v2_chunk_map()
    npz = np.load(V2_CACHE_NPZ)
    v2_emb_f32 = v2_chunk_emb.astype(np.float32)
    v2_raw_sims = v2_emb_f32 @ query_emb_f32.T
    v2_cat_sims = np.column_stack([v2_raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
    offsets_v2 = npz["ad_chunk_offsets"]
    cindices_v2 = npz["ad_chunk_indices"]

    splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
    }
    tr_rows = splits["pub_train"] + splits["v3_train"]
    cal_rows = splits["pub_calibration"] + splits["v3_validation"]
    n_pub_tr = len(splits["pub_train"])
    n_pub_cal = len(splits["pub_calibration"])

    y_tr_5, src_tr = cae._rows_xy_for_ids(tr_rows, cae.SCOPE_102_106, use_eval=True)
    y_cal_5, src_cal = cae._rows_xy_for_ids(cal_rows, cae.SCOPE_102_106, use_eval=True)

    sgd_heads = joblib.load(EXP_MODELS_DIR / "sgd_full_heads.joblib")["heads"]
    oof_data = np.load(OPT_DIR / "cache" / "original_oof.npz")
    p_sgd_oof_tr = oof_data["oof"]
    dup_groups_tr = oof_data["groups"]

    probs_A_cal = {}
    for sname in ("pub_calibration", "v3_validation"):
        feats, _ = cae.build_category_features_for_uids(
            npz[f"uid_{sname}"], offsets_v2, cindices_v2, v2_emb_f32, v2_cat_sims, return_selection=False
        )
        probs_A_cal[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    # Load existing baseline C_syntax_fixed cache for quick diff
    with (ROOT / "output" / "syntax-link-fix" / "cache" / "chunk_syntax_cache_fixed.pkl").open("rb") as f:
        base_syn = pickle.load(f)

    # Collect raw train+cal needed chunk keys for H3
    raw_keys: set[tuple[str, int]] = set()
    raw_glob: set[str] = set()
    for sname in ("pub_train", "v3_train", "pub_calibration", "v3_validation"):
        rows = splits[sname]
        for r_idx, uid in enumerate(npz[f"uid_{sname}"]):
            st, en = int(offsets_v2[int(uid)]), int(offsets_v2[int(uid) + 1])
            ad_cids = cindices_v2[st:en]
            sims_mat = v2_cat_sims[ad_cids]
            src = int(rows[r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch = v2_chunks[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch):
                    raw_glob.add(ch)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col = cae.ALL_IDS.index(cid)
                for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                    raw_keys.add((v2_chunks[int(ad_cids[p])], cid))

    # Build H1+H2 preprocessed chunks for train & cal and encode ONLY delta chunks
    print("2. Preparing H1+H2 preprocessed chunks and encoding delta chunks...", flush=True)
    chunk_emb_map = {ch: v2_emb_f32[i] for i, ch in enumerate(v2_chunks)}
    norm_chunks_by_split: dict[str, list[list[str]]] = {}
    norm_rows_by_split: dict[str, list[dict]] = {}
    new_chunks: list[str] = []
    seen_new: set[str] = set()
    for sname in ("pub_train", "v3_train", "pub_calibration", "v3_validation"):
        norm_chunks_by_split[sname] = []
        norm_rows_by_split[sname] = []
        for r in splits[sname]:
            nd = strip_prompt_injections(normalize_ad_text(r["description"]))
            norm_rows_by_split[sname].append({**r, "description": nd})
            chs = chunk_ad_fixed(nd) or [""]
            norm_chunks_by_split[sname].append(chs)
            for c in chs:
                if c not in chunk_emb_map and c not in seen_new:
                    seen_new.add(c)
                    new_chunks.append(c)

    print(f"   New unique chunks to encode on Qwen for H1+H2 (train+cal): {len(new_chunks)}", flush=True)
    delta_cache_path = OUT_CACHE / "h12_train_cal_delta.npz"
    if delta_cache_path.exists():
        dz = np.load(delta_cache_path, allow_pickle=False)
        d_chs = dz["chunks"].tolist()
        d_embs = dz["embs"].astype(np.float32)
        for c, v in zip(d_chs, d_embs):
            chunk_emb_map[c] = v
    missing_chunks = [c for c in new_chunks if c not in chunk_emb_map]
    if missing_chunks:
        t_q0 = time.perf_counter()
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        local_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", local_files_only=True)
        kw = "processor_kwargs" if "processor_kwargs" in inspect.signature(SentenceTransformer.__init__).parameters else "tokenizer_kwargs"
        enc = SentenceTransformer(
            local_dir, device=device, local_files_only=True,
            model_kwargs={"dtype": torch.float16 if device == "cuda" else torch.float32},
            **{kw: {"padding_side": "left"}},
        )
        enc.max_seq_length = 1024
        vecs = np.asarray(
            enc.encode(missing_chunks, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float16,
        ).astype(np.float32)
        for c, v in zip(missing_chunks, vecs):
            chunk_emb_map[c] = v
        all_new_vecs = np.vstack([chunk_emb_map[c] for c in new_chunks]).astype(np.float16) if new_chunks else np.zeros((0, 1024), dtype=np.float16)
        np.savez(delta_cache_path, chunks=np.array(new_chunks), embs=all_new_vecs)
        print(f"   Encoded {len(missing_chunks)} delta chunks in {time.perf_counter()-t_q0:.2f}s ({device}).", flush=True)

    # Assemble index arrays for H1+H2 train & cal
    norm_all_chunks = list(chunk_emb_map.keys())
    norm_c2id = {c: i for i, c in enumerate(norm_all_chunks)}
    norm_emb_f32 = np.vstack([chunk_emb_map[c] for c in norm_all_chunks])
    norm_raw = norm_emb_f32 @ query_emb_f32.T
    norm_sims = np.column_stack([norm_raw[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    def build_idx_for_split(sname: str):
        off = [0]
        cidx = []
        for chs in norm_chunks_by_split[sname]:
            for c in chs:
                cidx.append(norm_c2id[c])
            off.append(len(cidx))
        n = len(norm_chunks_by_split[sname])
        return np.arange(n, dtype=np.int32), np.asarray(off, dtype=np.int64), np.asarray(cidx, dtype=np.int32)

    norm_idx = {s: build_idx_for_split(s) for s in ("pub_train", "v3_train", "pub_calibration", "v3_validation")}

    # Collect H1+H2 needed syntax keys as well
    h123_keys = set(raw_keys)
    h123_glob = set(raw_glob)
    for sname in ("pub_train", "v3_train", "pub_calibration", "v3_validation"):
        uids_s, off_s, cidx_s = norm_idx[sname]
        rows = norm_rows_by_split[sname]
        for r_idx, uid in enumerate(uids_s):
            st, en = int(off_s[uid]), int(off_s[uid + 1])
            ad_cids = cidx_s[st:en]
            sims_mat = norm_sims[ad_cids]
            src = int(rows[r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch = norm_all_chunks[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch):
                    h123_glob.add(ch)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col = cae.ALL_IDS.index(cid)
                for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                    h123_keys.add((norm_all_chunks[int(ad_cids[p])], cid))

    print(f"3. Extracting Natasha syntax features for {len(h123_keys)} pairs & {len(h123_glob)} global chunks...", flush=True)
    syn_cache_path = OUT_CACHE / "h3_syntax_train_cal.pkl"
    if syn_cache_path.exists():
        with syn_cache_path.open("rb") as f:
            syn_d = pickle.load(f)
        h3_sig = syn_d["sig"]
        h3_glob = syn_d["glob"]
    else:
        sa = SyntaxAnalyzer()
        t_n0 = time.perf_counter()
        h3_sig = {k: extract_chunk_syntax_signals(sa, k[0], k[1]) for k in sorted(h123_keys)}
        h3_glob = {ch: detect_chunk_global_restriction(sa, ch) for ch in sorted(h123_glob)}
        print(f"   Natasha parsed in {time.perf_counter()-t_n0:.2f}s", flush=True)
        with syn_cache_path.open("wb") as f:
            pickle.dump({"sig": h3_sig, "glob": h3_glob}, f)

    c_cols = ["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES

    # --- Build Candidate 1: H3_retrained (raw chunks + H3 syntax) ---
    print("4. Training H3_retrained and H1+H2+H3_retrained CatBoost models...", flush=True)
    df_tr_h3 = pd.concat([
        build_pair_dataframe("pub_train", splits["pub_train"], npz["uid_pub_train"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_tr_5[:n_pub_tr], src_tr[:n_pub_tr], p_sgd_oof_tr[:n_pub_tr],
                             h3_sig, h3_glob, dup_groups=dup_groups_tr[:n_pub_tr]),
        build_pair_dataframe("v3_train", splits["v3_train"], npz["uid_v3_train"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_tr_5[n_pub_tr:], src_tr[n_pub_tr:], p_sgd_oof_tr[n_pub_tr:],
                             h3_sig, h3_glob, dup_groups=dup_groups_tr[n_pub_tr:]),
    ], ignore_index=True)
    df_cal_h3 = pd.concat([
        build_pair_dataframe("pub_calibration", splits["pub_calibration"], npz["uid_pub_calibration"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A_cal["pub_calibration"],
                             h3_sig, h3_glob),
        build_pair_dataframe("v3_validation", splits["v3_validation"], npz["uid_v3_validation"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A_cal["v3_validation"],
                             h3_sig, h3_glob),
    ], ignore_index=True)
    for c in SYNTAX_CAT_FEATURES:
        df_tr_h3[c] = df_tr_h3[c].astype(str)
        df_cal_h3[c] = df_cal_h3[c].astype(str)

    cb_h3 = CatBoostClassifier(**{**CATBOOST_PARAMS, "allow_writing_files": False})
    cb_h3.fit(
        Pool(df_tr_h3[c_cols], df_tr_h3["y_true"], cat_features=SYNTAX_CAT_FEATURES),
        eval_set=Pool(df_cal_h3[c_cols], df_cal_h3["y_true"], cat_features=SYNTAX_CAT_FEATURES),
        use_best_model=True,
    )
    cb_h3.save_model(str(OUT_MODELS / "catboost_H3_only.cbm"))

    # --- Build Candidate 2: H1+H2+H3_retrained (norm chunks + H3 syntax, recomputing cal p_sgd; keeping OOF p_sgd on unchanged train and full-head only? Wait: recompute clean OOF p_sgd if ad changed!) ---
    # Check how many train ads actually got different chunks under H1+H2
    p_sgd_cal_norm = {}
    for sname in ("pub_calibration", "v3_validation"):
        uids_s, off_s, cidx_s = norm_idx[sname]
        feats, _ = cae.build_category_features_for_uids(uids_s, off_s, cidx_s, norm_emb_f32, norm_sims, return_selection=False)
        p_sgd_cal_norm[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    # For train ads whose chunks changed under H1+H2, compute genuine out-of-fold SGD probability using exact 5-fold CV (StratifiedKFold per cid, matching cae)
    changed_tr_mask = []
    for sname in ("pub_train", "v3_train"):
        for r_idx, r in enumerate(splits[sname]):
            raw_ch = chunk_ad_fixed(r["description"]) or [""]
            changed_tr_mask.append(raw_ch != norm_chunks_by_split[sname][r_idx])
    changed_tr_mask = np.array(changed_tr_mask, dtype=bool)
    print(f"   Train ads whose chunks changed under H1+H2: {int(changed_tr_mask.sum())}/{len(tr_rows)}", flush=True)

    # Recompute exact OOF SGD ONLY where needed, or compute full 5-fold OOF SGD cleanly across all 6,319 normalized train ads
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    tr_norm_off = [0]
    tr_norm_cidx = []
    for sname in ("pub_train", "v3_train"):
        for chs in norm_chunks_by_split[sname]:
            for c in chs:
                tr_norm_cidx.append(norm_c2id[c])
            tr_norm_off.append(len(tr_norm_cidx))
    tr_norm_feats, _ = cae.build_category_features_for_uids(
        np.arange(len(tr_rows), dtype=np.int32),
        np.asarray(tr_norm_off, dtype=np.int64),
        np.asarray(tr_norm_cidx, dtype=np.int32),
        norm_emb_f32,
        norm_sims,
        return_selection=False,
    )
    p_sgd_oof_norm_tr = p_sgd_oof_tr.copy()
    sgkf = StratifiedGroupKFold(n_splits=5)
    for j, cid in enumerate(cae.SCOPE_102_106):
        X_j = tr_norm_feats[cid]
        y_j = y_tr_5[:, j]
        active_mask = (src_tr != cid)
        oof_j = np.zeros(len(tr_rows), dtype=np.float64)
        alpha_j = float(sgd_heads[cid].alpha)
        for tr_i, va_i in sgkf.split(X_j[active_mask], y_j[active_mask], dup_groups_tr[active_mask]):
            act_idx = np.where(active_mask)[0]
            clf = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=alpha_j, max_iter=2000, tol=1e-4,
                random_state=42, class_weight="balanced",
            ).fit(X_j[act_idx[tr_i]], y_j[act_idx[tr_i]])
            oof_j[act_idx[va_i]] = clf.predict_proba(X_j[act_idx[va_i]])[:, 1]
        # Replace ONLY changed ads to keep exact original OOF calibration on unchanged 97% of train!
        p_sgd_oof_norm_tr[changed_tr_mask, j] = oof_j[changed_tr_mask]

    df_tr_h123 = pd.concat([
        build_pair_dataframe("pub_train", norm_rows_by_split["pub_train"], *norm_idx["pub_train"],
                             norm_sims, norm_all_chunks, y_tr_5[:n_pub_tr], src_tr[:n_pub_tr], p_sgd_oof_norm_tr[:n_pub_tr],
                             h3_sig, h3_glob, dup_groups=dup_groups_tr[:n_pub_tr]),
        build_pair_dataframe("v3_train", norm_rows_by_split["v3_train"], *norm_idx["v3_train"],
                             norm_sims, norm_all_chunks, y_tr_5[n_pub_tr:], src_tr[n_pub_tr:], p_sgd_oof_norm_tr[n_pub_tr:],
                             h3_sig, h3_glob, dup_groups=dup_groups_tr[n_pub_tr:]),
    ], ignore_index=True)
    df_cal_h123 = pd.concat([
        build_pair_dataframe("pub_calibration", norm_rows_by_split["pub_calibration"], *norm_idx["pub_calibration"],
                             norm_sims, norm_all_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], p_sgd_cal_norm["pub_calibration"],
                             h3_sig, h3_glob),
        build_pair_dataframe("v3_validation", norm_rows_by_split["v3_validation"], *norm_idx["v3_validation"],
                             norm_sims, norm_all_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], p_sgd_cal_norm["v3_validation"],
                             h3_sig, h3_glob),
    ], ignore_index=True)
    for c in SYNTAX_CAT_FEATURES:
        df_tr_h123[c] = df_tr_h123[c].astype(str)
        df_cal_h123[c] = df_cal_h123[c].astype(str)

    cb_h123 = CatBoostClassifier(**{**CATBOOST_PARAMS, "allow_writing_files": False})
    cb_h123.fit(
        Pool(df_tr_h123[c_cols], df_tr_h123["y_true"], cat_features=SYNTAX_CAT_FEATURES),
        eval_set=Pool(df_cal_h123[c_cols], df_cal_h123["y_true"], cat_features=SYNTAX_CAT_FEATURES),
        use_best_model=True,
    )
    cb_h123.save_model(str(OUT_MODELS / "catboost_H123.cbm"))

    # Score baseline C_syntax_fixed on calibration
    cb_base = CatBoostClassifier()
    cb_base.load_model(str(ROOT / "output" / "syntax-link-fix" / "models" / "catboost_C_syntax_fixed.cbm"))
    df_cal_base = pd.concat([
        build_pair_dataframe("pub_calibration", splits["pub_calibration"], npz["uid_pub_calibration"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A_cal["pub_calibration"],
                             base_syn["chunk_sig_cache"], base_syn["chunk_global_cache"]),
        build_pair_dataframe("v3_validation", splits["v3_validation"], npz["uid_v3_validation"], offsets_v2, cindices_v2,
                             v2_cat_sims, v2_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A_cal["v3_validation"],
                             base_syn["chunk_sig_cache"], base_syn["chunk_global_cache"]),
    ], ignore_index=True)
    for c in SYNTAX_CAT_FEATURES:
        df_cal_base[c] = df_cal_base[c].astype(str)

    m_base, yp_base = score_cal(df_cal_base, cb_base, c_cols, cal_rows, n_pub_cal, y_cal_5, src_cal)
    m_h3, yp_h3 = score_cal(df_cal_h3, cb_h3, c_cols, cal_rows, n_pub_cal, y_cal_5, src_cal)
    m_h123, yp_h123 = score_cal(df_cal_h123, cb_h123, c_cols, cal_rows, n_pub_cal, y_cal_5, src_cal)

    print("\n=== CALIBRATION COMPARISON (Selection Gate, zero test leakage) ===", flush=True)
    for tag, m in [("baseline_C_syntax_fixed", m_base), ("H3_retrained", m_h3), ("H1+H2+H3_retrained", m_h123)]:
        print(f"  {tag:24s} micro_f1={m['micro_f1']:.6f} P={m['micro_precision']:.6f} R={m['micro_recall']:.6f} "
              f"TP={m['confusion']['tp']} FP={m['confusion']['fp']} FN={m['confusion']['fn']} EM={m['exact_match']:.6f}", flush=True)

    # Evaluate robustness_benchmark for both retrained candidates
    print("\n5. Evaluating retrained candidates on independent robustness_benchmark...", flush=True)
    cases = build_cases()
    runner = RobustPipelineRunner(
        cb_model_path=OUT_MODELS / "catboost_H3_only.cbm",
        sgd_heads_path=EXP_MODELS_DIR / "sgd_full_heads.joblib",
        queries_path=ROOT / "models" / "c_syntax" / "queries.npz",
    )
    runner._emb_cache.update(chunk_emb_map)
    dz_all = np.load(delta_cache_path, allow_pickle=False)
    for c, v in zip(dz_all["chunks"].tolist(), dz_all["embs"].astype(np.float32)):
        runner._emb_cache[c] = v
    runner.predict_one("Ремонт.")
    res_h3 = evaluate_suite(runner, cases, "H3_retrained")
    print(f"  [H3_retrained] passed={res_h3['passed']}/{res_h3['total']} ({res_h3['pass_rate']*100:.1f}%) "
          f"short={res_h3['short_median_ms']}ms long={res_h3['long_median_ms']}ms 75k={res_h3['stress_75k_pred']}({res_h3['stress_75k_ms']}ms)")

    runner.cb = cb_h123
    runner.preprocess_fn = lambda s: strip_prompt_injections(normalize_ad_text(s))
    res_h123 = evaluate_suite(runner, cases, "H1+H2+H3_retrained")
    print(f"  [H1+H2+H3_retrained] passed={res_h123['passed']}/{res_h123['total']} ({res_h123['pass_rate']*100:.1f}%) "
          f"short={res_h123['short_median_ms']}ms long={res_h123['long_median_ms']}ms 75k={res_h123['stress_75k_pred']}({res_h123['stress_75k_ms']}ms)")
    for r in res_h123["cases"]:
        if not r["ok"]:
            print(f"    FAIL {r['name']}: exp={r['expected']} pred={r['predicted']} probs={r['probs']}")

    summary = {
        "calibration": {
            "baseline_C_syntax_fixed": m_base,
            "H3_retrained": m_h3,
            "H1+H2+H3_retrained": m_h123,
        },
        "robustness": {
            "H3_retrained": res_h3,
            "H1+H2+H3_retrained": res_h123,
        },
    }
    (OUT_DIR / "ablation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
