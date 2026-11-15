#!/usr/bin/env python3
"""Isolated ablation evaluation on calibration (1,328 ads, zero test usage)."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import pickle
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics
from avito_compare.scripts.eval_public_tests_syntax import reconstruct_v2_chunk_map
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.run_syntax_experiment import build_pair_dataframe, probs_to_eval_matrix
from avito_compare.scripts.syntax_features import (
    GLOBAL_CANDIDATE_RE,
    SYNTAX_CAT_FEATURES,
    SyntaxAnalyzer,
    detect_chunk_global_restriction,
    extract_chunk_syntax_signals,
)
from avito_compare.scripts.text_preprocess import normalize_ad_text, strip_prompt_injections

PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
V2_CACHE_NPZ = ROOT / "output" / "cat-aware-v2" / "cache" / "chunks_embeddings.npz"


def main() -> None:
    splits = {
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
    }
    cal_rows = splits["pub_calibration"] + splits["v3_validation"]
    n_pub_cal = len(splits["pub_calibration"])
    v2_chunks, v2_chunk_emb, query_emb_f32 = reconstruct_v2_chunk_map()
    npz = np.load(V2_CACHE_NPZ)
    v2_emb_f32 = v2_chunk_emb.astype(np.float32)
    v2_raw_sims = v2_emb_f32 @ query_emb_f32.T
    v2_cat_sims = np.column_stack([v2_raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
    offsets_v2 = npz["ad_chunk_offsets"]
    cindices_v2 = npz["ad_chunk_indices"]
    y_cal_5, src_cal = cae._rows_xy_for_ids(cal_rows, cae.SCOPE_102_106, use_eval=True)

    sgd_heads = joblib.load(ROOT / "output" / "syntax-aware-experiment" / "models" / "sgd_full_heads.joblib")["heads"]
    cb = CatBoostClassifier()
    cb.load_model(str(ROOT / "output" / "syntax-link-fix" / "models" / "catboost_C_syntax_fixed.cbm"))

    probs_A = {}
    for sname in ("pub_calibration", "v3_validation"):
        feats, _ = cae.build_category_features_for_uids(
            npz[f"uid_{sname}"], offsets_v2, cindices_v2, v2_emb_f32, v2_cat_sims, return_selection=False
        )
        probs_A[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    with (ROOT / "output" / "syntax-link-fix" / "cache" / "chunk_syntax_cache_fixed.pkl").open("rb") as f:
        old_syn = pickle.load(f)

    def score_df(df_cal: pd.DataFrame, tag: str) -> np.ndarray:
        X_cal = df_cal[cb.feature_names_].copy()
        for c in SYNTAX_CAT_FEATURES:
            X_cal[c] = X_cal[c].astype(str)
        probs = cb.predict_proba(X_cal)[:, 1]
        if "split" in df_cal.columns and set(df_cal["split"].unique()) == {"pub_calibration", "v3_validation"}:
            m_pub = df_cal["split"].to_numpy() == "pub_calibration"
            m_v3 = df_cal["split"].to_numpy() == "v3_validation"
            p_pub = probs_to_eval_matrix(df_cal[m_pub], probs[m_pub], n_pub_cal, src_cal[:n_pub_cal], cae.SCOPE_102_106)
            p_v3 = probs_to_eval_matrix(df_cal[m_v3], probs[m_v3], len(cal_rows) - n_pub_cal, src_cal[n_pub_cal:], cae.SCOPE_102_106)
            p_mat = np.vstack([p_pub, p_v3])
        else:
            p_mat = probs_to_eval_matrix(df_cal, probs, len(cal_rows), src_cal, cae.SCOPE_102_106)
        m, yp = compute_detailed_metrics(y_cal_5, p_mat, np.full(5, 0.5), src_cal, cae.SCOPE_102_106)
        print(
            f"[{tag}] micro_f1={m['micro_f1']:.6f} P={m['micro_precision']:.6f} R={m['micro_recall']:.6f} "
            f"TP={m['confusion']['tp']} FP={m['confusion']['fp']} FN={m['confusion']['fn']} EM={m['exact_match']:.6f}",
            flush=True,
        )
        return yp

    # 1. Baseline
    df_base = pd.concat([
        build_pair_dataframe(
            "pub_calibration", splits["pub_calibration"], npz["uid_pub_calibration"], offsets_v2, cindices_v2,
            v2_cat_sims, v2_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A["pub_calibration"],
            old_syn["chunk_sig_cache"], old_syn["chunk_global_cache"],
        ),
        build_pair_dataframe(
            "v3_validation", splits["v3_validation"], npz["uid_v3_validation"], offsets_v2, cindices_v2,
            v2_cat_sims, v2_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A["v3_validation"],
            old_syn["chunk_sig_cache"], old_syn["chunk_global_cache"],
        ),
    ], ignore_index=True)
    yp_base = score_df(df_base, "baseline_C_syntax_fixed")

    # 2. H3 syntax only
    needed_keys: set[tuple[str, int]] = set()
    needed_glob: set[str] = set()
    for sname in ("pub_calibration", "v3_validation"):
        for r_idx, uid in enumerate(npz[f"uid_{sname}"]):
            st, en = int(offsets_v2[int(uid)]), int(offsets_v2[int(uid) + 1])
            ad_cids = cindices_v2[st:en]
            sims_mat = v2_cat_sims[ad_cids]
            src = int(splits[sname][r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch = v2_chunks[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch):
                    needed_glob.add(ch)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col = cae.ALL_IDS.index(cid)
                for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                    needed_keys.add((v2_chunks[int(ad_cids[p])], cid))

    sa = SyntaxAnalyzer()
    t0 = time.perf_counter()
    h3_sig = {k: extract_chunk_syntax_signals(sa, k[0], k[1]) for k in needed_keys}
    h3_glob = {ch: detect_chunk_global_restriction(sa, ch) for ch in needed_glob}
    print(f"Re-parsed H3 cal ({len(needed_keys)} pairs) in {time.perf_counter()-t0:.2f}s", flush=True)

    df_h3 = pd.concat([
        build_pair_dataframe(
            "pub_calibration", splits["pub_calibration"], npz["uid_pub_calibration"], offsets_v2, cindices_v2,
            v2_cat_sims, v2_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A["pub_calibration"],
            h3_sig, h3_glob,
        ),
        build_pair_dataframe(
            "v3_validation", splits["v3_validation"], npz["uid_v3_validation"], offsets_v2, cindices_v2,
            v2_cat_sims, v2_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A["v3_validation"],
            h3_sig, h3_glob,
        ),
    ], ignore_index=True)
    yp_h3 = score_df(df_h3, "H3_syntax_zeroshot")

    # 3. Pre-chunk H1+H2 (with Qwen encoding ONLY on newly created chunks)
    chunk_emb_map = {ch: v2_emb_f32[i] for i, ch in enumerate(v2_chunks)}
    norm_chunks_per_ad: list[list[str]] = []
    new_unique_chunks: list[str] = []
    seen_new: set[str] = set()
    changed_ads = 0
    for r in cal_rows:
        norm_desc = strip_prompt_injections(normalize_ad_text(r["description"]))
        if norm_desc != r["description"]:
            changed_ads += 1
        chs = chunk_ad_fixed(norm_desc) or [""]
        norm_chunks_per_ad.append(chs)
        for c in chs:
            if c not in chunk_emb_map and c not in seen_new:
                seen_new.add(c)
                new_unique_chunks.append(c)

    print(f"H1+H2 cal: changed_ads={changed_ads}/{len(cal_rows)}, new_chunks_to_encode={len(new_unique_chunks)}", flush=True)
    if new_unique_chunks:
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
        new_vecs = np.asarray(
            enc.encode(new_unique_chunks, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float16,
        ).astype(np.float32)
        for c, v in zip(new_unique_chunks, new_vecs):
            chunk_emb_map[c] = v

    norm_all_chunks = list(chunk_emb_map.keys())
    norm_c2id = {c: i for i, c in enumerate(norm_all_chunks)}
    norm_emb_f32 = np.vstack([chunk_emb_map[c] for c in norm_all_chunks])
    norm_raw = norm_emb_f32 @ query_emb_f32.T
    norm_sims = np.column_stack([norm_raw[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
    n_off = [0]
    n_cidx = []
    for chs in norm_chunks_per_ad:
        for c in chs:
            n_cidx.append(norm_c2id[c])
        n_off.append(len(n_cidx))
    n_off_arr = np.asarray(n_off, dtype=np.int64)
    n_cidx_arr = np.asarray(n_cidx, dtype=np.int32)
    cal_norm_rows = [{**r, "description": strip_prompt_injections(normalize_ad_text(r["description"]))} for r in cal_rows]
    cal_uids_norm = np.arange(len(cal_rows), dtype=np.int32)
    n_feats, _ = cae.build_category_features_for_uids(cal_uids_norm, n_off_arr, n_cidx_arr, norm_emb_f32, norm_sims, return_selection=False)
    p_sgd_norm = np.column_stack([sgd_heads[cid].predict_proba(n_feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    # Parse any new chunk syntax keys for H1+H2+H3 (plus baseline old_syn check for H1+H2 alone!)
    new_keys_only: set[tuple[str, int]] = set()
    new_glob_only: set[str] = set()
    for r_idx, uid in enumerate(cal_uids_norm):
        st, en = int(n_off_arr[uid]), int(n_off_arr[uid + 1])
        ad_cids = n_cidx_arr[st:en]
        sims_mat = norm_sims[ad_cids]
        src = int(cal_rows[r_idx].get("source_mcId") or 0)
        k_take = min(cae.TOP_K, len(ad_cids))
        for c_int in ad_cids:
            ch = norm_all_chunks[int(c_int)]
            if ch not in h3_glob and GLOBAL_CANDIDATE_RE.search(ch):
                new_glob_only.add(ch)
        for cid in cae.SCOPE_102_106:
            if src == cid:
                continue
            col = cae.ALL_IDS.index(cid)
            for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                key = (norm_all_chunks[int(ad_cids[p])], cid)
                if key not in h3_sig:
                    new_keys_only.add(key)

    for k in new_keys_only:
        h3_sig[k] = extract_chunk_syntax_signals(sa, k[0], k[1])
    for ch in new_glob_only:
        h3_glob[ch] = detect_chunk_global_restriction(sa, ch)

    # Also build H1+H2 alone with old syntax
    old_sig_plus = dict(old_syn["chunk_sig_cache"])
    old_glob_plus = dict(old_syn["chunk_global_cache"])
    # Wait: compute H1+H2+H3 directly first
    df_h123 = build_pair_dataframe(
        "cal_norm", cal_norm_rows, cal_uids_norm, n_off_arr, n_cidx_arr,
        norm_sims, norm_all_chunks, y_cal_5, src_cal, p_sgd_norm,
        h3_sig, h3_glob,
    )
    yp_123 = score_df(df_h123, "H1+H2+H3_zeroshot")

    # Show exact changed ads between baseline and H3, and baseline and H1+H2+H3
    for dtag, yp_cmp in [("H3_zeroshot", yp_h3), ("H1+H2+H3_zeroshot", yp_123)]:
        diff_mask = (yp_base != yp_cmp)
        n_fixed = int(((yp_base != y_cal_5) & (yp_cmp == y_cal_5)).sum())
        n_broken = int(((yp_base == y_cal_5) & (yp_cmp != y_cal_5)).sum())
        print(f"\nDiff baseline -> {dtag}: changed={int(diff_mask.sum())} FIXED={n_fixed} BROKEN={n_broken}")
        shown = 0
        for r_idx in range(len(cal_rows)):
            for j, cid in enumerate(cae.SCOPE_102_106):
                if src_cal[r_idx] == cid:
                    continue
                if yp_base[r_idx, j] != yp_cmp[r_idx, j]:
                    y = int(y_cal_5[r_idx, j])
                    status = "FIXED" if yp_cmp[r_idx, j] == y else "BROKEN"
                    if shown < 30:
                        print(f"  [{status}] r={r_idx} cat={cid} y={y} {yp_base[r_idx,j]}->{yp_cmp[r_idx,j]} text={cal_rows[r_idx]['description'][:90]!r}")
                        shown += 1


if __name__ == "__main__":
    main()
