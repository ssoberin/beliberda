#!/usr/bin/env python3
"""Once-only held-out evaluation & uncached end-to-end latency benchmark.

Compares:
  - C_syntax (models/c_syntax/catboost.cbm)
  - C_syntax_fixed (output/syntax-link-fix/models/catboost_C_syntax_fixed.cbm)
  - H3_retrained (output/final-robustness/models/catboost_H3_only.cbm)
  - H1+H2+H3_retrained (output/final-robustness/models/catboost_H123.cbm)

on frozen held-out sets (pub_validation, v3_test, public_test_208, public_test_clean),
including short (<1000) vs long (>=1000) breakdowns and uncached CPU latency.
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
from avito_compare.scripts.eval_public_tests_syntax import (
    DELTA_CHUNKS_JSON,
    DELTA_EMBS_NPZ,
    OFFICIAL_208_PQ,
    reconstruct_v2_chunk_map,
)
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.qwen3_embedding_experiment import load_official_parquet
from avito_compare.scripts.run_syntax_experiment import (
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
    extract_pair_features,
)
from avito_compare.scripts.text_preprocess import normalize_ad_text, strip_prompt_injections

PUBLIC_DIR = ROOT / "output" / "ngram-comparison" / "splits"
V3_CLEAN_DIR = ROOT / "datasets" / "own_v3_clean"
V2_CACHE_NPZ = ROOT / "output" / "cat-aware-v2" / "cache" / "chunks_embeddings.npz"
EXP_MODELS_DIR = ROOT / "output" / "syntax-aware-experiment" / "models"
OPT_DIR = ROOT / "output" / "catboost-optimization"
OUT_DIR = ROOT / "output" / "final-robustness"
OUT_CACHE = OUT_DIR / "cache"
OUT_MODELS = OUT_DIR / "models"


def metrics_summary(y_true: np.ndarray, probs: np.ndarray, thr: float, src: np.ndarray) -> tuple[dict, np.ndarray]:
    th_arr = np.full(5, thr, dtype=np.float64)
    m, yp = compute_detailed_metrics(y_true, probs, th_arr, src, cae.SCOPE_102_106)
    out = {
        "rows": int(len(y_true)),
        "micro_f1": round(float(m["micro_f1"]), 6),
        "precision": round(float(m["micro_precision"]), 6),
        "recall": round(float(m["micro_recall"]), 6),
        "macro_f1": round(float(m["macro_f1"]), 6),
        "exact_match": round(float(m["exact_match"]), 6),
        "tp": int(m["confusion"]["tp"]),
        "fp": int(m["confusion"]["fp"]),
        "fn": int(m["confusion"]["fn"]),
        "tn": int(m["confusion"]["tn"]),
        "per_cat": {},
    }
    for j, cid in enumerate(cae.SCOPE_102_106):
        mask = src != cid
        yt = y_true[mask, j]
        pr = yp[mask, j]
        tp = int(((yt == 1) & (pr == 1)).sum())
        fp = int(((yt == 0) & (pr == 1)).sum())
        fn = int(((yt == 1) & (pr == 0)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        out["per_cat"][str(cid)] = {
            "f1": round(f1, 6),
            "precision": round(prec, 6),
            "recall": round(rec, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return out, yp


def length_breakdown(y_true: np.ndarray, probs: np.ndarray, src: np.ndarray, rows: list[dict]) -> dict:
    lens = np.array([len(r["description"]) for r in rows])
    res = {}
    for bucket, mask in [("short_lt1000", lens < 1000), ("long_ge1000", lens >= 1000)]:
        if int(mask.sum()) == 0:
            continue
        m, _ = metrics_summary(y_true[mask], probs[mask], 0.5, src[mask])
        res[bucket] = {
            "rows": int(mask.sum()),
            "micro_f1": m["micro_f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "exact_match": m["exact_match"],
            "tp": m["tp"],
            "fp": m["fp"],
            "fn": m["fn"],
        }
    return res


def diff_summary(base_m: dict, new_m: dict, y_true: np.ndarray, yp_old: np.ndarray, yp_new: np.ndarray, src: np.ndarray) -> dict:
    trans = error_transition_counts(y_true, yp_old, yp_new, src, cae.SCOPE_102_106)
    return {
        "delta_micro_f1": round(new_m["micro_f1"] - base_m["micro_f1"], 6),
        "delta_precision": round(new_m["precision"] - base_m["precision"], 6),
        "delta_recall": round(new_m["recall"] - base_m["recall"], 6),
        "delta_exact_match": round(new_m["exact_match"] - base_m["exact_match"], 6),
        "delta_fp": new_m["fp"] - base_m["fp"],
        "delta_fn": new_m["fn"] - base_m["fn"],
        "fixed_errors": int(trans["base_wrong_cand_right_fixed"]),
        "broken_errors": int(trans["base_right_cand_wrong_broken"]),
        "net_error_reduction": int(trans["net_gain"]),
    }


def measure_uncached_latency(query_emb_f32: np.ndarray, sgd_heads: dict, cb_base: CatBoostClassifier, cb_h123: CatBoostClassifier) -> dict:
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    local_dir = snapshot_download("Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", local_files_only=True)
    kw = "processor_kwargs" if "processor_kwargs" in inspect.signature(SentenceTransformer.__init__).parameters else "tokenizer_kwargs"
    enc = SentenceTransformer(
        local_dir,
        device="cpu",
        local_files_only=True,
        model_kwargs={"dtype": torch.float32},
        **{kw: {"padding_side": "left"}},
    )
    enc.max_seq_length = 1024
    # Warmup
    _ = enc.encode(["Тестовый прогрев модели."], batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    sa = SyntaxAnalyzer()
    c_cols = ["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES

    short_txt = "Выполняем сантехнические работы: установка раковины, замена труб и смесителей. Укладываем плитку."
    med_txt = (short_txt + "\n\n") + ("Работаем по договору, чистый инструмент, выезд на замер, прозрачная смета без скрытых доплат. " * 10)
    long_10k_txt = (med_txt + "\n\n") + ("Соблюдаем чистоту на объекте, гарантия качества, помощь в подборе стройматериалов по чекам. " * 100)
    stress_75k_txt = (short_txt + "\n\n") + (
        ("Соблюдаем чистоту на объекте, работаем по договору, аккуратная доставка строительных материалов, "
         "выезд на замер по городу и области, прозрачная смета без скрытых доплат, гарантия качества. ") * 400
    )

    samples = [
        ("short", short_txt, 5),
        ("medium_1k", med_txt, 5),
        ("long_10k", long_10k_txt, 3),
        ("stress_75k", stress_75k_txt, 1),
    ]

    out = {}
    for cfg_name, use_prep, cb_m in [("baseline_raw", False, cb_base), ("H1+H2+H3", True, cb_h123)]:
        out[cfg_name] = {}
        for label, raw_text, repeats in samples:
            totals = []
            stages = {"prep_ms": [], "chunk_ms": [], "qwen_ms": [], "sgd_ms": [], "natasha_ms": [], "cb_ms": []}
            n_chunks_last = 0
            len_last = 0
            for _ in range(repeats):
                sa._sent_cache.clear()
                t0 = time.perf_counter()
                txt = strip_prompt_injections(normalize_ad_text(raw_text)) if use_prep else raw_text
                len_last = len(txt)
                t1 = time.perf_counter()
                chunks = chunk_ad_fixed(txt) or [""]
                n_chunks_last = len(chunks)
                unique = list(dict.fromkeys(chunks))
                u_pos = {t: i for i, t in enumerate(unique)}
                cidx = np.array([u_pos[t] for t in chunks], dtype=np.int32)
                t2 = time.perf_counter()
                u_embs = np.asarray(
                    enc.encode(unique, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
                    dtype=np.float32,
                )
                t3 = time.perf_counter()
                raw_sims = u_embs @ query_emb_f32.T
                cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
                off = np.array([0, len(chunks)], dtype=np.int64)
                uids = np.array([0], dtype=np.int32)
                feats, _ = cae.build_category_features_for_uids(uids, off, cidx, u_embs, cat_sims, return_selection=False)
                p_sgd = {cid: float(sgd_heads[cid].predict_proba(feats[cid])[0, 1]) for cid in cae.SCOPE_102_106}
                t4 = time.perf_counter()

                scores_per_cat = cat_sims[cidx]
                rows_df = []
                for cid in cae.SCOPE_102_106:
                    col = cae.ALL_IDS.index(cid)
                    sc = scores_per_cat[:, col]
                    sel_3 = np.argsort(-sc, kind="stable")[:min(3, len(chunks))].tolist()
                    syn = extract_pair_features(sa, chunks, sel_3, [float(sc[p]) for p in sel_3], cid, len(txt))
                    syn["p_sgd"] = p_sgd[cid]
                    rows_df.append(syn)
                df = pd.DataFrame(rows_df)
                for col in SYNTAX_CAT_FEATURES:
                    df[col] = df[col].astype(str)
                t5 = time.perf_counter()
                _ = cb_m.predict_proba(Pool(df[cb_m.feature_names_], cat_features=SYNTAX_CAT_FEATURES))[:, 1]
                t6 = time.perf_counter()

                stages["prep_ms"].append((t1 - t0) * 1000)
                stages["chunk_ms"].append((t2 - t1) * 1000)
                stages["qwen_ms"].append((t3 - t2) * 1000)
                stages["sgd_ms"].append((t4 - t3) * 1000)
                stages["natasha_ms"].append((t5 - t4) * 1000)
                stages["cb_ms"].append((t6 - t5) * 1000)
                totals.append((t6 - t0) * 1000)

            out[cfg_name][label] = {
                "chars": len_last,
                "chunks": n_chunks_last,
                "repeats": repeats,
                "median_total_ms": round(float(np.median(totals)), 2),
                "median_stages_ms": {k: round(float(np.median(v)), 2) for k, v in stages.items()},
            }
    return out


def main() -> None:
    t0 = time.perf_counter()
    print("1. Loading frozen held-out datasets & embedding caches...", flush=True)
    v2_chunks, v2_chunk_emb, query_emb_f32 = reconstruct_v2_chunk_map()
    npz = np.load(V2_CACHE_NPZ)
    v2_emb_f32 = v2_chunk_emb.astype(np.float32)
    v2_raw_sims = v2_emb_f32 @ query_emb_f32.T
    v2_cat_sims = np.column_stack([v2_raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
    offsets_v2 = npz["ad_chunk_offsets"]
    cindices_v2 = npz["ad_chunk_indices"]

    pub_val_rows = cae.load_public_rows(PUBLIC_DIR / "validation.jsonl")
    v3_test_rows = cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl")
    _, _, off_rows = load_official_parquet(OFFICIAL_208_PQ)
    for idx, r in enumerate(off_rows):
        r["item_id"] = f"official_208_{idx}"
        r["source_mcId"] = 101
        r["labels"] = r["evaluation_labels"]
        r["construction"] = ""
    excl = json.loads((OPT_DIR / "public_test_exclusions.json").read_text())["excluded_row_indices_zero_based"]
    clean_mask_208 = np.array([i not in set(excl) for i in range(len(off_rows))], dtype=bool)

    y_pv, src_pv = cae._rows_xy_for_ids(pub_val_rows, cae.SCOPE_102_106, use_eval=True)
    y_v3t, src_v3t = cae._rows_xy_for_ids(v3_test_rows, cae.SCOPE_102_106, use_eval=True)
    y_off, src_off = cae._rows_xy_for_ids(off_rows, cae.SCOPE_102_106, use_eval=True)

    # Raw chunk map (v2 + official delta)
    delta_sorted = json.loads(DELTA_CHUNKS_JSON.read_text(encoding="utf-8"))
    delta_emb = np.load(DELTA_EMBS_NPZ, allow_pickle=False)["embs"].astype(np.float32)
    raw_emb_map = {ch: v2_emb_f32[i] for i, ch in enumerate(v2_chunks)}
    for ch, vec in zip(delta_sorted, delta_emb):
        raw_emb_map[ch] = vec

    # official_208 raw chunks/index
    off_chunks: list[str] = []
    off_c2id: dict[str, int] = {}
    off_offsets = [0]
    off_cindices: list[int] = []
    for r in off_rows:
        for ch in chunk_ad_fixed(r["description"]):
            if ch not in off_c2id:
                off_c2id[ch] = len(off_chunks)
                off_chunks.append(ch)
            off_cindices.append(off_c2id[ch])
        off_offsets.append(len(off_cindices))
    off_off_arr = np.asarray(off_offsets, dtype=np.int64)
    off_cidx_arr = np.asarray(off_cindices, dtype=np.int32)
    off_uids = np.arange(len(off_rows), dtype=np.int32)
    off_emb_f32 = np.vstack([raw_emb_map[ch] for ch in off_chunks])
    off_raw_sims = off_emb_f32 @ query_emb_f32.T
    off_cat_sims = np.column_stack([off_raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    sgd_heads = joblib.load(EXP_MODELS_DIR / "sgd_full_heads.joblib")["heads"]
    probs_raw = {}
    for sname in ("pub_validation", "v3_test"):
        feats, _ = cae.build_category_features_for_uids(npz[f"uid_{sname}"], offsets_v2, cindices_v2, v2_emb_f32, v2_cat_sims, return_selection=False)
        probs_raw[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])
    off_feats, _ = cae.build_category_features_for_uids(off_uids, off_off_arr, off_cidx_arr, off_emb_f32, off_cat_sims, return_selection=False)
    probs_raw["official_208"] = np.column_stack([sgd_heads[cid].predict_proba(off_feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    # Frozen C_syntax frames from final_eval.pkl & baseline fixed cache
    with (OPT_DIR / "cache" / "final_eval.pkl").open("rb") as f:
        final_eval_old = pickle.load(f)
    old_frames = {
        "pub_validation": final_eval_old["frames"]["pub_validation"].copy(),
        "official_208": final_eval_old["frames"]["official_208"].copy(),
        "v3_test": final_eval_old["frames"]["v3_test"].copy(),
    }
    with (ROOT / "output" / "syntax-link-fix" / "cache" / "chunk_syntax_cache_fixed.pkl").open("rb") as f:
        base_fixed_syn = pickle.load(f)

    # 2. Load H1+H2 delta embeddings cache and prepare normalized held-out chunks
    print("2. Preparing H1+H2 normalized held-out splits & extracting current Natasha signals...", flush=True)
    h12_emb_map = dict(raw_emb_map)
    dz = np.load(OUT_CACHE / "h12_train_cal_delta.npz", allow_pickle=False)
    for ch, vec in zip(dz["chunks"].tolist(), dz["embs"].astype(np.float32)):
        h12_emb_map[ch] = vec

    raw_specs = [
        ("pub_validation", pub_val_rows, npz["uid_pub_validation"], offsets_v2, cindices_v2, v2_cat_sims, v2_chunks, y_pv, src_pv, probs_raw["pub_validation"]),
        ("v3_test", v3_test_rows, npz["uid_v3_test"], offsets_v2, cindices_v2, v2_cat_sims, v2_chunks, y_v3t, src_v3t, probs_raw["v3_test"]),
        ("official_208", off_rows, off_uids, off_off_arr, off_cidx_arr, off_cat_sims, off_chunks, y_off, src_off, probs_raw["official_208"]),
    ]

    norm_specs = {}
    missing_norm = []
    for sname, rows, _, _, _, _, _, y_mat, src_vec, _ in raw_specs:
        norm_rows = [{**r, "description": strip_prompt_injections(normalize_ad_text(r["description"]))} for r in rows]
        chs_list = [chunk_ad_fixed(r["description"]) or [""] for r in norm_rows]
        u_chunks: list[str] = []
        c2id: dict[str, int] = {}
        off = [0]
        cidx = []
        for chs in chs_list:
            for c in chs:
                if c not in h12_emb_map:
                    missing_norm.append(c)
                if c not in c2id:
                    c2id[c] = len(u_chunks)
                    u_chunks.append(c)
                cidx.append(c2id[c])
            off.append(len(cidx))
        norm_specs[sname] = (norm_rows, u_chunks, np.arange(len(rows), dtype=np.int32), np.asarray(off, dtype=np.int64), np.asarray(cidx, dtype=np.int32), y_mat, src_vec)

    assert len(missing_norm) == 0, f"Unexpected uncached H1+H2 chunks in held-out eval: {len(missing_norm)}"

    # Extract Natasha features on raw & norm held-out keys (shared single pass)
    needed_keys: set[tuple[str, int]] = set()
    needed_glob: set[str] = set()
    for sname, rows, uids_s, off_s, cidx_s, sims_all, ch_vocab, _, _, _ in raw_specs:
        for r_idx, uid in enumerate(uids_s):
            st, en = int(off_s[int(uid)]), int(off_s[int(uid) + 1])
            ad_cids = cidx_s[st:en]
            sims_mat = sims_all[ad_cids]
            src = int(rows[r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch = ch_vocab[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch):
                    needed_glob.add(ch)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col = cae.ALL_IDS.index(cid)
                for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                    needed_keys.add((ch_vocab[int(ad_cids[p])], cid))

    norm_built = {}
    for sname, (n_rows, u_chs, uids_s, off_s, cidx_s, y_mat, src_vec) in norm_specs.items():
        emb_f32 = np.vstack([h12_emb_map[c] for c in u_chs])
        r_sims = emb_f32 @ query_emb_f32.T
        cat_sims = np.column_stack([r_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])
        feats, _ = cae.build_category_features_for_uids(uids_s, off_s, cidx_s, emb_f32, cat_sims, return_selection=False)
        p_sgd = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])
        norm_built[sname] = (n_rows, uids_s, off_s, cidx_s, cat_sims, u_chs, y_mat, src_vec, p_sgd)
        for r_idx, uid in enumerate(uids_s):
            st, en = int(off_s[uid]), int(off_s[uid + 1])
            ad_cids = cidx_s[st:en]
            sims_mat = cat_sims[ad_cids]
            src = int(n_rows[r_idx].get("source_mcId") or 0)
            k_take = min(cae.TOP_K, len(ad_cids))
            for c_int in ad_cids:
                ch = u_chs[int(c_int)]
                if GLOBAL_CANDIDATE_RE.search(ch):
                    needed_glob.add(ch)
            for cid in cae.SCOPE_102_106:
                if src == cid:
                    continue
                col = cae.ALL_IDS.index(cid)
                for p in np.argsort(-sims_mat[:, col], kind="stable")[:k_take]:
                    needed_keys.add((u_chs[int(ad_cids[p])], cid))

    print(f"   Extracting Natasha for {len(needed_keys)} pairs & {len(needed_glob)} global chunks...", flush=True)
    sa = SyntaxAnalyzer()
    t_nat = time.perf_counter()
    h3_sig = {k: extract_chunk_syntax_signals(sa, k[0], k[1]) for k in sorted(needed_keys)}
    h3_glob = {ch: detect_chunk_global_restriction(sa, ch) for ch in sorted(needed_glob)}
    print(f"   Natasha done in {time.perf_counter()-t_nat:.2f}s", flush=True)

    # 3. Build DataFrames and score all 4 models
    c_cols = ["p_sgd"] + NON_SYNTAX_META_FEATURES + SYNTAX_ONLY_FEATURES
    cb_c_old = CatBoostClassifier(); cb_c_old.load_model(str(ROOT / "models" / "c_syntax" / "catboost.cbm"))
    cb_c_fixed = CatBoostClassifier(); cb_c_fixed.load_model(str(ROOT / "output" / "syntax-link-fix" / "models" / "catboost_C_syntax_fixed.cbm"))
    cb_h3 = CatBoostClassifier(); cb_h3.load_model(str(OUT_MODELS / "catboost_H3_only.cbm"))
    cb_h123 = CatBoostClassifier(); cb_h123.load_model(str(OUT_MODELS / "catboost_H123.cbm"))

    frames_by_cfg = {"C_syntax": {}, "C_syntax_fixed": {}, "H3_retrained": {}, "H1+H2+H3_retrained": {}}
    probs_by_cfg = {"C_syntax": {}, "C_syntax_fixed": {}, "H3_retrained": {}, "H1+H2+H3_retrained": {}}
    preds_by_cfg = {"C_syntax": {}, "C_syntax_fixed": {}, "H3_retrained": {}, "H1+H2+H3_retrained": {}}
    meta_by_split = {}

    for sname, rows, uids_s, off_s, cidx_s, sims_all, ch_vocab, y_mat, src_vec, p_sgd_r in raw_specs:
        meta_by_split[sname] = (rows, y_mat, src_vec)
        df_old = old_frames[sname]
        df_base_fix = build_pair_dataframe(sname, rows, uids_s, off_s, cidx_s, sims_all, ch_vocab, y_mat, src_vec, p_sgd_r,
                                           base_fixed_syn["chunk_sig_cache"], base_fixed_syn["chunk_global_cache"])
        df_h3 = build_pair_dataframe(sname, rows, uids_s, off_s, cidx_s, sims_all, ch_vocab, y_mat, src_vec, p_sgd_r, h3_sig, h3_glob)
        n_rows, n_uids, n_off, n_cidx, n_sims, n_chs, _, _, n_psgd = norm_built[sname]
        df_h123 = build_pair_dataframe(sname, n_rows, n_uids, n_off, n_cidx, n_sims, n_chs, y_mat, src_vec, n_psgd, h3_sig, h3_glob)

        for df_x in (df_old, df_base_fix, df_h3, df_h123):
            for c in SYNTAX_CAT_FEATURES:
                df_x[c] = df_x[c].astype(str)

        for cfg, df_x, model in [
            ("C_syntax", df_old, cb_c_old),
            ("C_syntax_fixed", df_base_fix, cb_c_fixed),
            ("H3_retrained", df_h3, cb_h3),
            ("H1+H2+H3_retrained", df_h123, cb_h123),
        ]:
            p = model.predict_proba(Pool(df_x[c_cols], cat_features=SYNTAX_CAT_FEATURES))[:, 1]
            mat = probs_to_eval_matrix(df_x, p, len(rows), src_vec, cae.SCOPE_102_106)
            probs_by_cfg[cfg][sname] = mat
            _, yp = metrics_summary(y_mat, mat, 0.5, src_vec)
            preds_by_cfg[cfg][sname] = yp

    eval_defs = [
        ("pub_validation", "pub_validation", None),
        ("v3_test", "v3_test", None),
        ("public_test_208", "official_208", None),
        ("public_test_clean", "official_208", clean_mask_208),
    ]

    results = {}
    for ds_name, sname, mask in eval_defs:
        rows_all, y_all, src_all = meta_by_split[sname]
        if mask is not None:
            rows_ev = [r for r, m in zip(rows_all, mask) if m]
            y_ev = y_all[mask]
            src_ev = src_all[mask]
        else:
            rows_ev, y_ev, src_ev = rows_all, y_all, src_all

        results[ds_name] = {}
        yp_base_fix = preds_by_cfg["C_syntax_fixed"][sname][mask] if mask is not None else preds_by_cfg["C_syntax_fixed"][sname]
        m_base_fix, _ = metrics_summary(y_ev, probs_by_cfg["C_syntax_fixed"][sname][mask] if mask is not None else probs_by_cfg["C_syntax_fixed"][sname], 0.5, src_ev)

        for cfg in ("C_syntax", "C_syntax_fixed", "H3_retrained", "H1+H2+H3_retrained"):
            p_mat = probs_by_cfg[cfg][sname][mask] if mask is not None else probs_by_cfg[cfg][sname]
            m, yp = metrics_summary(y_ev, p_mat, 0.5, src_ev)
            lb = length_breakdown(y_ev, p_mat, src_ev, rows_ev)
            entry = {"overall": m, "by_length": lb}
            if cfg in ("H3_retrained", "H1+H2+H3_retrained"):
                entry["vs_C_syntax_fixed"] = diff_summary(m_base_fix, m, y_ev, yp_base_fix, yp, src_ev)
            results[ds_name][cfg] = entry

    print("\n=== HELD-OUT EVALUATION RESULTS (thr=0.5) ===", flush=True)
    for ds_name, block in results.items():
        print(f"\n[{ds_name}]", flush=True)
        for cfg in ("C_syntax", "C_syntax_fixed", "H3_retrained", "H1+H2+H3_retrained"):
            o = block[cfg]["overall"]
            vs = block[cfg].get("vs_C_syntax_fixed")
            vs_s = (
                f" | vs C_syntax_fixed: dF1={vs['delta_micro_f1']:+.4f} FP{vs['delta_fp']:+d} FN{vs['delta_fn']:+d} "
                f"(fixed={vs['fixed_errors']}, broken={vs['broken_errors']})"
                if vs else ""
            )
            print(
                f"  {cfg:20s}: F1={o['micro_f1']:.4f} P={o['precision']:.4f} R={o['recall']:.4f} "
                f"EM={o['exact_match']:.4f} TP={o['tp']} FP={o['fp']} FN={o['fn']}{vs_s}",
                flush=True,
            )

    print("\n3. Running uncached end-to-end CPU inference latency benchmark...", flush=True)
    latency = measure_uncached_latency(query_emb_f32, sgd_heads, cb_c_fixed, cb_h123)
    for cfg_name, blk in latency.items():
        print(f"  [{cfg_name}]", flush=True)
        for lbl, stats in blk.items():
            print(f"    {lbl:11s} ({stats['chars']:5d} chars, {stats['chunks']:3d} chunks): total={stats['median_total_ms']:7.2f} ms | {stats['median_stages_ms']}", flush=True)

    payload = {
        "eval_results": results,
        "uncached_latency_cpu": latency,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (OUT_DIR / "final_metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nSaved output/final-robustness/final_metrics.json in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
