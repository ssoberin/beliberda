#!/usr/bin/env python3
"""Step 1 Audit on train and calibration sets (no test labels or test examples used)."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec

import pickle
import re
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.eval_public_tests_syntax import reconstruct_v2_chunk_map
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.run_syntax_experiment import build_pair_dataframe
from avito_compare.scripts.syntax_features import (
    ACTION_LEMMAS,
    CATEGORY_PATTERNS,
    PARTIAL_MOD_RE,
    SYNTAX_CAT_FEATURES,
    SyntaxAnalyzer,
    _analyze_sentence_for_category,
)

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
    with (ROOT / "output" / "syntax-link-fix" / "cache" / "chunk_syntax_cache_fixed.pkl").open("rb") as f:
        cached_syn = pickle.load(f)
    chunk_sig_cache = cached_syn["chunk_sig_cache"]
    chunk_global_cache = cached_syn["chunk_global_cache"]

    probs_A = {}
    for sname in ("pub_calibration", "v3_validation"):
        feats, _ = cae.build_category_features_for_uids(
            npz[f"uid_{sname}"], offsets_v2, cindices_v2, v2_emb_f32, v2_cat_sims, return_selection=False
        )
        probs_A[sname] = np.column_stack([sgd_heads[cid].predict_proba(feats[cid])[:, 1] for cid in cae.SCOPE_102_106])

    df_cal = pd.concat(
        [
            build_pair_dataframe(
                "pub_calibration", splits["pub_calibration"], npz["uid_pub_calibration"], offsets_v2, cindices_v2,
                v2_cat_sims, v2_chunks, y_cal_5[:n_pub_cal], src_cal[:n_pub_cal], probs_A["pub_calibration"],
                chunk_sig_cache, chunk_global_cache,
            ),
            build_pair_dataframe(
                "v3_validation", splits["v3_validation"], npz["uid_v3_validation"], offsets_v2, cindices_v2,
                v2_cat_sims, v2_chunks, y_cal_5[n_pub_cal:], src_cal[n_pub_cal:], probs_A["v3_validation"],
                chunk_sig_cache, chunk_global_cache,
            ),
        ],
        ignore_index=True,
    )

    cb = CatBoostClassifier()
    cb.load_model(str(ROOT / "output" / "syntax-link-fix" / "models" / "catboost_C_syntax_fixed.cbm"))
    X_cal = df_cal[cb.feature_names_].copy()
    for c in SYNTAX_CAT_FEATURES:
        X_cal[c] = X_cal[c].astype(str)
    df_cal["p_cb"] = cb.predict_proba(X_cal)[:, 1]
    df_cal["pred_cb"] = (df_cal["p_cb"] >= 0.5).astype(int)

    cal_uids_by_split = {
        "pub_calibration": npz["uid_pub_calibration"],
        "v3_validation": npz["uid_v3_validation"],
    }

    sa = SyntaxAnalyzer()
    dup_top3_pairs = 0
    fp_verbs = Counter()
    direct_not_action_tp = Counter()
    direct_not_action_fp = Counter()
    anaphora_partial_triggers = 0
    fp_non_exec_examples = []
    fn_homoglyph_cnt = 0
    MIXED_SCRIPT_RE = re.compile(r"\b(?=[A-Za-zА-Яа-яЁё]*[A-Za-z])(?=[A-Za-zА-Яа-яЁё]*[А-Яа-яЁё])[A-Za-zА-Яа-яЁё]+\b")

    for row in df_cal.itertuples(index=False):
        split = row.split
        r_idx = int(row.row_idx)
        cid = int(row.cat_id)
        uid = int(cal_uids_by_split[split][r_idx])
        ad_row = splits[split][r_idx]
        y = int(row.y_true)
        pred_y = int(row.pred_cb)
        p = float(row.p_cb)

        st, en = int(offsets_v2[uid]), int(offsets_v2[uid + 1])
        ad_cids = cindices_v2[st:en]
        sims_mat = v2_cat_sims[ad_cids]
        k_take = min(cae.TOP_K, len(ad_cids))
        col = cae.ALL_IDS.index(cid)
        top_pos = np.argsort(-sims_mat[:, col], kind="stable")[:k_take]
        top_texts = [v2_chunks[int(ad_cids[pos])] for pos in top_pos]
        if len(top_texts) != len(set(top_texts)):
            dup_top3_pairs += 1

        if y == 1 and pred_y == 0 and MIXED_SCRIPT_RE.search(ad_row["description"]):
            fn_homoglyph_cnt += 1

        if y == 0 and pred_y == 0:
            continue

        for ctext in top_texts:
            sents = sa.parse_text(ctext)
            for i, sent in enumerate(sents):
                prev_has_cat = i > 0 and any(
                    CATEGORY_PATTERNS[cid].match(t.lemma) or CATEGORY_PATTERNS[cid].match(t.text)
                    for t in sents[i - 1].tokens
                )
                res = _analyze_sentence_for_category(sent, cid, allow_anaphora=prev_has_cat)
                if not res["matched"]:
                    continue
                for cl in res["clauses"]:
                    if cl.get("link_kind") == "anaphora_cross_sent" and any(
                        PARTIAL_MOD_RE.match(t.lemma) or PARTIAL_MOD_RE.match(t.text) for t in sent.tokens
                    ):
                        anaphora_partial_triggers += 1
                    if cl["polarity"] == "offered":
                        lemma = cl["pred"].split("(")[-1].rstrip(")")
                        if lemma not in ACTION_LEMMAS:
                            key = f"{lemma}|{cl['link_kind']}|{cl['actor']}"
                            if y == 1:
                                direct_not_action_tp[key] += 1
                            else:
                                direct_not_action_fp[key] += 1
                                if pred_y == 1 and len(fp_non_exec_examples) < 12:
                                    fp_non_exec_examples.append((cid, round(p, 3), cl["pred"], cl["link_kind"], sent.text[:140]))
                    if y == 0 and pred_y == 1 and cl["polarity"] == "offered":
                        lemma = cl["pred"].split("(")[-1].rstrip(")")
                        fp_verbs[f"{lemma}|{cl['link_kind']}"] += 1

    print(f"[calibration] rows={len(df_cal)}, dup_top3_pairs={dup_top3_pairs}, anaphora_partial_triggers={anaphora_partial_triggers}, fn_with_homoglyph={fn_homoglyph_cnt}", flush=True)
    print("\n[calibration] Non-ACTION_LEMMAS promoted to 'offered' on y=1 (TP/FN risk):", flush=True)
    for k, v in direct_not_action_tp.most_common(35):
        print(f"  {k}: {v}")
    print("\n[calibration] Non-ACTION_LEMMAS promoted to 'offered' on y=0 (FP source):", flush=True)
    for k, v in direct_not_action_fp.most_common(25):
        print(f"  {k}: {v}")
    print("\n[calibration] Top FP offered pred lemmas:", flush=True)
    for k, v in fp_verbs.most_common(25):
        print(f"  {k}: {v}")
    print("\n[calibration] Sample FP non-action direct_verb_obj:", flush=True)
    for ex in fp_non_exec_examples:
        print(" ", ex)

    # Synthetic 75k latency benchmark (chunker + natasha top3)
    base_block = "Выполняем сантехнические работы и монтаж отопления. Укладываем плитку и керамогранит. "
    noise_block = "Статья: как выбрать строительные материалы для дачи, советы и инструкции по уходу за домом. " * 20
    long_75k = (base_block + noise_block) * 38
    t0 = time.perf_counter()
    chunks_75k = chunk_ad_fixed(long_75k)
    t_ch = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    for c in chunks_75k[:10]:
        sa.parse_text(c)
    t_nat10 = (time.perf_counter() - t1) * 1000
    print(f"\n[Latency 75k] chars={len(long_75k)} n_chunks={len(chunks_75k)} chunk_ms={t_ch:.2f} natasha_10chunks_ms={t_nat10:.2f}", flush=True)


if __name__ == "__main__":
    main()
