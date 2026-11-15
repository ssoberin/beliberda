#!/usr/bin/env python3
"""Export all Fixed-Chunker+SGD errors, named examples, stratified inspection sample, and polarity probe."""
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed

PUBLIC_DIR = ROOT / "output/ngram-comparison/splits"
V3_CLEAN_DIR = ROOT / "datasets/own_v3_clean"
CACHE_DIR = ROOT / "output/cat-aware-v2/cache"
OUT_DIR = ROOT / "output/cat-aware-v2"

GLOBAL_TURNKEY_RE = re.compile(
    r"(?:только\s+под\s+ключ|работа\w*\s+только\s+(?:под\s+ключ|целиком|комплекс)|"
    r"частичн\w+\s+(?:работ\w*|ремонт\w*)\s+не|с\s+частичк\w+\s+(?:не|помочь\s+не)|"
    r"отдельн\w*\s+(?:работ\w*|услуг\w*|заказ\w*)\s+не\s+(?:выполн\w*|дела\w*|бер[её]м))",
    re.I,
)
PAST_OR_REFUSAL_RE = re.compile(
    r"(?:раньше\s+(?:мы\s+)?(?:делал|занимал|выполн|оказыва|брал)|в\s+прошл\w+|"
    r"больше\s+не\s+|сейчас\s+(?:уже\s+)?не\s+|в\s+данный\s+момент\s+не|"
    r"не\s+выполня\w+|не\s+дела\w+|не\s+оказыва\w+|не\s+бер[её]м|не\s+подключа\w+)",
    re.I,
)


def main():
    z = np.load(CACHE_DIR / "chunks_embeddings.npz", allow_pickle=False)
    chunk_emb_f32 = z["chunk_emb"].astype(np.float32)
    query_emb = z["query_emb"].astype(np.float32)
    offsets = z["ad_chunk_offsets"]
    cindices = z["ad_chunk_indices"]

    # Reconstruct exact uniq_chunks lookup matching build_fixed_chunk_cache.py
    active_splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }
    uniq_ads = []
    ad_to_uid = {}
    for rows in active_splits.values():
        for r in rows:
            txt = r["description"]
            if txt not in ad_to_uid:
                ad_to_uid[txt] = len(uniq_ads)
                uniq_ads.append(txt)

    uniq_chunks = []
    chunk_to_cid = {}
    for txt in uniq_ads:
        for ch in chunk_ad_fixed(txt):
            if ch not in chunk_to_cid:
                chunk_to_cid[ch] = len(uniq_chunks)
                uniq_chunks.append(ch)
    assert len(uniq_chunks) == len(chunk_emb_f32), (len(uniq_chunks), len(chunk_emb_f32))

    # Build flat list of 18 query texts matching CAE ALL_IDS * 3 descriptions
    query_texts = []
    for cid in cae.ALL_IDS:
        query_texts.extend(cae.CATEGORY_DESCRIPTIONS[cid])
    raw_sims = (chunk_emb_f32 @ query_emb.T).astype(np.float32)  # (N_chunks, 18)
    chunk_cat_sims = raw_sims.reshape(len(chunk_emb_f32), len(cae.ALL_IDS), 3).max(axis=2)

    all_errors = []
    named_targets = {
        ("pub_validation", 423),
        ("pub_validation", 654),
        ("v3_test", "own-v3-test-0179"),
        ("v3_test", "own-v3-test-0227"),
        ("v3_test", "own-v3-test-0513"),
    }
    named_examples = {}
    polarity_probe = {}

    for sname in ("pub_validation", "v3_test"):
        rows = active_splits[sname]
        uids = z[f"uid_{sname}"]
        sgd_p = np.load(OUT_DIR / "sgd" / f"{sname}_preds.npz")
        mlp_p = np.load(OUT_DIR / "mlp" / f"{sname}_preds.npz")

        y_true = sgd_p["y_true"]
        src = sgd_p["source_mcId"]
        sgd_probs = sgd_p["probs"]
        sgd_05 = sgd_p["pred_05"]
        sgd_cal = sgd_p["pred_cal"]
        mlp_probs = mlp_p["probs"]
        mlp_05 = mlp_p["pred_05"]

        # Polarity stats for Top-3 chunks containing negation / contractor / turnkey cues
        pol_pairs = []

        for i, r in enumerate(rows):
            uid = int(uids[i])
            assert ad_to_uid[r["description"]] == uid
            start, end = int(offsets[uid]), int(offsets[uid + 1])
            row_cids = cindices[start:end]
            row_chunks = [uniq_chunks[int(c)] for c in row_cids]
            m = len(row_cids)
            padded_slots = max(0, cae.TOP_K - m)
            sims_mat = chunk_cat_sims[row_cids]  # (m, 6)
            raw_mat = raw_sims[row_cids]        # (m, 18)
            k_take = min(cae.TOP_K, m)

            item_id = r.get("item_id") or r.get("id") or (f"pub_val#{i}" if sname == "pub_validation" else f"{sname}#{i}")
            gold_cats = [int(c) for j, c in enumerate(cae.SCOPE_102_106) if int(y_true[i, j]) == 1]
            sgd_cats = [int(c) for j, c in enumerate(cae.SCOPE_102_106) if int(sgd_05[i, j]) == 1]
            mlp_cats = [int(c) for j, c in enumerate(cae.SCOPE_102_106) if int(mlp_05[i, j]) == 1]

            is_named = (sname, i) in named_targets or (sname, item_id) in named_targets
            named_cats_info = {} if is_named else None

            for col, cid in enumerate(cae.SCOPE_102_106):
                sim_col = cae.ALL_IDS.index(cid)
                order = np.argsort(-sims_mat[:, sim_col], kind="stable")[:k_take]
                top3 = []
                top3_has_neg = False
                top3_has_contr = False
                top3_has_turnkey = False
                for slot, lp in enumerate(order):
                    g_cid = int(row_cids[lp])
                    ch_txt = row_chunks[int(lp)]
                    best_q_local = int(np.argmax(raw_mat[lp, sim_col * 3 : (sim_col + 1) * 3]))
                    best_q_idx = sim_col * 3 + best_q_local
                    if cae.NEG_CUE_RE.search(ch_txt) or PAST_OR_REFUSAL_RE.search(ch_txt):
                        top3_has_neg = True
                    if cae.CONTRACTOR_RE.search(ch_txt):
                        top3_has_contr = True
                    if GLOBAL_TURNKEY_RE.search(ch_txt):
                        top3_has_turnkey = True
                    top3.append({
                        "slot": slot,
                        "chunk_pos": int(lp),
                        "cid": g_cid,
                        "chunk_text": ch_txt,
                        "sim": round(float(sims_mat[lp, sim_col]), 4),
                        "best_query_idx": best_q_idx,
                        "best_query_text": query_texts[best_q_idx],
                        "context_before": row_chunks[int(lp) - 1] if int(lp) > 0 else None,
                        "context_after": row_chunks[int(lp) + 1] if int(lp) + 1 < m else None,
                    })

                if is_named:
                    named_cats_info[str(cid)] = {
                        "excluded_source": int(src[i]) == cid,
                        "gold": int(y_true[i, col]),
                        "sgd_prob": round(float(sgd_probs[i, col]), 4),
                        "sgd_pred_05": int(sgd_05[i, col]),
                        "sgd_pred_cal": int(sgd_cal[i, col]),
                        "mlp_prob": round(float(mlp_probs[i, col]), 4),
                        "mlp_pred_05": int(mlp_05[i, col]),
                        "top3": top3,
                    }

                if int(src[i]) == cid:
                    continue

                yt = int(y_true[i, col])
                ps = int(sgd_05[i, col])
                pm = int(mlp_05[i, col])
                if top3_has_neg or top3_has_contr or top3_has_turnkey:
                    pol_pairs.append({
                        "gold": yt,
                        "sgd": ps,
                        "mlp": pm,
                        "top0_sim": top3[0]["sim"],
                        "has_neg": top3_has_neg,
                        "has_contr": top3_has_contr,
                        "has_turnkey": top3_has_turnkey,
                    })

                if ps != yt:
                    err_type = "FP" if (ps == 1 and yt == 0) else "FN"
                    all_errors.append({
                        "split": sname,
                        "row_idx": i,
                        "item_id": item_id,
                        "cat_id": cid,
                        "error_type": err_type,
                        "construction": r.get("construction"),
                        "gold": yt,
                        "pred_sgd_05": ps,
                        "prob_sgd": round(float(sgd_probs[i, col]), 4),
                        "pred_sgd_cal": int(sgd_cal[i, col]),
                        "pred_mlp_05": pm,
                        "prob_mlp": round(float(mlp_probs[i, col]), 4),
                        "source_mcId": int(src[i]),
                        "gold_cats": gold_cats,
                        "pred_sgd_05_cats": sgd_cats,
                        "pred_mlp_05_cats": mlp_cats,
                        "n_chunks": m,
                        "padded_slots": padded_slots,
                        "full_text": r["description"],
                        "all_chunks": row_chunks,
                        "top3": top3,
                    })

            if is_named:
                named_examples[item_id] = {
                    "split": sname,
                    "row_idx": i,
                    "item_id": item_id,
                    "construction": r.get("construction"),
                    "source_mcId": int(src[i]),
                    "gold_cats": gold_cats,
                    "pred_sgd_05_cats": sgd_cats,
                    "pred_mlp_05_cats": mlp_cats,
                    "n_chunks": m,
                    "padded_slots": padded_slots,
                    "full_text": r["description"],
                    "all_chunks": row_chunks,
                    "categories": named_cats_info,
                }

        # Summarize polarity probe for this split
        if pol_pairs:
            g = np.array([x["gold"] for x in pol_pairs])
            s = np.array([x["sgd"] for x in pol_pairs])
            m_arr = np.array([x["mlp"] for x in pol_pairs])
            sim0 = np.array([x["top0_sim"] for x in pol_pairs])
            pos = g == 1
            neg = g == 0
            polarity_probe[sname] = {
                "pairs_with_cue_in_top3": len(pol_pairs),
                "gold_pos": int(pos.sum()),
                "gold_neg": int(neg.sum()),
                "top0_sim_mean_pos": round(float(sim0[pos].mean()), 4) if pos.any() else None,
                "top0_sim_mean_neg": round(float(sim0[neg].mean()), 4) if neg.any() else None,
                "sgd_fp": int(((s == 1) & neg).sum()),
                "sgd_fn": int(((s == 0) & pos).sum()),
                "sgd_fpr": round(float(((s == 1) & neg).sum() / max(neg.sum(), 1)), 4),
                "sgd_fnr": round(float(((s == 0) & pos).sum() / max(pos.sum(), 1)), 4),
                "sgd_acc": round(float((s == g).mean()), 4),
                "mlp_fp": int(((m_arr == 1) & neg).sum()),
                "mlp_fn": int(((m_arr == 0) & pos).sum()),
                "mlp_fpr": round(float(((m_arr == 1) & neg).sum() / max(neg.sum(), 1)), 4),
                "mlp_fnr": round(float(((m_arr == 0) & pos).sum() / max(pos.sum(), 1)), 4),
                "mlp_acc": round(float((m_arr == g).mean()), 4),
                "mlp_rescues_sgd_errors": int(((s != g) & (m_arr == g)).sum()),
                "mlp_breaks_sgd_correct": int(((s == g) & (m_arr != g)).sum()),
            }

    (OUT_DIR / "all_errors_sgd.json").write_text(
        json.dumps(all_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "named_examples_v2.json").write_text(
        json.dumps(named_examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUT_DIR / "polarity_probe.json").write_text(
        json.dumps(polarity_probe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # Build deterministic stratified inspection sample (~72 errors)
    rng = np.random.default_rng(42)
    sample = []
    seen_keys = set()
    # Priority: include errors from named examples first
    for e in all_errors:
        k = (e["split"], e["item_id"], e["cat_id"])
        if e["item_id"] in ("pub_val#423", "pub_val#654", "own-v3-test-0179", "own-v3-test-0227", "own-v3-test-0513"):
            sample.append(e)
            seen_keys.add(k)

    for split_name in ("pub_validation", "v3_test"):
        for etype in ("FP", "FN"):
            pool = [e for e in all_errors if e["split"] == split_name and e["error_type"] == etype and (e["split"], e["item_id"], e["cat_id"]) not in seen_keys]
            # Sort deterministically, then sample up to 18 covering different categories/constructions
            rng.shuffle(pool)
            picked = pool[:18]
            for e in picked:
                sample.append(e)
                seen_keys.add((e["split"], e["item_id"], e["cat_id"]))

    (OUT_DIR / "audit_sample_for_inspection.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_bucket = {}
    for e in all_errors:
        bk = f"{e['split']}_{e['error_type']}"
        by_bucket[bk] = by_bucket.get(bk, 0) + 1
    print("Total SGD(0.5) errors exported:", len(all_errors), by_bucket)
    print("Stratified audit sample size:", len(sample))
    print("Polarity probe:", json.dumps(polarity_probe, ensure_ascii=False))


if __name__ == "__main__":
    main()
