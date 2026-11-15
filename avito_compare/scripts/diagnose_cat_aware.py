#!/usr/bin/env python3
"""Phase 1 diagnostic audit for Category-Aware Qwen3 + SGDClassifier (public_plus_v3 only)."""
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae

PUBLIC_DIR = ROOT / "output/ngram-comparison/splits"
V3_ORIG_DIR = Path("/tmp/innopolis-dataset-v3-review/datasets/own_v3")
V3_CLEAN_DIR = ROOT / "datasets/own_v3_clean"
OFFICIAL_PARQUET = ROOT / "datasets/public/public-test-dataset.parquet"
CACHE_DIR = ROOT / "output/qwen3-category-aware/cache"
OUT_DIR = ROOT / "output/cat-aware-diagnosis"

GLOBAL_RESTRICTION_RE = re.compile(
    r"(?:только\s+под\s+ключ|работаем\s+только\s+под\s+ключ|частичн\w+\s+(?:работ\w*|ремонт\w*)\s+не|"
    r"отдельн\w*\s+(?:работ\w*|услуг\w*)\s+не\s+(?:выполн\w*|дела\w*|бер[её]м)|"
    r"без\s+частичн\w+|мелк\w+\s+(?:ремонт\w*|работ\w*)\s+не\s+(?:выполн\w*|дела\w*|бер[её]м)|"
    r"единичн\w+\s+работ\w*\s+не|только\s+комплексн\w+)",
    re.I,
)


def audit_dropped_lines_old_chunker(text: str):
    """Simulate old chunk_ad header handling and return dropped lines + chunk output."""
    text_s = text.strip()
    if not text_s:
        return {"chunks": [""], "dropped_headers": [], "all_source_lines": []}
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text_s) if p.strip()]
    units = []
    source_lines = []
    for p_idx, p in enumerate(raw_paras):
        if p_idx > 0:
            units.append(("break", ""))
        for raw_ln in p.splitlines():
            ln = re.sub(r"[ \t]+", " ", raw_ln).strip()
            if not ln or re.fullmatch(r"[\W_]{3,}", ln):
                continue
            source_lines.append(ln)
            if cae._is_heading(ln):
                units.append(("header", ln))
            elif cae.BULLET_RE.match(ln):
                units.append(("bullet", ln))
            elif len(ln) <= 240:
                units.append(("prose", ln))
            else:
                for s in cae.SENT_SPLIT.split(ln):
                    s = s.strip()
                    if s:
                        units.append(("prose", s))

    cur_header = None
    header_pending = False
    overwritten_headers = []
    flushed_unconsumed = []
    for kind, val in units:
        if kind == "break":
            if not header_pending:
                cur_header = None
            else:
                # header stays pending across blank line
                pass
            continue
        if kind == "header":
            if header_pending and cur_header is not None:
                overwritten_headers.append(cur_header)
            elif not header_pending and cur_header is not None:
                # actually cur_header was kept after bullets and then overwritten by a new header? Wait:
                # after bullets header_pending=False, cur_header was kept only to attach to subsequent bullets; overwriting isn't dropping UNLESS cur_header was never emitted!
                pass
            cur_header = val
            header_pending = True
            continue
        header_pending = False
    chunks = cae.chunk_ad(text)
    joined_chunks = "\n".join(chunks)
    missing_lines = [ln for ln in source_lines if ln not in joined_chunks]
    return {
        "chunks": chunks,
        "overwritten_pending_headers": overwritten_headers,
        "missing_lines": missing_lines,
        "source_lines": source_lines,
    }


def eval_scope_detailed(y_true, probs, th, src, ids):
    m, yp = cae._eval_binary_scope(y_true, probs, th, src, ids)
    active_mask = np.ones_like(y_true, dtype=bool)
    for i, s in enumerate(src):
        if int(s) in ids:
            active_mask[i, ids.index(int(s))] = False
    ap_by_cat = {}
    for j, cid in enumerate(ids):
        mk = active_mask[:, j]
        ap_by_cat[str(cid)] = float(average_precision_score(y_true[mk, j], probs[mk, j]))
    m["ap_by_category"] = ap_by_cat
    m["macro_ap"] = float(np.mean(list(ap_by_cat.values())))
    return m, yp


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits_orig = cae.collect_all_splits(PUBLIC_DIR, V3_ORIG_DIR, OFFICIAL_PARQUET)
    v3_clean_train = cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl")
    clean_train_desc = [r["description"] for r in v3_clean_train]

    # Verify clean validation/test match orig byte-for-byte
    assert [r["description"] for r in cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl")] == [
        r["description"] for r in splits_orig["v3_test"]
    ]
    assert [r["description"] for r in cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl")] == [
        r["description"] for r in splits_orig["v3_validation"]
    ]

    chunk_npz = np.load(CACHE_DIR / "chunks_embeddings.npz")
    chunk_emb_f32 = chunk_npz["chunk_emb"].astype(np.float32)
    query_emb_f32 = chunk_npz["query_emb"].astype(np.float32)
    offsets = chunk_npz["ad_chunk_offsets"]
    cindices = chunk_npz["ad_chunk_indices"]
    uniq_chunks = cae._build_unique_chunk_lookup(splits_orig)
    assert len(uniq_chunks) == len(chunk_emb_f32)

    raw_sims = chunk_emb_f32 @ query_emb_f32.T
    chunk_cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    # Build map from v3_orig train descriptions to their uid in chunk_npz
    orig_v3_tr_uids = chunk_npz["uid_v3_train"]
    orig_desc_to_uid = {r["description"]: int(uid) for r, uid in zip(splits_orig["v3_train"], orig_v3_tr_uids)}
    clean_v3_tr_uids = np.asarray([orig_desc_to_uid[r["description"]] for r in v3_clean_train], dtype=np.int32)

    split_feats = {}
    split_sel = {}
    for sname in ("pub_train", "pub_calibration", "pub_validation", "v3_train", "v3_validation", "v3_test"):
        feats, sel = cae.build_category_features_for_uids(
            chunk_npz[f"uid_{sname}"], offsets, cindices, chunk_emb_f32, chunk_cat_sims, return_selection=True
        )
        split_feats[sname] = feats
        split_sel[sname] = sel

    feats_v3_clean_tr, _ = cae.build_category_features_for_uids(
        clean_v3_tr_uids, offsets, cindices, chunk_emb_f32, chunk_cat_sims, return_selection=False
    )
    split_feats["v3_clean_train"] = feats_v3_clean_tr

    # 1. Check Old Chunker + SGD on public_plus_v3 (orig vs clean v3_train)
    dataset_variants = {}
    for vtag, v3_tr_name, v3_tr_rows in [
        ("orig_v3_3000", "v3_train", splits_orig["v3_train"]),
        ("clean_v3_2921", "v3_clean_train", v3_clean_train),
    ]:
        tr_rows = splits_orig["pub_train"] + v3_tr_rows
        cal_rows = splits_orig["pub_calibration"] + splits_orig["v3_validation"]
        y_tr_all, _ = cae._rows_xy_for_ids(tr_rows, cae.ALL_IDS, use_eval=False)
        y_cal_all, src_cal_all = cae._rows_xy_for_ids(cal_rows, cae.ALL_IDS, use_eval=True)
        has101_tr = np.asarray([r["has_101_label"] for r in tr_rows], dtype=bool)
        has101_cal = np.asarray([r["has_101_label"] for r in cal_rows], dtype=bool)

        heads = {}
        head_info = {}
        ths_cal = []
        for j, cid in enumerate(cae.ALL_IDS):
            X_tr_c = np.vstack([split_feats["pub_train"][cid], split_feats[v3_tr_name][cid]])
            X_cal_c = np.vstack([split_feats["pub_calibration"][cid], split_feats["v3_validation"][cid]])
            mask_tr = has101_tr if cid == 101 else np.ones(len(tr_rows), dtype=bool)
            clf_c, best_alpha, full_tr_f1, alpha_diag = cae._fit_cv_binary_head(X_tr_c[mask_tr], y_tr_all[mask_tr, j])
            heads[cid] = clf_c
            p_cal_c = clf_c.predict_proba(X_cal_c)[:, 1]
            mask_cal = (src_cal_all != cid) & (has101_cal if cid == 101 else True)
            grid = np.arange(0.20, 0.81, 0.025)
            best_th = float(max(grid, key=lambda t: f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= t, zero_division=0)))
            ths_cal.append(best_th)
            head_info[str(cid)] = {
                "alpha": best_alpha,
                "train_f1_05": round(full_tr_f1, 6),
                "cv_oof_f1": alpha_diag[str(best_alpha)]["oof_f1"],
                "cal_f1_05": round(float(f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= 0.5, zero_division=0)), 6),
                "cal_th": best_th,
            }

        eval_out = {}
        preds_cache = {}
        for sname in ("pub_validation", "v3_test"):
            rows = splits_orig[sname]
            y_ev_5, src_ev = cae._rows_xy_for_ids(rows, cae.SCOPE_102_106, use_eval=True)
            probs_5 = np.column_stack([heads[cid].predict_proba(split_feats[sname][cid])[:, 1] for cid in cae.SCOPE_102_106])
            m05, yp05 = eval_scope_detailed(y_ev_5, probs_5, np.full(5, 0.5), src_ev, cae.SCOPE_102_106)
            mcal, ypcal = eval_scope_detailed(y_ev_5, probs_5, np.asarray(ths_cal[1:]), src_ev, cae.SCOPE_102_106)
            eval_out[sname] = {"th_05": m05, "th_cal": mcal}
            preds_cache[sname] = (y_ev_5, src_ev, probs_5, yp05)
        dataset_variants[vtag] = {"heads": head_info, "eval": eval_out, "preds": preds_cache}

    # 2. Chunker dropped lines & short lines audit across splits
    chunker_audit = {}
    for sname in ("pub_train", "pub_validation", "v3_test"):
        rows = v3_clean_train if sname == "v3_clean_train" else splits_orig[sname]
        n_missing = 0
        total_missing_lines = 0
        n_single_chunk = 0
        n_two_chunks = 0
        sample_dropped = []
        for idx, r in enumerate(rows):
            res = audit_dropped_lines_old_chunker(r["description"])
            if len(res["chunks"]) == 1:
                n_single_chunk += 1
            elif len(res["chunks"]) == 2:
                n_two_chunks += 1
            if res["missing_lines"]:
                n_missing += 1
                total_missing_lines += len(res["missing_lines"])
                if len(sample_dropped) < 8:
                    sample_dropped.append({
                        "idx": idx,
                        "item_id": r.get("item_id"),
                        "n_source_lines": len(res["source_lines"]),
                        "n_missing": len(res["missing_lines"]),
                        "missing_preview": res["missing_lines"][:10],
                        "chunks": res["chunks"],
                    })
        chunker_audit[sname] = {
            "rows": len(rows),
            "ads_with_missing_lines": n_missing,
            "total_missing_lines": total_missing_lines,
            "n_chunks_eq_1": n_single_chunk,
            "n_chunks_eq_2": n_two_chunks,
            "samples": sample_dropped,
        }
    # also check v3_clean_train
    res_clean = {"rows": len(v3_clean_train), "ads_with_missing_lines": 0, "total_missing_lines": 0, "n_chunks_eq_1": 0, "n_chunks_eq_2": 0}
    for r in v3_clean_train:
        a = audit_dropped_lines_old_chunker(r["description"])
        if len(a["chunks"]) == 1:
            res_clean["n_chunks_eq_1"] += 1
        elif len(a["chunks"]) == 2:
            res_clean["n_chunks_eq_2"] += 1
        if a["missing_lines"]:
            res_clean["ads_with_missing_lines"] += 1
            res_clean["total_missing_lines"] += len(a["missing_lines"])
    chunker_audit["v3_clean_train"] = res_clean

    # 3. Zero-padding & Global context audit on clean baseline
    padding_audit = {}
    global_ctx_audit = {}
    for sname in ("pub_validation", "v3_test"):
        rows = splits_orig[sname]
        y_ev_5, src_ev, probs_5, yp05 = dataset_variants["clean_v3_2921"]["preds"][sname]
        uids = chunk_npz[f"uid_{sname}"]
        n_chunks_arr = np.asarray([int(offsets[u + 1] - offsets[u]) for u in uids])
        pad_buckets = {}
        for label, mask in [
            ("n_chunks_1", n_chunks_arr == 1),
            ("n_chunks_2", n_chunks_arr == 2),
            ("n_chunks_ge_3", n_chunks_arr >= 3),
        ]:
            if int(mask.sum()) == 0:
                continue
            mb, _ = eval_scope_detailed(y_ev_5[mask], probs_5[mask], np.full(5, 0.5), src_ev[mask], cae.SCOPE_102_106)
            pad_buckets[label] = {
                "rows": int(mask.sum()),
                "micro_f1": round(mb["micro_f1"], 4),
                "fp": mb["confusion_counts"]["fp"],
                "fn": mb["confusion_counts"]["fn"],
                "tp": mb["confusion_counts"]["tp"],
                "exact_match": round(mb["exact_set_accuracy"], 4),
                "ads_with_missing_lines_in_bucket": sum(1 for k in np.where(mask)[0] if audit_dropped_lines_old_chunker(rows[k]["description"])["missing_lines"]),
            }
        padding_audit[sname] = pad_buckets

        # Global context restriction audit
        g_rows = [i for i, r in enumerate(rows) if GLOBAL_RESTRICTION_RE.search(r["description"])]
        fp_global = 0
        pairs_global_neg = 0
        missed_by_top3 = 0
        for i in g_rows:
            for j, cid in enumerate(cae.SCOPE_102_106):
                if int(src_ev[i]) == cid or y_ev_5[i, j] != 0:
                    continue
                pairs_global_neg += 1
                top3_texts = [uniq_chunks[cidx] for cidx, _ in split_sel[sname][cid][i]]
                has_global_in_top3 = any(GLOBAL_RESTRICTION_RE.search(t) for t in top3_texts)
                if not has_global_in_top3:
                    missed_by_top3 += 1
                if yp05[i, j] == 1:
                    fp_global += 1
        global_ctx_audit[sname] = {
            "ads_with_global_restriction_cue": len(g_rows),
            "neg_active_pairs_in_these_ads": pairs_global_neg,
            "top3_misses_global_cue_pairs": missed_by_top3,
            "fp_on_these_ads_05": fp_global,
            "sample_indices": [(i, rows[i].get("item_id")) for i in g_rows[:10]],
        }

    # 4. Inspect the 5 named audit examples
    named_targets = [
        ("pub_validation", 423),
        ("pub_validation", 654),
        ("v3_test", next(i for i, r in enumerate(splits_orig["v3_test"]) if r.get("item_id") == "own-v3-test-0179")),
        ("v3_test", next(i for i, r in enumerate(splits_orig["v3_test"]) if r.get("item_id") == "own-v3-test-0227")),
        ("v3_test", next(i for i, r in enumerate(splits_orig["v3_test"]) if r.get("item_id") == "own-v3-test-0513")),
    ]
    named_examples = []
    for sname, idx in named_targets:
        r = splits_orig[sname][idx]
        uid = int(chunk_npz[f"uid_{sname}"][idx])
        cids_all = cindices[int(offsets[uid]) : int(offsets[uid + 1])]
        y_ev_5, src_ev, probs_5, yp05 = dataset_variants["clean_v3_2921"]["preds"][sname]
        y_orig_5, _, probs_orig_5, yp05_orig = dataset_variants["orig_v3_3000"]["preds"][sname]
        top3_by_cat = {}
        for cid in cae.SCOPE_102_106:
            top3_by_cat[str(cid)] = [
                {
                    "pos": int(np.where(cids_all == cidx)[0][0]) if cidx in cids_all else -1,
                    "sim": sim,
                    "text": uniq_chunks[cidx],
                }
                for cidx, sim in split_sel[sname][cid][idx]
            ]
        named_examples.append({
            "split": sname,
            "index": idx,
            "item_id": r.get("item_id"),
            "source_mcId": int(r["mcId"]),
            "gold_102_106": [cae.SCOPE_102_106[j] for j in range(5) if y_ev_5[idx, j] == 1],
            "orig_baseline_pred_05": [cae.SCOPE_102_106[j] for j in range(5) if yp05_orig[idx, j] == 1],
            "clean_baseline_pred_05": [cae.SCOPE_102_106[j] for j in range(5) if yp05[idx, j] == 1],
            "clean_baseline_probs": {str(cae.SCOPE_102_106[j]): round(float(probs_5[idx, j]), 4) for j in range(5)},
            "all_old_chunks": [uniq_chunks[int(c)] for c in cids_all],
            "top3_by_cat": top3_by_cat,
            "description": r["description"],
        })

    # strip heavy arrays before saving JSON
    summary_json = {
        "baseline_variants": {
            k: {"heads": v["heads"], "eval": v["eval"]} for k, v in dataset_variants.items()
        },
        "chunker_dropped_lines_audit": chunker_audit,
        "zero_padding_audit": padding_audit,
        "global_context_audit": global_ctx_audit,
        "named_examples": named_examples,
    }
    (OUT_DIR / "phase1_baseline_diagnosis.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== BASELINE VARIANTS (public_plus_v3, thr=0.5) ===")
    for k, v in dataset_variants.items():
        for sp in ("pub_validation", "v3_test"):
            m = v["eval"][sp]["th_05"]
            cc = m["confusion_counts"]
            print(f"{k:15s} | {sp:14s} | Micro-F1={m['micro_f1']:.4f} | FP={cc['fp']} FN={cc['fn']} TP={cc['tp']} | EM={m['exact_set_accuracy']:.4f} | Macro-AP={m['macro_ap']:.4f}")
    print("\n=== CHUNKER DROPPED LINES & SHORT CHUNKS ===")
    for k, v in chunker_audit.items():
        print(f"{k:15s} | rows={v['rows']} | ads_missing_lines={v['ads_with_missing_lines']} (total={v['total_missing_lines']}) | n_chunks==1: {v['n_chunks_eq_1']} | n_chunks==2: {v['n_chunks_eq_2']}")
    print("\n=== ZERO PADDING BUCKETS (clean_v3_2921, thr=0.5) ===")
    for sp, bk in padding_audit.items():
        print(f"[{sp}]")
        for bname, bv in bk.items():
            print(f"  {bname:14s}: rows={bv['rows']:3d} Micro-F1={bv['micro_f1']:.4f} FP={bv['fp']:3d} FN={bv['fn']:3d} missing_lines_ads={bv['ads_with_missing_lines_in_bucket']}")
    print("\n=== GLOBAL CONTEXT CUE AUDIT (clean_v3_2921, thr=0.5) ===")
    for sp, gv in global_ctx_audit.items():
        print(f"  {sp}: {gv}")


if __name__ == "__main__":
    main()
