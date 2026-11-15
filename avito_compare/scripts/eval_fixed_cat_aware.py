#!/usr/bin/env python3
"""Evaluate Fixed Segmentation + SGDClassifier and Fixed Segmentation + Diagnostic MLP (public_plus_v3 only)."""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed

PUBLIC_DIR = ROOT / "output/ngram-comparison/splits"
V3_CLEAN_DIR = ROOT / "datasets/own_v3_clean"
CACHE_DIR = ROOT / "output/cat-aware-v2/cache"
OUT_DIR = ROOT / "output/cat-aware-v2"

MLP_CONFIG = {
    "hidden_dim": 128,
    "activation": "ReLU",
    "dropout": 0.2,
    "loss": "BCEWithLogitsLoss",
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-2,
    "batch_size": 128,
    "max_epochs": 40,
    "early_stopping_patience": 5,
    "early_stopping_metric": "calibration_bce_loss",
    "seed": 42,
}

GLOBAL_TURNKEY_RE = re.compile(
    r"(?:только\s+под\s+ключ|работа\w*\s+только\s+(?:под\s+ключ|целиком|комплекс)|"
    r"частичн\w+\s+(?:работ\w*|ремонт\w*)\s+не|с\s+частичк\w+\s+(?:не|помочь\s+не)|"
    r"отдельн\w*\s+(?:работ\w*|услуг\w*|заказ\w*)\s+не\s+(?:выполн\w*|дела\w*|бер[её]м))",
    re.I,
)


class DiagnosticMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_detailed_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, sources: np.ndarray, ids_tuple: tuple[int, ...]):
    yp = (probs >= thresholds).astype(np.int8)
    active_mask = np.ones_like(y_true, dtype=bool)
    for i, src in enumerate(sources):
        if int(src) in ids_tuple:
            j_src = ids_tuple.index(int(src))
            yp[i, j_src] = 0
            active_mask[i, j_src] = False

    tp = int(np.sum((yp == 1) & (y_true == 1) & active_mask))
    fp = int(np.sum((yp == 1) & (y_true == 0) & active_mask))
    fn = int(np.sum((yp == 0) & (y_true == 1) & active_mask))
    tn = int(np.sum((yp == 0) & (y_true == 0) & active_mask))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    fnr = fn / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (fp + tn) if (fp + tn) else 0.0

    per_cat = {}
    for j, cid in enumerate(ids_tuple):
        mk = active_mask[:, j]
        yt_c, yp_c, pr_c = y_true[mk, j], yp[mk, j], probs[mk, j]
        tp_c = int(np.sum((yp_c == 1) & (yt_c == 1)))
        fp_c = int(np.sum((yp_c == 1) & (yt_c == 0)))
        fn_c = int(np.sum((yp_c == 0) & (yt_c == 1)))
        tn_c = int(np.sum((yp_c == 0) & (yt_c == 0)))
        p_c = tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0
        r_c = tp_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0
        f1_c = 2 * tp_c / (2 * tp_c + fp_c + fn_c) if (2 * tp_c + fp_c + fn_c) else 0.0
        ap_c = float(average_precision_score(yt_c, pr_c)) if len(np.unique(yt_c)) > 1 else 0.0
        per_cat[str(cid)] = {
            "support": int(yt_c.sum()),
            "tp": tp_c,
            "fp": fp_c,
            "fn": fn_c,
            "tn": tn_c,
            "precision": round(p_c, 6),
            "recall": round(r_c, 6),
            "f1": round(f1_c, 6),
            "fpr": round(fp_c / (fp_c + tn_c) if (fp_c + tn_c) else 0.0, 6),
            "fnr": round(fn_c / (tp_c + fn_c) if (tp_c + fn_c) else 0.0, 6),
            "specificity": round(tn_c / (fp_c + tn_c) if (fp_c + tn_c) else 0.0, 6),
            "ap": round(ap_c, 6),
        }

    macro_f1 = float(np.mean([v["f1"] for v in per_cat.values()]))
    macro_ap = float(np.mean([v["ap"] for v in per_cat.values()]))
    micro_ap = float(average_precision_score(y_true[active_mask], probs[active_mask]))

    return {
        "rows": int(len(y_true)),
        "micro_f1": round(micro_f1, 6),
        "macro_f1": round(macro_f1, 6),
        "micro_precision": round(prec, 6),
        "micro_recall": round(rec, 6),
        "fpr": round(fpr, 6),
        "fnr": round(fnr, 6),
        "specificity": round(specificity, 6),
        "exact_match": round(float(np.mean(np.all(yp == y_true, axis=1))), 6),
        "micro_ap": round(micro_ap, 6),
        "macro_ap": round(macro_ap, 6),
        "empty_predictions": int(np.sum(yp.sum(axis=1) == 0)),
        "empty_gold": int(np.sum(y_true.sum(axis=1) == 0)),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "per_category": per_cat,
    }, yp


def build_split_slices(sname: str, rows: list[dict], n_chunks_arr: np.ndarray) -> dict[str, np.ndarray]:
    base = {
        "all": np.ones(len(rows), dtype=bool),
        "short_lt_300": np.asarray([len(r["description"]) < 300 for r in rows], dtype=bool),
        "n_chunks_1": n_chunks_arr == 1,
        "n_chunks_2": n_chunks_arr == 2,
        "n_chunks_ge_3": n_chunks_arr >= 3,
    }
    if sname == "pub_validation":
        base.update({
            "long_ads_ge_1467": np.asarray([len(r["description"]) >= 1467 for r in rows], dtype=bool),
            "negations": np.asarray([bool(cae.NEG_CUE_RE.search(r["description"])) for r in rows], dtype=bool),
            "service_lists": np.asarray(
                [len(re.findall(r"(?:^|\n)\s*(?:[•\-–—✔✅🔹🔸▪◾●➤]|\d+[.)])\s*", r["description"])) >= 4 for r in rows],
                dtype=bool,
            ),
            "global_turnkey_only": np.asarray([bool(GLOBAL_TURNKEY_RE.search(r["description"])) for r in rows], dtype=bool),
        })
    elif sname == "v3_test":
        for c in ("direct", "condition", "faq", "not_only", "past", "price", "team"):
            base[f"c_{c}"] = np.asarray([r.get("construction") == c for r in rows], dtype=bool)
        base["negation_constructions"] = np.asarray(
            [r.get("construction") in ("not_only", "past", "price", "team") for r in rows], dtype=bool
        )
    return base


def fit_mlp_binary_head(X_tr: np.ndarray, y_tr: np.ndarray, X_cal: np.ndarray, y_cal: np.ndarray, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    gen = torch.Generator().manual_seed(seed)

    model = DiagnosticMLP(X_tr.shape[1], hidden_dim=MLP_CONFIG["hidden_dim"], dropout=MLP_CONFIG["dropout"])
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=MLP_CONFIG["lr"], weight_decay=MLP_CONFIG["weight_decay"])

    ds_tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr.astype(np.float32)))
    loader = DataLoader(ds_tr, batch_size=MLP_CONFIG["batch_size"], shuffle=True, generator=gen)

    X_cal_t = torch.from_numpy(X_cal).float()
    y_cal_t = torch.from_numpy(y_cal.astype(np.float32))

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_imp = 0
    history = []

    for epoch in range(1, MLP_CONFIG["max_epochs"] + 1):
        model.train()
        tr_loss_sum = 0.0
        n_tr = 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_loss_sum += float(loss.item()) * len(xb)
            n_tr += len(xb)

        model.eval()
        with torch.no_grad():
            cal_logits = model(X_cal_t)
            cal_loss = float(criterion(cal_logits, y_cal_t).item())
            cal_probs = torch.sigmoid(cal_logits).numpy()
            cal_f1 = float(f1_score(y_cal, cal_probs >= 0.5, zero_division=0))

        history.append({"epoch": epoch, "train_bce": round(tr_loss_sum / n_tr, 5), "cal_bce": round(cal_loss, 5), "cal_f1_05": round(cal_f1, 5)})
        if cal_loss < best_loss - 1e-5:
            best_loss = cal_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= MLP_CONFIG["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model, {"best_epoch": best_epoch, "best_cal_bce": round(best_loss, 5), "epochs_run": len(history), "history_tail": history[-5:]}


@torch.no_grad()
def predict_mlp(model: DiagnosticMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    logits = model(torch.from_numpy(X).float())
    return torch.sigmoid(logits).numpy().astype(np.float64)


def build_features_repeat_pad(uids, offsets, cindices, chunk_emb_f32, chunk_cat_sims):
    """Diagnostic variant: pad missing slots with slot-0 embedding (sim=0, mask=0) vs zero-padding."""
    n = len(uids)
    dim = chunk_emb_f32.shape[1]
    feat_dim = cae.TOP_K * dim + 2 * cae.TOP_K
    feats = {cid: np.zeros((n, feat_dim), dtype=np.float32) for cid in cae.ALL_IDS}
    for row_idx, uid in enumerate(uids):
        start, end = int(offsets[uid]), int(offsets[uid + 1])
        cids = ad_chunk_indices[start:end] if False else cindices[start:end]
        m = len(cids)
        sims_mat = chunk_cat_sims[cids]
        embs_mat = chunk_emb_f32[cids]
        k_take = min(cae.TOP_K, m)
        for col, cid in enumerate(cae.ALL_IDS):
            order = np.argsort(-sims_mat[:, col], kind="stable")[:k_take]
            vec = feats[cid][row_idx]
            for slot in range(cae.TOP_K):
                if slot < k_take:
                    lp = order[slot]
                    vec[slot * dim : (slot + 1) * dim] = embs_mat[lp]
                    vec[cae.TOP_K * dim + slot] = float(sims_mat[lp, col])
                    vec[cae.TOP_K * dim + cae.TOP_K + slot] = 1.0
                else:
                    vec[slot * dim : (slot + 1) * dim] = embs_mat[order[-1]]
                    vec[cae.TOP_K * dim + slot] = 0.0
                    vec[cae.TOP_K * dim + cae.TOP_K + slot] = 0.0
    return feats


def pairwise_agreement(y_true, yp_a, yp_b, src, ids):
    both_ok = a_only = b_only = both_err = fn_b_saves_a = fn_a_total = fp_b_blocks_a = fp_a_total = 0
    for i in range(len(y_true)):
        for j, cid in enumerate(ids):
            if int(src[i]) == cid:
                continue
            yt, pa, pb = int(y_true[i, j]), int(yp_a[i, j]), int(yp_b[i, j])
            if pa == yt and pb == yt:
                both_ok += 1
            elif pa != yt and pb == yt:
                a_only += 1
            elif pa == yt and pb != yt:
                b_only += 1
            else:
                both_err += 1
            if yt == 1 and pa == 0:
                fn_a_total += 1
                if pb == 1:
                    fn_b_saves_a += 1
            if yt == 0 and pa == 1:
                fp_a_total += 1
                if pb == 0:
                    fp_b_blocks_a += 1
    tot = both_ok + a_only + b_only + both_err
    return {
        "active_pairs": tot,
        "both_correct": both_ok,
        "only_sgd_wrong": a_only,
        "only_mlp_wrong": b_only,
        "both_wrong": both_err,
        "disagreement_rate": round((a_only + b_only) / tot, 4),
        "fn_mlp_rescues_sgd": f"{fn_b_saves_a}/{fn_a_total}",
        "fp_mlp_blocks_sgd": f"{fp_b_blocks_a}/{fp_a_total}",
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sgd_dir = OUT_DIR / "sgd"
    mlp_dir = OUT_DIR / "mlp"
    sgd_dir.mkdir(exist_ok=True)
    mlp_dir.mkdir(exist_ok=True)

    splits = {
        "pub_train": cae.load_public_rows(PUBLIC_DIR / "train.jsonl"),
        "pub_calibration": cae.load_public_rows(PUBLIC_DIR / "calibration.jsonl"),
        "pub_validation": cae.load_public_rows(PUBLIC_DIR / "validation.jsonl"),
        "v3_train": cae.load_v3_rows_full(V3_CLEAN_DIR / "train.jsonl"),
        "v3_validation": cae.load_v3_rows_full(V3_CLEAN_DIR / "validation.jsonl"),
        "v3_test": cae.load_v3_rows_full(V3_CLEAN_DIR / "test.jsonl"),
    }

    npz = np.load(CACHE_DIR / "chunks_embeddings.npz")
    chunk_emb_f32 = npz["chunk_emb"].astype(np.float32)
    query_emb_f32 = npz["query_emb"].astype(np.float32)
    offsets = npz["ad_chunk_offsets"]
    cindices = npz["ad_chunk_indices"]

    raw_sims = chunk_emb_f32 @ query_emb_f32.T
    chunk_cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(cae.ALL_IDS))])

    split_feats = {}
    split_sel = {}
    split_nchunks = {}
    for sname in splits:
        uids = npz[f"uid_{sname}"]
        feats, sel = cae.build_category_features_for_uids(uids, offsets, cindices, chunk_emb_f32, chunk_cat_sims, return_selection=True)
        split_feats[sname] = feats
        split_sel[sname] = sel
        split_nchunks[sname] = np.asarray([int(offsets[u + 1] - offsets[u]) for u in uids], dtype=np.int32)

    tr_names, cal_names = ["pub_train", "v3_train"], ["pub_calibration", "v3_validation"]
    tr_rows = [r for s in tr_names for r in splits[s]]
    cal_rows = [r for s in cal_names for r in splits[s]]

    y_tr_all, src_tr_all = cae._rows_xy_for_ids(tr_rows, cae.ALL_IDS, use_eval=False)
    y_cal_all, src_cal_all = cae._rows_xy_for_ids(cal_rows, cae.ALL_IDS, use_eval=True)
    has101_tr = np.asarray([r["has_101_label"] for r in tr_rows], dtype=bool)
    has101_cal = np.asarray([r["has_101_label"] for r in cal_rows], dtype=bool)

    # --- A. Train Fixed Chunker + SGDClassifier (exact baseline protocol) ---
    sgd_heads = {}
    sgd_diag = {}
    sgd_cal_ths = []
    t0 = time.perf_counter()
    for j, cid in enumerate(cae.ALL_IDS):
        X_tr_c = np.vstack([split_feats[s][cid] for s in tr_names])
        X_cal_c = np.vstack([split_feats[s][cid] for s in cal_names])
        mask_tr = has101_tr if cid == 101 else np.ones(len(tr_rows), dtype=bool)
        clf_c, best_alpha, full_tr_f1, alpha_diag = cae._fit_cv_binary_head(X_tr_c[mask_tr], y_tr_all[mask_tr, j])
        sgd_heads[cid] = clf_c

        p_tr_c = clf_c.predict_proba(X_tr_c[mask_tr])[:, 1]
        p_cal_c = clf_c.predict_proba(X_cal_c)[:, 1]
        mask_cal = (src_cal_all != cid) & (has101_cal if cid == 101 else True)
        grid = np.arange(0.20, 0.81, 0.025)
        best_th = float(max(grid, key=lambda t: f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= t, zero_division=0)))
        sgd_cal_ths.append(best_th)
        cal_f1_05 = float(f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= 0.5, zero_division=0))
        sgd_diag[str(cid)] = {
            "alpha": best_alpha,
            "train_f1_05": round(full_tr_f1, 6),
            "cv_oof_f1": alpha_diag[str(best_alpha)]["oof_f1"],
            "cal_f1_05": round(cal_f1_05, 6),
            "train_minus_cal05_gap": round(full_tr_f1 - cal_f1_05, 6),
            "cal_th": round(best_th, 4),
        }
    sgd_fit_s = time.perf_counter() - t0
    joblib.dump({"heads": sgd_heads, "thresholds_101_106": sgd_cal_ths, "all_ids": cae.ALL_IDS, "diag": sgd_diag}, sgd_dir / "sgd_heads.joblib")

    # --- B. Train Fixed Chunker + Diagnostic MLP (identical frozen features) ---
    torch.set_num_threads(4)
    mlp_heads = {}
    mlp_diag = {}
    mlp_cal_ths = []
    t1 = time.perf_counter()
    for j, cid in enumerate(cae.ALL_IDS):
        X_tr_c = np.vstack([split_feats[s][cid] for s in tr_names])
        X_cal_c = np.vstack([split_feats[s][cid] for s in cal_names])
        mask_tr = has101_tr if cid == 101 else np.ones(len(tr_rows), dtype=bool)
        mask_cal = (src_cal_all != cid) & (has101_cal if cid == 101 else True)

        model_c, m_info = fit_mlp_binary_head(
            X_tr_c[mask_tr],
            y_tr_all[mask_tr, j],
            X_cal_c[mask_cal],
            y_cal_all[mask_cal, j],
            seed=MLP_CONFIG["seed"] + j,
        )
        mlp_heads[cid] = model_c
        p_tr_c = predict_mlp(model_c, X_tr_c[mask_tr])
        p_cal_c = predict_mlp(model_c, X_cal_c)
        tr_f1_05 = float(f1_score(y_tr_all[mask_tr, j], p_tr_c >= 0.5, zero_division=0))
        cal_f1_05 = float(f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= 0.5, zero_division=0))
        grid = np.arange(0.20, 0.81, 0.025)
        best_th = float(max(grid, key=lambda t: f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= t, zero_division=0)))
        mlp_cal_ths.append(best_th)
        mlp_diag[str(cid)] = {
            **m_info,
            "train_f1_05": round(tr_f1_05, 6),
            "cal_f1_05": round(cal_f1_05, 6),
            "train_minus_cal05_gap": round(tr_f1_05 - cal_f1_05, 6),
            "cal_th": round(best_th, 4),
        }
    mlp_fit_s = time.perf_counter() - t1
    torch.save({str(cid): m.state_dict() for cid, m in mlp_heads.items()}, mlp_dir / "mlp_heads.pt")
    (mlp_dir / "mlp_config.json").write_text(json.dumps({"config": MLP_CONFIG, "head_diag": mlp_diag}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # --- Evaluate both models separately on pub_validation and v3_test (plus cal split for overfitting check) ---
    results = {"sgd_fit_seconds": sgd_fit_s, "mlp_fit_seconds": mlp_fit_s, "sgd_heads": sgd_diag, "mlp_heads": mlp_diag, "splits": {}}
    th05_5 = np.full(5, 0.5)
    sgd_cal_5 = np.asarray(sgd_cal_ths[1:], dtype=float)
    mlp_cal_5 = np.asarray(mlp_cal_ths[1:], dtype=float)

    for sname in ("pub_calibration", "v3_validation", "pub_validation", "v3_test"):
        rows = splits[sname]
        y_ev_5, src_ev = cae._rows_xy_for_ids(rows, cae.SCOPE_102_106, use_eval=True)
        probs_sgd = np.column_stack([sgd_heads[cid].predict_proba(split_feats[sname][cid])[:, 1] for cid in cae.SCOPE_102_106])
        probs_mlp = np.column_stack([predict_mlp(mlp_heads[cid], split_feats[sname][cid]) for cid in cae.SCOPE_102_106])

        m_sgd_05, yp_sgd_05 = compute_detailed_metrics(y_ev_5, probs_sgd, th05_5, src_ev, cae.SCOPE_102_106)
        m_sgd_cal, yp_sgd_cal = compute_detailed_metrics(y_ev_5, probs_sgd, sgd_cal_5, src_ev, cae.SCOPE_102_106)
        m_mlp_05, yp_mlp_05 = compute_detailed_metrics(y_ev_5, probs_mlp, th05_5, src_ev, cae.SCOPE_102_106)
        m_mlp_cal, yp_mlp_cal = compute_detailed_metrics(y_ev_5, probs_mlp, mlp_cal_5, src_ev, cae.SCOPE_102_106)

        slice_eval = {}
        if sname in ("pub_validation", "v3_test"):
            slices = build_split_slices(sname, rows, split_nchunks[sname])
            for sl_name, mk in slices.items():
                if int(mk.sum()) == 0:
                    continue
                ms_05, _ = compute_detailed_metrics(y_ev_5[mk], probs_sgd[mk], th05_5, src_ev[mk], cae.SCOPE_102_106)
                mm_05, _ = compute_detailed_metrics(y_ev_5[mk], probs_mlp[mk], th05_5, src_ev[mk], cae.SCOPE_102_106)
                slice_eval[sl_name] = {"rows": int(mk.sum()), "sgd_05": ms_05, "mlp_05": mm_05}

        # Category 101 separately on v3 splits
        cat101_eval = None
        if sname.startswith("v3_"):
            y_101, _ = cae._rows_xy_for_ids(rows, (101,), use_eval=True)
            non101 = src_ev != 101
            pr101_sgd = sgd_heads[101].predict_proba(split_feats[sname][101])[:, 1]
            pr101_mlp = predict_mlp(mlp_heads[101], split_feats[sname][101])
            m101_sgd, _ = compute_detailed_metrics(y_101[non101], pr101_sgd[non101, None], np.array([0.5]), src_ev[non101], (101,))
            m101_mlp, _ = compute_detailed_metrics(y_101[non101], pr101_mlp[non101, None], np.array([0.5]), src_ev[non101], (101,))
            cat101_eval = {"sgd_05": m101_sgd, "mlp_05": m101_mlp}

        results["splits"][sname] = {
            "sgd_05": m_sgd_05,
            "sgd_cal": m_sgd_cal,
            "mlp_05": m_mlp_05,
            "mlp_cal": m_mlp_cal,
            "sgd_vs_mlp_pairwise_05": pairwise_agreement(y_ev_5, yp_sgd_05, yp_mlp_05, src_ev, cae.SCOPE_102_106),
            "slices_05": slice_eval,
            "cat_101_non101_05": cat101_eval,
        }

        if sname in ("pub_validation", "v3_test"):
            np.savez(
                sgd_dir / f"{sname}_preds.npz",
                y_true=y_ev_5,
                source_mcId=src_ev,
                probs=probs_sgd,
                pred_05=yp_sgd_05,
                pred_cal=yp_sgd_cal,
            )
            np.savez(
                mlp_dir / f"{sname}_preds.npz",
                y_true=y_ev_5,
                source_mcId=src_ev,
                probs=probs_mlp,
                pred_05=yp_mlp_05,
                pred_cal=yp_mlp_cal,
            )

    # --- C. Controlled Zero-Padding Ablation on SGD (zero-pad vs repeat-last-pad, NO text loss difference) ---
    rep_feats = {}
    for sname in ("pub_train", "v3_train", "pub_validation", "v3_test"):
        rep_feats[sname] = build_features_repeat_pad(npz[f"uid_{sname}"], offsets, cindices, chunk_emb_f32, chunk_cat_sims)
    rep_heads = {}
    for j, cid in enumerate(cae.ALL_IDS):
        if cid == 101:
            continue
        X_tr_r = np.vstack([rep_feats["pub_train"][cid], rep_feats["v3_train"][cid]])
        # Keep identical alpha from sgd_diag to isolate ONLY padding representation
        alpha_fixed = sgd_diag[str(cid)]["alpha"]
        from sklearn.linear_model import SGDClassifier
        rep_heads[cid] = SGDClassifier(**{**cae.SGD_PARAMS, "alpha": alpha_fixed}).fit(X_tr_r, y_tr_all[:, j])
    pad_ablation = {}
    for sname in ("pub_validation", "v3_test"):
        rows = splits[sname]
        y_ev_5, src_ev = cae._rows_xy_for_ids(rows, cae.SCOPE_102_106, use_eval=True)
        pr_rep = np.column_stack([rep_heads[cid].predict_proba(rep_feats[sname][cid])[:, 1] for cid in cae.SCOPE_102_106])
        m_rep, _ = compute_detailed_metrics(y_ev_5, pr_rep, th05_5, src_ev, cae.SCOPE_102_106)
        mk1 = split_nchunks[sname] == 1
        m_rep_1, _ = compute_detailed_metrics(y_ev_5[mk1], pr_rep[mk1], th05_5, src_ev[mk1], cae.SCOPE_102_106)
        pad_ablation[sname] = {"repeat_pad_all_05": m_rep, "repeat_pad_nchunks1_05": m_rep_1}
    results["padding_ablation_repeat_slot_sgd"] = pad_ablation

    (OUT_DIR / "metrics_comparison.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== FIXED SEGMENTATION + SGD vs MLP (public_plus_v3, thr=0.5) ===")
    for sp in ("pub_validation", "v3_test"):
        for tag in ("sgd_05", "mlp_05"):
            m = results["splits"][sp][tag]
            cc = m["confusion"]
            print(
                f"{sp:14s} | {tag:7s} | Micro-F1={m['micro_f1']:.4f} Macro-F1={m['macro_f1']:.4f} "
                f"P={m['micro_precision']:.4f} R={m['micro_recall']:.4f} FPR={m['fpr']:.4f} Spec={m['specificity']:.4f} "
                f"FP={cc['fp']:3d} FN={cc['fn']:3d} EM={m['exact_match']:.4f} Macro-AP={m['macro_ap']:.4f}"
            )
    print("\n=== OVERFITTING CHECK (Train -> Cal -> Test Micro/Mean-F1 at 0.5) ===")
    for tag, heads_d in [("SGD", sgd_diag), ("MLP", mlp_diag)]:
        tr_m = np.mean([heads_d[str(c)]["train_f1_05"] for c in cae.SCOPE_102_106])
        cal_m = np.mean([heads_d[str(c)]["cal_f1_05"] for c in cae.SCOPE_102_106])
        print(f"  {tag}: mean_train_F1={tr_m:.4f}, mean_cal_F1={cal_m:.4f}, gap={tr_m-cal_m:.4f}")


if __name__ == "__main__":
    main()
