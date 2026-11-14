#!/usr/bin/env python3
"""Category-Aware Qwen3-Embedding-0.6B (top-3 fragments per category) + binary SGDClassifier."""
import argparse
import inspect
import json
import math
import platform
import re
import time
from pathlib import Path

import numpy as np

from .qwen3_embedding_experiment import (
    IDS as SCOPE_102_106,
    MODEL_NAME,
    SGD_PARAMS,
    compute_metrics,
    load_official_parquet,
    normalize,
    paired_bootstrap_ci,
    peak_rss_mib,
    sha256,
)
from .qwen3_v3_experiment import adapt_v3_rows

ALL_IDS = (101, 102, 103, 104, 105, 106)
TOP_K = 3
ALPHA_GRID = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6)
QUERY_PROMPT = (
    "Instruct: Given a renovation service microcategory definition, retrieve ad passages "
    "mentioning these works, including direct offers, refusals, conditions, FAQ answers, or subcontractors\nQuery:"
)

# Grounded directly in mvp/innoglobalhack_avito/innoglobalhack_avito/microcategories-guide.md
CATEGORY_DESCRIPTIONS = {
    101: [
        "Комплексный ремонт квартиры, дома, коттеджа или помещения под ключ: от демонтажа и черновых работ до чистовой отделки и сдачи готового объекта.",
        "Косметический ремонт, капитальный ремонт или дизайнерский ремонт по проекту с обновлением отделки и инженерных коммуникаций.",
        "Полный цикл отделки помещения, комплексный ремонт комнаты, ванной или санузла под ключ по единому договору и смете.",
    ],
    102: [
        "Штукатурные работы: оштукатуривание и выравнивание стен, потолков, углов и откосов штукатурными составами и растворами.",
        "Штукатурка стен по маякам и без маяков, машинная и механизированная штукатурка гипсовой или цементной смесью.",
        "Формирование углов при оштукатуривании, восстановление геометрии откосов и выравнивание перегородок штукатуркой.",
    ],
    103: [
        "Плиточные работы: укладка и облицовка стен и полов керамической плиткой, кафелем, керамогранитом и мозаикой.",
        "Облицовка ванной комнаты, санузла, кухонного фартука, душевого поддона и бортика плиткой или крупноформатным керамогранитом.",
        "Затирка и перезатирка плиточных швов, запил плитки под 45 градусов, подгонка и обработка кромки керамогранита.",
    ],
    104: [
        "Сантехнические работы: монтаж, замена, подключение и ремонт сантехнического оборудования, водоснабжения, канализации и водяного отопления.",
        "Разводка труб водоснабжения и канализации, установка и подключение раковин, унитазов, инсталляций, ванн, душевых систем, смесителей и сифонов.",
        "Устранение протечек, течи под раковиной, прочистка засоров, установка радиаторов отопления, фильтров на водопровод, водонагревателей и подключение стиральных машин.",
    ],
    105: [
        "Электромонтажные работы: прокладка кабеля, замена электропроводки, штробление стен и диагностика неисправностей электросети.",
        "Монтаж, замена и перенос розеток, выключателей освещения, люстр, светильников и электроточек.",
        "Сборка и замена электрощита с автоматическими выключателями, УЗО и дифавтоматами, подключение электрического тёплого пола и нагревательных матов.",
    ],
    106: [
        "Малярные работы и поклейка обоев: финишная отделка стен и потолков обоями, стеклохолстом или покраской.",
        "Шпаклёвка стен и потолков под покраску и обои, грунтование, шлифовка основания и снятие старых обоев.",
        "Окраска и безвоздушная покраска стен, потолков и откосов, наклеивание виниловых, флизелиновых и бумажных обоев.",
    ],
}

BULLET_RE = re.compile(r"^(?:[•\-—*✔✅☑🔹🔸▪▪️📌🔧🔨⚡🚿🎨🧱🏠🏡📲📞📸⭐🔥🎁📍❗]+|\d+[.)])\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z0-9«\"])")
SHORT_ANSWER = re.compile(
    r"^(?:да|нет|верно|конечно|отвечаем|услуга\s+(?:доступна|не)|только\s+|однако\b|но\b)",
    re.I,
)
NEG_CUE_RE = re.compile(
    r"(?:\bне\s+(?:дела\w+|выполн\w+|бер[её]м\w*|оказыва\w+|предлаг\w+|занима\w+|подключа\w+|укладыва\w+|крас\w+|штукатур\w+|ремонтир\w+|принима\w+)|"
    r"\bбез\s+|отказ\w*|не\s+входит|больше\s+не\s+действует|таких\s+услуг\s+(?:у\s+нас\s+)?нет|случайно\s+попали)",
    re.I,
)
CONTRACTOR_RE = re.compile(
    r"(?:подрядчик\w*|сторонн\w+|чуж\w+|нанял\s+самостоятельно|исполнитель,\s+которого|штатные\s+мастера|по\s+нашему\s+договору)",
    re.I,
)


def _is_heading(ln: str) -> bool:
    if BULLET_RE.match(ln):
        return False
    if ln.endswith(":") and len(ln) <= 90:
        return True
    return len(ln) <= 55 and not re.search(r"[.!?]$", ln) and len(ln.split()) <= 7


def chunk_ad(text: str, target_chars: int = 300, max_chars: int = 420) -> list[str]:
    """Segment full ad text into semantic chunks without truncation, keeping headings, Q+A, negations and bullets intact."""
    text = text.strip()
    if not text:
        return [""]
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    units = []
    for p_idx, p in enumerate(raw_paras):
        if p_idx > 0:
            units.append(("break", ""))
        for raw_ln in p.splitlines():
            ln = re.sub(r"[ \t]+", " ", raw_ln).strip()
            if not ln or re.fullmatch(r"[\W_]{3,}", ln):
                continue
            if _is_heading(ln):
                units.append(("header", ln))
            elif BULLET_RE.match(ln):
                units.append(("bullet", ln))
            elif len(ln) <= 240:
                units.append(("prose", ln))
            else:
                for s in SENT_SPLIT.split(ln):
                    s = s.strip()
                    if s:
                        units.append(("prose", s))

    chunks = []
    cur_header = None
    header_pending = False
    i = 0
    while i < len(units):
        kind, val = units[i]
        if kind == "break":
            # Keep an immediately preceding header alive across a blank line so "Что делаем:\n\n— ..." stays attached.
            if not header_pending:
                cur_header = None
            i += 1
            continue
        if kind == "header":
            cur_header = val
            header_pending = True
            i += 1
            continue
        header_pending = False
        if kind == "bullet":
            block = val
            while (
                i + 1 < len(units)
                and units[i + 1][0] == "prose"
                and (block.endswith("?") or SHORT_ANSWER.match(units[i + 1][1]) or len(units[i + 1][1]) < 45)
                and len(block) + len(units[i + 1][1]) + 1 <= max_chars
            ):
                i += 1
                block = block + " " + units[i][1]
            chunks.append(f"{cur_header}\n{block}" if cur_header and len(cur_header) + len(block) + 1 <= max_chars else block)
            i += 1
            continue
        buf = [val]
        prefix = f"{cur_header}\n" if cur_header else ""
        cur_len = len(prefix) + len(val)
        j = i + 1
        while j < len(units) and units[j][0] == "prose":
            nxt = units[j][1]
            must_attach = buf[-1].endswith("?") or SHORT_ANSWER.match(nxt) or len(nxt) < 55
            if cur_len + 1 + len(nxt) <= (max_chars if must_attach else target_chars):
                buf.append(nxt)
                cur_len += 1 + len(nxt)
                j += 1
            else:
                break
        chunks.append(prefix + " ".join(buf))
        cur_header = None  # prose paragraph consumes the heading
        if j < len(units) and units[j][0] == "prose" and len(buf) >= 2:
            i = j - 1
        else:
            i = j
    if header_pending and cur_header:
        chunks.append(cur_header)
    return chunks or [text[:max_chars]]


def load_public_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for r in rows:
        lbls = sorted(set(r["labels"]) & set(SCOPE_102_106))
        out.append({
            "description": r["description"],
            "mcId": int(r.get("mcId") or 101),
            "labels": lbls,
            "evaluation_labels": lbls,
            "has_101_label": False,
            "group_id": normalize(r["description"]),
        })
    return out


def load_v3_rows_full(path: Path) -> list[dict]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    out = []
    for r in raw:
        src = int(r["mcId"])
        offered = sorted(set(r.get("offered_microcategories", [])) & set(ALL_IDS))
        found = sorted({e["mcId"] for e in r["found_microcategories"]} & set(ALL_IDS))
        if sorted(set(offered) - {src}) != found:
            raise ValueError(f"Inconsistent v3 offered/found labels for item {r.get('item_id')}")
        out.append({
            "description": r["description"],
            "mcId": src,
            "labels": offered,
            "evaluation_labels": found,
            "has_101_label": True,
            "item_id": r.get("item_id"),
            "group_id": r.get("group_id", normalize(r["description"])),
            "construction": r.get("construction"),
            "length_bin": r.get("length_bin"),
        })
    return out


def collect_all_splits(public_dir: Path, v3_dir: Path, official_parquet: Path | None):
    splits = {
        "pub_train": load_public_rows(public_dir / "train.jsonl"),
        "pub_calibration": load_public_rows(public_dir / "calibration.jsonl"),
        "pub_validation": load_public_rows(public_dir / "validation.jsonl"),
        "v3_train": load_v3_rows_full(v3_dir / "train.jsonl"),
        "v3_validation": load_v3_rows_full(v3_dir / "validation.jsonl"),
        "v3_test": load_v3_rows_full(v3_dir / "test.jsonl"),
        "v3_challenge": load_v3_rows_full(v3_dir / "challenge.jsonl"),
    }
    if official_parquet is not None and official_parquet.exists():
        _, _, off_rows = load_official_parquet(official_parquet)
        for r in off_rows:
            r["has_101_label"] = False
            r["group_id"] = normalize(r["description"])
        splits["official_208"] = off_rows
    return splits


def encode_chunks(
    public_dir: Path,
    v3_dir: Path,
    cache_dir: Path,
    official_parquet: Path | None = None,
    batch_size: int | None = None,
    max_seq_length: int = 1024,
    device: str | None = None,
    force: bool = False,
):
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / "chunks_embeddings.npz"
    meta_path = cache_dir / "chunk_encoder_meta.json"
    splits = collect_all_splits(public_dir, v3_dir, official_parquet)

    data_hashes = {
        "pub_train": sha256(public_dir / "train.jsonl"),
        "pub_cal": sha256(public_dir / "calibration.jsonl"),
        "pub_val": sha256(public_dir / "validation.jsonl"),
        "v3_train": sha256(v3_dir / "train.jsonl"),
        "v3_val": sha256(v3_dir / "validation.jsonl"),
        "v3_test": sha256(v3_dir / "test.jsonl"),
        "v3_challenge": sha256(v3_dir / "challenge.jsonl"),
        "official": sha256(official_parquet) if official_parquet and official_parquet.exists() else None,
        "categories": CATEGORY_DESCRIPTIONS,
    }
    if not force and cache_npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("data_hashes") == data_hashes and meta.get("prompt") == QUERY_PROMPT:
            print(f"Reusing cached chunk embeddings in {cache_dir}", flush=True)
            return cache_npz, meta_path

    # Deduplicate full ad texts across all splits and segment each unique ad once.
    uniq_ads = []
    ad_to_uid = {}
    split_ad_uids = {}
    for sname, rows in splits.items():
        uids = []
        for r in rows:
            txt = r["description"]
            if txt not in ad_to_uid:
                ad_to_uid[txt] = len(uniq_ads)
                uniq_ads.append(txt)
            uids.append(ad_to_uid[txt])
        split_ad_uids[sname] = np.asarray(uids, dtype=np.int32)

    uniq_chunks = []
    chunk_to_cid = {}
    ad_chunk_offsets = [0]
    ad_chunk_indices = []
    ad_chunk_counts = []
    for txt in uniq_ads:
        cs = chunk_ad(txt)
        ad_chunk_counts.append(len(cs))
        for ch in cs:
            if ch not in chunk_to_cid:
                chunk_to_cid[ch] = len(uniq_chunks)
                uniq_chunks.append(ch)
            ad_chunk_indices.append(chunk_to_cid[ch])
        ad_chunk_offsets.append(len(ad_chunk_indices))

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if batch_size is None:
        batch_size = 64 if device == "cuda" else 8

    sig = inspect.signature(SentenceTransformer.__init__).parameters
    kw_name = "processor_kwargs" if "processor_kwargs" in sig else "tokenizer_kwargs"
    t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={"dtype": dtype},
        **{kw_name: {"padding_side": "left"}},
    )
    model.max_seq_length = max_seq_length
    load_s = time.perf_counter() - t0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    query_texts = [d for cid in ALL_IDS for d in CATEGORY_DESCRIPTIONS[cid]]
    qt0 = time.perf_counter()
    query_emb = model.encode(
        query_texts,
        prompt=QUERY_PROMPT,
        batch_size=len(query_texts),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).astype(np.float32)
    if device == "cuda":
        torch.cuda.synchronize()
    query_encode_s = time.perf_counter() - qt0

    ct0 = time.perf_counter()
    chunk_emb = model.encode(
        uniq_chunks,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float16)
    if device == "cuda":
        torch.cuda.synchronize()
    chunk_encode_s = time.perf_counter() - ct0

    # Measure single-ad end-to-end fragment encoding latency after warmup on 20 pub_validation ads.
    val_texts = [r["description"] for r in splits["pub_validation"][:23]]
    for wt in val_texts[:3]:
        wcs = chunk_ad(wt)
        model.encode(wcs, batch_size=len(wcs), normalize_embeddings=True, show_progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()
    single_lat = []
    for st in val_texts[3:23]:
        lt0 = time.perf_counter()
        scs = chunk_ad(st)
        _ = model.encode(scs, batch_size=len(scs), normalize_embeddings=True, show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        single_lat.append(time.perf_counter() - lt0)

    arrays = {
        "chunk_emb": chunk_emb,
        "query_emb": query_emb,
        "ad_chunk_offsets": np.asarray(ad_chunk_offsets, dtype=np.int32),
        "ad_chunk_indices": np.asarray(ad_chunk_indices, dtype=np.int32),
    }
    for sname, arr in split_ad_uids.items():
        arrays[f"uid_{sname}"] = arr
    np.savez_compressed(cache_npz, **arrays)

    counts_sorted = sorted(ad_chunk_counts)
    n_ads = len(counts_sorted)
    norms = np.linalg.norm(chunk_emb.astype(np.float32), axis=1)
    meta = {
        "model_name": MODEL_NAME,
        "prompt": QUERY_PROMPT,
        "categories": {str(k): v for k, v in CATEGORY_DESCRIPTIONS.items()},
        "max_seq_length": max_seq_length,
        "batch_size": batch_size,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "dtype": str(dtype),
        "unique_ads": len(uniq_ads),
        "unique_chunks": len(uniq_chunks),
        "total_chunk_refs": len(ad_chunk_indices),
        "chunks_per_ad_stats": {
            "min": counts_sorted[0],
            "median": counts_sorted[n_ads // 2],
            "p90": counts_sorted[int(n_ads * 0.9)],
            "max": counts_sorted[-1],
        },
        "chunk_norm_stats": {"min": float(norms.min()), "max": float(norms.max())},
        "model_load_seconds": load_s,
        "query_encode_seconds": query_encode_s,
        "chunk_encode_seconds": chunk_encode_s,
        "single_ad_chunk_encode_seconds_after_warmup": {
            "n_samples": len(single_lat),
            "mean": float(np.mean(single_lat)),
            "median": float(np.median(single_lat)),
            "p95": float(np.percentile(single_lat, 95)),
        },
        "encode_peak_process_rss_mib": peak_rss_mib(),
        "encode_peak_cuda_allocated_mib": float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0,
        "data_hashes": data_hashes,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved chunk cache: {len(uniq_chunks)} unique chunks in {chunk_encode_s:.2f}s", flush=True)
    return cache_npz, meta_path


def build_category_features_for_uids(
    uids: np.ndarray,
    ad_chunk_offsets: np.ndarray,
    ad_chunk_indices: np.ndarray,
    chunk_emb_f32: np.ndarray,
    chunk_cat_sims: np.ndarray,
    return_selection: bool = False,
):
    """Build [TOP_K * dim + 2 * TOP_K] fixed feature matrices for each category 101..106."""
    n = len(uids)
    dim = chunk_emb_f32.shape[1]
    feat_dim = TOP_K * dim + 2 * TOP_K  # 3*1024 embeddings + 3 sims + 3 presence masks = 3078
    feats = {cid: np.zeros((n, feat_dim), dtype=np.float32) for cid in ALL_IDS}
    selections = {cid: [] for cid in ALL_IDS} if return_selection else None

    for row_idx, uid in enumerate(uids):
        start = int(ad_chunk_offsets[uid])
        end = int(ad_chunk_offsets[uid + 1])
        cids = ad_chunk_indices[start:end]
        m = len(cids)
        sims_mat = chunk_cat_sims[cids]  # (m, 6)
        embs_mat = chunk_emb_f32[cids]   # (m, dim)
        k_take = min(TOP_K, m)
        for col, cid in enumerate(ALL_IDS):
            order = np.argsort(-sims_mat[:, col], kind="stable")[:k_take]
            vec = feats[cid][row_idx]
            sel_info = []
            for slot, local_pos in enumerate(order):
                vec[slot * dim : (slot + 1) * dim] = embs_mat[local_pos]
                sim_val = float(sims_mat[local_pos, col])
                vec[TOP_K * dim + slot] = sim_val
                vec[TOP_K * dim + TOP_K + slot] = 1.0
                if return_selection:
                    sel_info.append((int(cids[local_pos]), round(sim_val, 4)))
            if return_selection:
                selections[cid].append(sel_info)
    return feats, selections


def _fit_cv_binary_head(X: np.ndarray, y: np.ndarray):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    alpha_diag = {}
    best_alpha = ALPHA_GRID[0]
    best_oof = -1.0
    for alpha in ALPHA_GRID:
        oof = np.zeros(len(y), dtype=np.float64)
        tr_scores = []
        for tr_i, va_i in skf.split(X, y):
            clf = SGDClassifier(**{**SGD_PARAMS, "alpha": alpha}).fit(X[tr_i], y[tr_i])
            tr_scores.append(f1_score(y[tr_i], clf.predict_proba(X[tr_i])[:, 1] >= 0.5, zero_division=0))
            oof[va_i] = clf.predict_proba(X[va_i])[:, 1]
        oof_f1 = float(f1_score(y, oof >= 0.5, zero_division=0))
        mean_tr_f1 = float(np.mean(tr_scores))
        alpha_diag[str(alpha)] = {
            "oof_f1": round(oof_f1, 6),
            "mean_train_f1": round(mean_tr_f1, 6),
            "train_minus_oof_gap": round(mean_tr_f1 - oof_f1, 6),
        }
        if oof_f1 > best_oof:
            best_oof = oof_f1
            best_alpha = alpha

    final_clf = SGDClassifier(**{**SGD_PARAMS, "alpha": best_alpha}).fit(X, y)
    full_train_f1 = float(f1_score(y, final_clf.predict_proba(X)[:, 1] >= 0.5, zero_division=0))
    return final_clf, best_alpha, full_train_f1, alpha_diag


def _eval_binary_scope(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray, sources: np.ndarray, ids_tuple: tuple[int, ...]):
    yp = (probs >= thresholds).astype(np.int8)
    for i, src in enumerate(sources):
        if int(src) in ids_tuple:
            yp[i, ids_tuple.index(int(src))] = 0
    tp = int(np.sum((yp == 1) & (y_true == 1)))
    fp = int(np.sum((yp == 1) & (y_true == 0)))
    fn = int(np.sum((yp == 0) & (y_true == 1)))
    tn = int(np.sum((yp == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    f1_by_cat = {}
    for j, cid in enumerate(ids_tuple):
        tp_c = int(np.sum((yp[:, j] == 1) & (y_true[:, j] == 1)))
        fp_c = int(np.sum((yp[:, j] == 1) & (y_true[:, j] == 0)))
        fn_c = int(np.sum((yp[:, j] == 0) & (y_true[:, j] == 1)))
        f1_by_cat[str(cid)] = (2 * tp_c / (2 * tp_c + fp_c + fn_c)) if (2 * tp_c + fp_c + fn_c) else 0.0
    return {
        "rows": int(len(y_true)),
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(list(f1_by_cat.values()))),
        "micro_precision": prec,
        "micro_recall": rec,
        "exact_set_accuracy": float(np.mean(np.all(yp == y_true, axis=1))),
        "empty_predictions": int(np.sum(yp.sum(axis=1) == 0)),
        "empty_gold": int(np.sum(y_true.sum(axis=1) == 0)),
        "confusion_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "f1_by_category": f1_by_cat,
    }, yp


def _fit_baseline_a_and_predict(regime: str, pub_npz, v3_npz, pub_rows, v3_rows):
    """Fit full-ad Qwen3 Baseline A with identical split & 5-fold CV alpha grid for fair slice/empty comparison."""
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold
    from sklearn.multiclass import OneVsRestClassifier

    if regime == "public_only":
        X_tr, y_tr = pub_npz["X_train"], pub_npz["y_train"]
        X_cal, y_cal, src_cal = pub_npz["X_calibration"], pub_npz["y_calibration"], np.full(len(pub_rows["cal"]), 101)
    elif regime == "v3_only":
        X_tr, y_tr = v3_npz["X_train"], v3_npz["y_train_train"]
        X_cal, y_cal, src_cal = v3_npz["X_validation"], v3_npz["y_eval_validation"], v3_npz["sources_validation"]
    else:
        X_tr = np.vstack([pub_npz["X_train"], v3_npz["X_train"]])
        y_tr = np.vstack([pub_npz["y_train"], v3_npz["y_train_train"]])
        X_cal = np.vstack([pub_npz["X_calibration"], v3_npz["X_validation"]])
        y_cal = np.vstack([pub_npz["y_calibration"], v3_npz["y_eval_validation"]])
        src_cal = np.concatenate([np.full(len(pub_rows["cal"]), 101), v3_npz["sources_validation"]])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_a, best_s = 1e-4, -1.0
    for alpha in (1e-4, 1e-5, 3e-6, 1e-6, 3e-7):
        oof = np.zeros_like(y_tr, dtype=float)
        for tri, vai in kf.split(X_tr):
            c = OneVsRestClassifier(SGDClassifier(**{**SGD_PARAMS, "alpha": alpha})).fit(X_tr[tri], y_tr[tri])
            oof[vai] = c.predict_proba(X_tr[vai])
        sc = float(f1_score(y_tr, oof >= 0.5, average="micro", zero_division=0))
        if sc > best_s:
            best_s, best_a = sc, alpha

    clf = OneVsRestClassifier(SGDClassifier(**{**SGD_PARAMS, "alpha": best_a})).fit(X_tr, y_tr)
    p_cal = clf.predict_proba(X_cal)
    grid = np.arange(0.20, 0.81, 0.025)
    ths = np.asarray([
        max(grid, key=lambda t: f1_score(y_cal[src_cal != cid, j], p_cal[src_cal != cid, j] >= t, zero_division=0))
        for j, cid in enumerate(SCOPE_102_106)
    ])
    return clf, ths


def _rows_xy_for_ids(rows: list[dict], ids_tuple: tuple[int, ...], use_eval: bool = False):
    key = "evaluation_labels" if use_eval else "labels"
    y = np.asarray([[int(cid in r[key]) for cid in ids_tuple] for r in rows], dtype=np.int8)
    src = np.asarray([r["mcId"] for r in rows], dtype=np.int16)
    return y, src


def _slice_masks(eval_name: str, rows: list[dict]):
    if eval_name == "pub_validation":
        return {
            "all": np.ones(len(rows), dtype=bool),
            "long_ads_ge_1467": np.asarray([len(r["description"]) >= 1467 for r in rows]),
            "negations": np.asarray([bool(NEG_CUE_RE.search(r["description"])) for r in rows]),
            "service_lists": np.asarray([len(re.findall(r"(?:^|\n)\s*(?:[•\-—✔✅🔹🔸▪️]|\d+[.)])\s+", r["description"])) >= 4 for r in rows]),
        }
    if eval_name == "v3_test":
        return {
            "all": np.ones(len(rows), dtype=bool),
            "negation_constructions": np.asarray([r.get("construction") in ("not_only", "past", "price", "team") for r in rows]),
            "contractor_and_team": np.asarray([r.get("construction") in ("team", "not_only") for r in rows]),
            "long_and_extended": np.asarray([r.get("length_bin") in ("long", "extended") for r in rows]),
            "faq_pairs": np.asarray([r.get("construction") == "faq" for r in rows]),
        }
    return {"all": np.ones(len(rows), dtype=bool)}


def _build_unique_chunk_lookup(splits: dict) -> list[str]:
    uniq_ads = {}
    uniq_chunks = []
    chunk_set = {}
    for rows in splits.values():
        for r in rows:
            txt = r["description"]
            if txt in uniq_ads:
                continue
            uniq_ads[txt] = 1
            for ch in chunk_ad(txt):
                if ch not in chunk_set:
                    chunk_set[ch] = len(uniq_chunks)
                    uniq_chunks.append(ch)
    return uniq_chunks


def _compute_sample_size_note():
    """Binomial precision & EPV justification for annotating additional category 101 in real public flow."""
    # For binary classification over 3078 frozen L2-regularized dims + binomial CI (half-width <= 0.04 at 95% CL):
    # n_test >= z^2 * p*(1-p) / d^2; at worst-case p=0.5, z=1.96, d=0.04 => 600 (or at p=0.25 => 450)
    z = 1.96
    return {
        "public_additional_101_labeled_positives": 0,
        "public_additional_101_labeled_negatives_verified": 0,
        "recommended_annotation_budget_real_flow": {
            "train_non_101_ads": "1000-1500 реальных объявлений (mcId 102-106), из них минимум 300-400 подтверждённых 101=1 и >=400 сложных негативов (много услуг без 'под ключ' и 'укладка плитки под ключ')",
            "calibration_non_101_ads": "250-300 объявлений (для подбора порога 101 отдельно от train/test)",
            "test_non_101_ads": "450-600 объявлений (95% биномиальный ДИ шириной <= +-4..4.5 п.п. при p=0.25..0.5: ceil(1.96^2 * 0.25 / 0.04^2) = 601, при p=0.25 = 451)",
            "total_minimum_to_collect": "1700-2400 размеченных объявлений с исходными mcId 102-106",
        },
        "v3_available_101_counts": {"train_non101_pos": 995, "train_non101_neg": 1505, "val_non101_pos": 199, "test_non101_pos": 199, "challenge_non101_pos": 10},
        "binomial_ci_formula": f"n >= ({z}^2 * p*(1-p)) / d^2 = {math.ceil((z**2 * 0.25) / (0.04**2))} (p=0.5, d=0.04)",
    }


def train_and_eval(
    public_dir: Path,
    v3_dir: Path,
    cache_dir: Path,
    public_emb_cache: Path,
    v3_emb_cache: Path,
    output_dir: Path,
    official_parquet: Path | None = None,
):
    from sklearn.metrics import f1_score

    output_dir.mkdir(parents=True, exist_ok=True)
    splits = collect_all_splits(public_dir, v3_dir, official_parquet)
    chunk_npz = np.load(cache_dir / "chunks_embeddings.npz")
    chunk_meta = json.loads((cache_dir / "chunk_encoder_meta.json").read_text(encoding="utf-8"))
    uniq_chunks = _build_unique_chunk_lookup(splits)

    chunk_emb_f32 = chunk_npz["chunk_emb"].astype(np.float32)
    query_emb_f32 = chunk_npz["query_emb"].astype(np.float32)  # (18, 1024)
    offsets = chunk_npz["ad_chunk_offsets"]
    cindices = chunk_npz["ad_chunk_indices"]
    if len(uniq_chunks) != len(chunk_emb_f32):
        raise ValueError(f"Chunk count mismatch: {len(uniq_chunks)} vs {len(chunk_emb_f32)}")

    # Cosine similarity of every unique chunk against 6 categories (max over 3 handbook descriptors per category)
    sim_t0 = time.perf_counter()
    raw_sims = chunk_emb_f32 @ query_emb_f32.T  # (U, 18)
    chunk_cat_sims = np.column_stack([raw_sims[:, j * 3 : (j + 1) * 3].max(axis=1) for j in range(len(ALL_IDS))])
    sim_seconds = time.perf_counter() - sim_t0

    # Pre-build per-split category features (never using labels)
    split_feats = {}
    split_sel = {}
    feat_t0 = time.perf_counter()
    for sname in splits:
        need_sel = sname in ("pub_validation", "v3_test")
        feats, sel = build_category_features_for_uids(
            chunk_npz[f"uid_{sname}"], offsets, cindices, chunk_emb_f32, chunk_cat_sims, return_selection=need_sel
        )
        split_feats[sname] = feats
        if need_sel:
            split_sel[sname] = sel
    feat_build_seconds = time.perf_counter() - feat_t0

    # Measure single-ad CPU feature build + 6-head SGD inference latency (after warmup)
    pub_npz = np.load(public_emb_cache / "embeddings.npz")
    v3_npz = np.load(v3_emb_cache / "v3_embeddings.npz")
    pub_rows_map = {"train": splits["pub_train"], "cal": splits["pub_calibration"], "val": splits["pub_validation"]}
    v3_rows_map = {"train": splits["v3_train"], "val": splits["v3_validation"], "test": splits["v3_test"]}

    eval_split_names = [s for s in ("pub_validation", "official_208", "v3_test", "v3_challenge") if s in splits]
    report = {
        "encoder_meta": chunk_meta,
        "similarity_matrix_seconds": sim_seconds,
        "feature_build_all_splits_seconds": feat_build_seconds,
        "category_101_audit": _compute_sample_size_note(),
        "regimes": {},
    }
    errors_20 = {}

    for regime in ("public_only", "v3_only", "public_plus_v3"):
        if regime == "public_only":
            tr_names, cal_names = ["pub_train"], ["pub_calibration"]
        elif regime == "v3_only":
            tr_names, cal_names = ["v3_train"], ["v3_validation"]
        else:
            tr_names, cal_names = ["pub_train", "v3_train"], ["pub_calibration", "v3_validation"]

        tr_rows = [r for s in tr_names for r in splits[s]]
        cal_rows = [r for s in cal_names for r in splits[s]]
        y_tr_all, _ = _rows_xy_for_ids(tr_rows, ALL_IDS, use_eval=False)
        y_cal_all, src_cal_all = _rows_xy_for_ids(cal_rows, ALL_IDS, use_eval=True)
        has101_tr = np.asarray([r["has_101_label"] for r in tr_rows], dtype=bool)
        has101_cal = np.asarray([r["has_101_label"] for r in cal_rows], dtype=bool)

        heads = {}
        head_diag = {}
        cal_probs = {}
        cal_ths_101_106 = []
        fit_t0 = time.perf_counter()

        for j, cid in enumerate(ALL_IDS):
            X_tr_c = np.vstack([split_feats[s][cid] for s in tr_names])
            X_cal_c = np.vstack([split_feats[s][cid] for s in cal_names])
            if cid == 101 and not np.any(has101_tr):
                # Do NOT fabricate 101=0 on public_only
                head_diag[str(cid)] = {"trained": False, "reason": "No reliable additional 101 labels in public dataset"}
                cal_probs[cid] = np.zeros(len(cal_rows), dtype=np.float64)
                cal_ths_101_106.append(0.5)
                continue

            mask_tr = has101_tr if cid == 101 else np.ones(len(tr_rows), dtype=bool)
            clf_c, best_alpha, full_tr_f1, alpha_diag = _fit_cv_binary_head(X_tr_c[mask_tr], y_tr_all[mask_tr, j])
            heads[cid] = clf_c
            p_cal_c = clf_c.predict_proba(X_cal_c)[:, 1]
            cal_probs[cid] = p_cal_c

            mask_cal = (src_cal_all != cid) & (has101_cal if cid == 101 else True)
            grid = np.arange(0.20, 0.81, 0.025)
            best_th = max(grid, key=lambda t: f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= t, zero_division=0))
            cal_f1_at_05 = float(f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= 0.5, zero_division=0))
            cal_f1_at_th = float(f1_score(y_cal_all[mask_cal, j], p_cal_c[mask_cal] >= best_th, zero_division=0))
            cal_ths_101_106.append(float(best_th))

            head_diag[str(cid)] = {
                "trained": True,
                "train_rows": int(np.sum(mask_tr)),
                "train_positives": int(np.sum(y_tr_all[mask_tr, j])),
                "selected_alpha": best_alpha,
                "full_train_f1_05": round(full_tr_f1, 6),
                "cv_alpha_table": alpha_diag,
                "cal_rows": int(np.sum(mask_cal)),
                "cal_f1_05": round(cal_f1_at_05, 6),
                "cal_f1_calibrated": round(cal_f1_at_th, 6),
                "calibrated_threshold": float(best_th),
                "train_minus_cal05_gap": round(full_tr_f1 - cal_f1_at_05, 6),
            }

        cat_fit_seconds = time.perf_counter() - fit_t0

        # Single-ad CPU feature extraction + SGD heads latency (200 ads after warmup)
        val_uids = chunk_npz["uid_pub_validation"][:200]
        inf_t0 = time.perf_counter()
        tmp_feats, _ = build_category_features_for_uids(val_uids, offsets, cindices, chunk_emb_f32, chunk_cat_sims)
        for cid, clf_c in heads.items():
            _ = clf_c.predict_proba(tmp_feats[cid])[:, 1]
        per_ad_feat_sgd_s = (time.perf_counter() - inf_t0) / len(val_uids)

        # Fit matched Baseline A (full-ad Qwen3 + SGD)
        base_clf, base_ths = _fit_baseline_a_and_predict(regime, pub_npz, v3_npz, pub_rows_map, v3_rows_map)

        ths_102_106 = np.asarray(cal_ths_101_106[1:], dtype=float)
        ths_101_106 = np.asarray(cal_ths_101_106, dtype=float)
        th05_5 = np.full(5, 0.5)
        th05_6 = np.full(6, 0.5)

        eval_results = {}
        err_samples = []
        for sname in eval_split_names:
            rows = splits[sname]
            y_ev_5, src_ev = _rows_xy_for_ids(rows, SCOPE_102_106, use_eval=True)
            probs_5 = np.column_stack([heads[cid].predict_proba(split_feats[sname][cid])[:, 1] for cid in SCOPE_102_106])

            m_b05, yp_b05 = _eval_binary_scope(y_ev_5, probs_5, th05_5, src_ev, SCOPE_102_106)
            m_bcal, yp_bcal = _eval_binary_scope(y_ev_5, probs_5, ths_102_106, src_ev, SCOPE_102_106)

            # Baseline A embeddings for this eval split
            if sname == "pub_validation":
                X_base = pub_npz["X_validation"]
            elif sname == "official_208":
                X_base = pub_npz["X_official"]
            elif sname == "v3_test":
                X_base = v3_npz["X_test"]
            else:
                X_base = v3_npz["X_challenge"]
            p_base = base_clf.predict_proba(X_base)
            m_a05, yp_a05 = _eval_binary_scope(y_ev_5, p_base, th05_5, src_ev, SCOPE_102_106)
            m_acal, yp_acal = _eval_binary_scope(y_ev_5, p_base, base_ths, src_ev, SCOPE_102_106)

            groups = [r["group_id"] for r in rows]
            slice_res = {}
            for sl_name, mask in _slice_masks(sname, rows).items():
                if int(np.sum(mask)) == 0:
                    continue
                ms_bcal, _ = _eval_binary_scope(y_ev_5[mask], probs_5[mask], ths_102_106, src_ev[mask], SCOPE_102_106)
                ms_b05, _ = _eval_binary_scope(y_ev_5[mask], probs_5[mask], th05_5, src_ev[mask], SCOPE_102_106)
                ms_acal, _ = _eval_binary_scope(y_ev_5[mask], p_base[mask], base_ths, src_ev[mask], SCOPE_102_106)
                ms_a05, _ = _eval_binary_scope(y_ev_5[mask], p_base[mask], th05_5, src_ev[mask], SCOPE_102_106)
                slice_res[sl_name] = {
                    "rows": int(np.sum(mask)),
                    "cat_aware_05": ms_b05,
                    "cat_aware_cal": ms_bcal,
                    "baseline_a_05": ms_a05,
                    "baseline_a_cal": ms_acal,
                }

            split_entry = {
                "scope_102_106": {
                    "cat_aware_0_5": m_b05,
                    "cat_aware_calibrated": m_bcal,
                    "baseline_a_0_5": m_a05,
                    "baseline_a_calibrated": m_acal,
                    "bootstrap_b05_vs_a05": paired_bootstrap_ci(y_ev_5, yp_a05, yp_b05, groups),
                    "bootstrap_bcal_vs_acal": paired_bootstrap_ci(y_ev_5, yp_acal, yp_bcal, groups),
                    "slices": slice_res,
                }
            }

            # Evaluate 101-106 and 101 behavior on v3_test & v3_challenge when 101 head is trained
            if 101 in heads and sname in ("v3_test", "v3_challenge"):
                y_ev_6, _ = _rows_xy_for_ids(rows, ALL_IDS, use_eval=True)
                probs_6 = np.column_stack([heads[cid].predict_proba(split_feats[sname][cid])[:, 1] for cid in ALL_IDS])
                m6_05, yp6_05 = _eval_binary_scope(y_ev_6, probs_6, th05_6, src_ev, ALL_IDS)
                m6_cal, yp6_cal = _eval_binary_scope(y_ev_6, probs_6, ths_101_106, src_ev, ALL_IDS)

                non101 = src_ev != 101
                y101_true = y_ev_6[non101, 0]
                p101_pred_cal = yp6_cal[non101, 0]
                multi_no_turnkey = non101 & (y_ev_6[:, 0] == 0) & (y_ev_5.sum(axis=1) >= 2)
                n_mnt = int(np.sum(multi_no_turnkey))
                fp_mnt = int(np.sum(yp6_cal[multi_no_turnkey, 0] == 1)) if n_mnt else 0

                split_entry["scope_101_106"] = {
                    "cat_aware_0_5": m6_05,
                    "cat_aware_calibrated": m6_cal,
                    "category_101_on_non101_sources": {
                        "rows": int(np.sum(non101)),
                        "gold_positives": int(np.sum(y101_true)),
                        "pred_positives_cal": int(np.sum(p101_pred_cal)),
                        "f1_101_cal": m6_cal["f1_by_category"]["101"],
                        "multi_service_without_turnkey_rows": n_mnt,
                        "multi_service_without_turnkey_101_fp": fp_mnt,
                        "multi_service_without_turnkey_101_fp_rate": (fp_mnt / n_mnt) if n_mnt else 0.0,
                    },
                }

            # Verify retrieval + negation attribution on v3_test negative items
            if sname == "v3_test":
                neg_audit = {"total_negative_pairs_in_neg_constructions": 0, "top3_contains_neg_cue": 0, "top3_neg_cue_rejected_by_cat_aware": 0, "top3_neg_cue_rejected_by_base_a": 0}
                neg_mask = [i for i, r in enumerate(rows) if r.get("construction") in ("past", "not_only", "price", "team")]
                for i in neg_mask:
                    for j, cid in enumerate(SCOPE_102_106):
                        if int(src_ev[i]) == cid or y_ev_5[i, j] != 0:
                            continue
                        top3 = split_sel[sname][cid][i]
                        has_neg = any(bool(NEG_CUE_RE.search(uniq_chunks[c_idx]) or CONTRACTOR_RE.search(uniq_chunks[c_idx])) for c_idx, _ in top3)
                        neg_audit["total_negative_pairs_in_neg_constructions"] += 1
                        if has_neg:
                            neg_audit["top3_contains_neg_cue"] += 1
                            if yp_bcal[i, j] == 0:
                                neg_audit["top3_neg_cue_rejected_by_cat_aware"] += 1
                            if yp_acal[i, j] == 0:
                                neg_audit["top3_neg_cue_rejected_by_base_a"] += 1
                split_entry["negation_retrieval_audit"] = neg_audit

            eval_results[sname] = split_entry

            # Collect 10 error samples each from pub_validation and v3_test (total 20 per regime)
            if sname in ("pub_validation", "v3_test"):
                count_added = 0
                for i, r in enumerate(rows):
                    if np.array_equal(yp_bcal[i], y_ev_5[i]):
                        continue
                    top_frags = {}
                    probs_dict = {}
                    for cid in ALL_IDS:
                        top_frags[str(cid)] = [
                            {"sim": sim, "text": uniq_chunks[c_idx]} for c_idx, sim in split_sel[sname][cid][i]
                        ]
                        probs_dict[str(cid)] = round(float(heads[cid].predict_proba(split_feats[sname][cid][i : i + 1])[:, 1][0]), 4) if cid in heads else None
                    err_samples.append({
                        "split": sname,
                        "index": i,
                        "construction": r.get("construction"),
                        "source_mcId": int(r["mcId"]),
                        "gold_102_106": [SCOPE_102_106[j] for j in range(5) if y_ev_5[i, j] == 1],
                        "cat_aware_pred_102_106": [SCOPE_102_106[j] for j in range(5) if yp_bcal[i, j] == 1],
                        "baseline_a_pred_102_106": [SCOPE_102_106[j] for j in range(5) if yp_acal[i, j] == 1],
                        "gold_101_offered": (101 in r["evaluation_labels"]) if r.get("has_101_label") else None,
                        "probs": probs_dict,
                        "selected_top3_by_category": top_frags,
                        "description": r["description"],
                    })
                    count_added += 1
                    if count_added >= 10:
                        break

        errors_20[regime] = err_samples[:20]
        report["regimes"][regime] = {
            "category_heads": head_diag,
            "cat_aware_fit_seconds": cat_fit_seconds,
            "single_ad_cpu_feat_plus_sgd_ms": per_ad_feat_sgd_s * 1000.0,
            "train_eval_peak_rss_mib": peak_rss_mib(),
            "evaluations": eval_results,
        }

    (output_dir / "measurement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "errors_20.json").write_text(json.dumps(errors_20, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {output_dir}", flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--public-dir", type=Path, default=Path("output/ngram-comparison/splits"))
    p.add_argument("--v3-dir", type=Path, default=Path("/tmp/innopolis-dataset-v3-review/datasets/own_v3"))
    p.add_argument("--public-emb-cache", type=Path, default=Path("output/qwen3-embedding-sgd/cache"))
    p.add_argument("--v3-emb-cache", type=Path, default=Path("output/qwen3-v3-experiment/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("output/qwen3-category-aware"))
    default_parquet = Path("datasets/public/public-test-dataset.parquet")
    if not default_parquet.exists():
        default_parquet = Path("mvp/innoglobalhack_avito/innoglobalhack_avito/public-test-dataset.parquet")
    p.add_argument("--official-parquet", type=Path, default=default_parquet)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--device", default=None)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--train-eval-only", action="store_true")
    p.add_argument("--force-encode", action="store_true")
    args = p.parse_args()

    cache_dir = args.output_dir / "cache"
    if not args.train_eval_only:
        encode_chunks(
            public_dir=args.public_dir,
            v3_dir=args.v3_dir,
            cache_dir=cache_dir,
            official_parquet=args.official_parquet if args.official_parquet.exists() else None,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            device=args.device,
            force=args.force_encode,
        )
    if not args.encode_only:
        train_and_eval(
            public_dir=args.public_dir,
            v3_dir=args.v3_dir,
            cache_dir=cache_dir,
            public_emb_cache=args.public_emb_cache,
            v3_emb_cache=args.v3_emb_cache,
            output_dir=args.output_dir,
            official_parquet=args.official_parquet if args.official_parquet.exists() else None,
        )


if __name__ == "__main__":
    main()
