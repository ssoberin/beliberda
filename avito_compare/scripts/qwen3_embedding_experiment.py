#!/usr/bin/env python3
"""Experiment: Qwen/Qwen3-Embedding-0.6B (frozen) + OneVsRestClassifier(SGDClassifier)."""
import argparse
import hashlib
import inspect
import json
import platform
import re
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

IDS = (102, 103, 104, 105, 106)
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
CLASSIFICATION_PROMPT = (
    "Instruct: Classify the renovation and repair service microcategories "
    "offered in the given Russian classified ad description\nQuery:"
)
LABELS_RU = {
    "Штукатурные работы": 102,
    "Плиточные работы": 103,
    "Сантехника": 104,
    "Электрика": 105,
    "Поклейка обоев и малярные работы": 106,
}
CAT_NAMES = {102: "Штукатурные работы", 103: "Плиточные работы", 104: "Сантехника", 105: "Электрика", 106: "Поклейка обоев и малярные работы"}
SGD_PARAMS = {"loss": "log_loss", "alpha": 1e-6, "max_iter": 1000, "tol": 1e-3, "random_state": 42}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def peak_rss_mib() -> float:
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def load_jsonl_split(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [r["description"][:6000] for r in rows]
    y = np.asarray([[int(mid in r["labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    return texts, y, rows


def load_official_parquet(path: Path):
    try:
        import duckdb
        raw = duckdb.connect().execute("SELECT title, description, categories FROM read_parquet(?)", [str(path)]).fetchall()
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(path, columns=["title", "description", "categories"])
        raw = list(df.itertuples(index=False, name=None))
    rows = []
    for i, (title, description, categories) in enumerate(raw):
        rows.append({
            "row_index_zero_based": i,
            "title": title,
            "description": description,
            "mcId": 101,
            "evaluation_labels": sorted(LABELS_RU[c] for c in categories),
        })
    texts = [r["description"][:6000] for r in rows]
    y = np.asarray([[int(mid in r["evaluation_labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    return texts, y, rows


def check_splits_integrity(data_dir: Path):
    splits = {}
    split_hashes = {}
    data = {}
    for name in ("train", "calibration", "validation"):
        path = data_dir / f"{name}.jsonl"
        split_hashes[name] = sha256(path)
        texts, y, rows = load_jsonl_split(path)
        normed = set(map(normalize, texts))
        if len(normed) != len(texts):
            raise ValueError(f"Duplicate effective descriptions within {name}")
        splits[name] = normed
        data[name] = (texts, y, rows)
    for a, b in (("train", "calibration"), ("train", "validation"), ("calibration", "validation")):
        if splits[a] & splits[b]:
            raise ValueError(f"Data leakage between {a} and {b}")
    return data, split_hashes


def encode_splits(
    data_dir: Path,
    cache_dir: Path,
    official_parquet: Path | None = None,
    batch_size: int | None = None,
    max_seq_length: int = 8192,
    device: str | None = None,
    force: bool = False,
):
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    data, split_hashes = check_splits_integrity(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / "embeddings.npz"
    meta_path = cache_dir / "encoder_meta.json"

    if not force and cache_npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("split_sha256") == split_hashes
            and meta.get("model_name") == MODEL_NAME
            and meta.get("prompt") == CLASSIFICATION_PROMPT
            and (official_parquet is None or meta.get("official_parquet_sha256") == sha256(official_parquet))
        ):
            print(f"Reusing cached embeddings in {cache_dir}", flush=True)
            return cache_npz, meta_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if batch_size is None:
        batch_size = 16 if device == "cuda" else 4

    sig = inspect.signature(SentenceTransformer.__init__).parameters
    kw_name = "processor_kwargs" if "processor_kwargs" in sig else "tokenizer_kwargs"
    load_t0 = time.perf_counter()
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={"dtype": dtype},
        **{kw_name: {"padding_side": "left"}},
    )
    model.max_seq_length = max_seq_length
    model_load_seconds = time.perf_counter() - load_t0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    arrays = {}
    encode_times = {}
    norm_stats = {}
    total_t0 = time.perf_counter()
    for name in ("train", "calibration", "validation"):
        texts, y, _ = data[name]
        t0 = time.perf_counter()
        emb = model.encode(
            texts,
            prompt=CLASSIFICATION_PROMPT,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_times[name] = time.perf_counter() - t0
        norms = np.linalg.norm(emb, axis=1)
        norm_stats[name] = {"min": float(norms.min()), "max": float(norms.max())}
        arrays[f"X_{name}"] = emb
        arrays[f"y_{name}"] = y
        print(f"Encoded {name} ({len(texts)}): shape={emb.shape}, time={encode_times[name]:.2f}s", flush=True)

    official_hash = None
    if official_parquet is not None and official_parquet.exists():
        official_hash = sha256(official_parquet)
        off_texts, off_y, _ = load_official_parquet(official_parquet)
        t0 = time.perf_counter()
        off_emb = model.encode(
            off_texts,
            prompt=CLASSIFICATION_PROMPT,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_times["official_208"] = time.perf_counter() - t0
        arrays["X_official"] = off_emb
        arrays["y_official"] = off_y
        print(f"Encoded official ({len(off_texts)}): shape={off_emb.shape}, time={encode_times['official_208']:.2f}s", flush=True)

    total_encode_seconds = time.perf_counter() - total_t0

    # Measure single-ad encoder inference latency after warmup.
    val_texts = data["validation"][0]
    for warm_text in val_texts[:3]:
        model.encode([warm_text], prompt=CLASSIFICATION_PROMPT, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
    if device == "cuda":
        torch.cuda.synchronize()

    single_latencies = []
    for sample_text in val_texts[:20]:
        t0 = time.perf_counter()
        model.encode([sample_text], prompt=CLASSIFICATION_PROMPT, batch_size=1, normalize_embeddings=True, show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        single_latencies.append(time.perf_counter() - t0)

    peak_vram_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
    np.savez_compressed(cache_npz, **arrays)
    meta = {
        "model_name": MODEL_NAME,
        "fine_tuned": False,
        "prompt": CLASSIFICATION_PROMPT,
        "normalize_embeddings": True,
        "max_seq_length": max_seq_length,
        "embedding_dim": int(arrays["X_train"].shape[1]),
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "dtype": str(dtype),
        "batch_size": batch_size,
        "model_load_seconds": model_load_seconds,
        "encode_wall_seconds": encode_times,
        "total_encode_wall_seconds": total_encode_seconds,
        "single_ad_encode_seconds_after_warmup": {
            "n_samples": len(single_latencies),
            "mean": float(np.mean(single_latencies)),
            "median": float(np.median(single_latencies)),
            "p95": float(np.percentile(single_latencies, 95)),
        },
        "encode_peak_process_rss_mib": peak_rss_mib(),
        "encode_peak_cuda_allocated_mib": peak_vram_mib,
        "norm_stats": norm_stats,
        "split_sha256": split_hashes,
        "official_parquet_sha256": official_hash,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
            "numpy": np.__version__,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cache_npz, meta_path


def compute_metrics(y_true, y_prob, thresholds):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    yp = (y_prob >= thresholds).astype(np.int8)
    counts = {
        key: int(mask.sum())
        for key, mask in (
            ("tp", (y_true == 1) & (yp == 1)),
            ("fp", (y_true == 0) & (yp == 1)),
            ("fn", (y_true == 1) & (yp == 0)),
            ("tn", (y_true == 0) & (yp == 0)),
        )
    }
    per_label_counts = {
        key: dict(zip(map(str, IDS), map(int, mask.sum(axis=0))))
        for key, mask in (
            ("tp", (y_true == 1) & (yp == 1)),
            ("fp", (y_true == 0) & (yp == 1)),
            ("fn", (y_true == 1) & (yp == 0)),
            ("tn", (y_true == 0) & (yp == 0)),
        )
    }
    return {
        "rows": int(len(y_true)),
        "micro_f1": float(f1_score(y_true, yp, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, yp, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, yp, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, yp, average="micro", zero_division=0)),
        "exact_set_accuracy": float(accuracy_score(y_true, yp)),
        "confusion_counts": counts,
        "confusion_counts_by_category": per_label_counts,
        "f1_by_category": dict(zip(map(str, IDS), map(float, f1_score(y_true, yp, average=None, zero_division=0)))),
    }, yp


def paired_bootstrap_ci(gold, old_pred, new_pred, groups, n_boot=2000, seed=42):
    locations = defaultdict(list)
    for i, g in enumerate(groups):
        locations[g].append(i)
    indices = list(locations.values())
    per_group = []
    for pred in (old_pred, new_pred):
        rc = np.column_stack([
            ((gold == 1) & (pred == 1)).sum(1),
            ((gold == 0) & (pred == 1)).sum(1),
            ((gold == 1) & (pred == 0)).sum(1),
        ])
        per_group.append(np.asarray([rc[idx].sum(0) for idx in indices]))
    sample = np.random.default_rng(seed).integers(0, len(indices), size=(n_boot, len(indices)))
    fs = []
    for counts in per_group:
        tot = counts[sample].sum(1)
        denom = 2 * tot[:, 0] + tot[:, 1] + tot[:, 2]
        fs.append(np.divide(2 * tot[:, 0], denom, out=np.zeros(len(tot), dtype=float), where=denom != 0))
    diff = fs[1] - fs[0]
    return {
        "ci_95": np.quantile(diff, [0.025, 0.975]).tolist(),
        "prob_new_gt_old": float((diff > 0).mean()),
    }


def evaluate_tfidf_baseline(baseline_dir: Path, x_cal, y_cal, x_val, y_val):
    import joblib
    from sklearn.metrics import f1_score

    model = joblib.load(baseline_dir / "tfidf_baseline.joblib")
    pipe = model["pipeline"]
    p_cal = pipe.predict_proba(x_cal)
    p_val = pipe.predict_proba(x_val)
    grid = np.arange(0.20, 0.81, 0.025)
    cal_th = np.asarray([float(grid[int(np.argmax([f1_score(y_cal[:, i], p_cal[:, i] >= t, zero_division=0) for t in grid]))]) for i in range(len(IDS))])
    m_05, pred_05 = compute_metrics(y_val, p_val, np.full(len(IDS), 0.5))
    m_cal, pred_cal = compute_metrics(y_val, p_val, cal_th)

    # Single-ad TF-IDF inference latency after warmup.
    for t in x_val[:3]:
        pipe.predict_proba([t])
    lats = []
    for t in x_val[:100]:
        t0 = time.perf_counter()
        pipe.predict_proba([t])
        lats.append(time.perf_counter() - t0)

    return {
        "thresholds_0_5": m_05,
        "thresholds_calibrated": {"thresholds": dict(zip(map(str, IDS), cal_th.tolist())), **m_cal},
        "pred_0_5": pred_05,
        "pred_cal": pred_cal,
        "single_ad_infer_seconds": {"mean": float(np.mean(lats)), "median": float(np.median(lats))},
    }


def collect_errors(rows, y_true, y_prob, y_pred, tfidf_pred=None, limit=15):
    errors = []
    for i, row in enumerate(rows):
        if np.array_equal(y_true[i], y_pred[i]):
            continue
        gold_ids = [IDS[j] for j in range(len(IDS)) if y_true[i, j] == 1]
        pred_ids = [IDS[j] for j in range(len(IDS)) if y_pred[i, j] == 1]
        fp_ids = [mid for mid in pred_ids if mid not in gold_ids]
        fn_ids = [mid for mid in gold_ids if mid not in pred_ids]
        tfidf_ids = [IDS[j] for j in range(len(IDS)) if tfidf_pred[i, j] == 1] if tfidf_pred is not None else None
        errors.append({
            "index": i,
            "gold": gold_ids,
            "predicted": pred_ids,
            "fp": fp_ids,
            "fn": fn_ids,
            "tfidf_predicted": tfidf_ids,
            "probs": {str(IDS[j]): round(float(y_prob[i, j]), 4) for j in range(len(IDS))},
            "text_preview": row["description"][:800],
            "text_len": len(row["description"]),
        })
    # Pick diverse errors: some FP-only, some FN-only, some mixed, some where TF-IDF was right.
    fp_only = [e for e in errors if e["fp"] and not e["fn"]]
    fn_only = [e for e in errors if e["fn"] and not e["fp"]]
    both = [e for e in errors if e["fp"] and e["fn"]]
    chosen = []
    for bucket in (fp_only, fn_only, both):
        chosen.extend(bucket[:5])
    if len(chosen) < limit:
        seen = {e["index"] for e in chosen}
        for e in errors:
            if e["index"] not in seen:
                chosen.append(e)
            if len(chosen) >= limit:
                break
    return chosen[:limit], len(errors)


def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    baseline_dir: Path,
    official_parquet: Path | None = None,
    alpha: float = SGD_PARAMS["alpha"],
):
    import joblib
    import scipy
    import sklearn
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.multiclass import OneVsRestClassifier

    data, split_hashes = check_splits_integrity(data_dir)
    cache_npz = cache_dir / "embeddings.npz"
    meta_path = cache_dir / "encoder_meta.json"
    if not cache_npz.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Cached embeddings not found in {cache_dir}. Run encoding first.")

    encoder_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if encoder_meta["split_sha256"] != split_hashes:
        raise ValueError("Cached split_sha256 does not match current split files")

    with np.load(cache_npz) as z:
        X_train, y_train = z["X_train"], z["y_train"]
        X_cal, y_cal = z["X_calibration"], z["y_calibration"]
        X_val, y_val = z["X_validation"], z["y_validation"]
        has_official = "X_official" in z and "y_official" in z
        X_off = z["X_official"] if has_official else None
        y_off = z["y_official"] if has_official else None

    sgd_params = {**SGD_PARAMS, "alpha": alpha}
    clf = OneVsRestClassifier(SGDClassifier(**sgd_params))
    t0, cpu0 = time.perf_counter(), time.process_time()
    clf.fit(X_train, y_train)
    fit_wall = time.perf_counter() - t0
    fit_cpu = time.process_time() - cpu0

    p_cal = clf.predict_proba(X_cal)
    p_val = clf.predict_proba(X_val)

    # Calibrate per-category thresholds strictly on calibration split (inside train, never test).
    grid = np.arange(0.20, 0.81, 0.025)
    cal_thresholds = []
    for i in range(len(IDS)):
        scores = [f1_score(y_cal[:, i], p_cal[:, i] >= th, zero_division=0) for th in grid]
        cal_thresholds.append(float(grid[int(np.argmax(scores))]))
    cal_thresholds = np.asarray(cal_thresholds)

    m_05, pred_05 = compute_metrics(y_val, p_val, np.full(len(IDS), 0.5))
    m_cal, pred_cal = compute_metrics(y_val, p_val, cal_thresholds)

    # Single-ad classifier inference latency after warmup.
    for i in range(3):
        clf.predict_proba(X_val[i : i + 1])
    sgd_lats = []
    for i in range(min(200, len(X_val))):
        t0 = time.perf_counter()
        clf.predict_proba(X_val[i : i + 1])
        sgd_lats.append(time.perf_counter() - t0)

    x_cal_txt, _, _ = data["calibration"]
    x_val_txt, _, val_rows = data["validation"]
    tfidf_res = evaluate_tfidf_baseline(baseline_dir, x_cal_txt, y_cal, x_val_txt, y_val)

    groups_val = [normalize(r["description"][:6000]) for r in val_rows]
    boot_vs_05 = paired_bootstrap_ci(y_val, tfidf_res["pred_0_5"], pred_05, groups_val)
    boot_cal_vs_05 = paired_bootstrap_ci(y_val, tfidf_res["pred_0_5"], pred_cal, groups_val)
    boot_cal_vs_cal = paired_bootstrap_ci(y_val, tfidf_res["pred_cal"], pred_cal, groups_val)

    errors_sample, total_error_rows = collect_errors(
        val_rows, y_val, p_val, pred_cal, tfidf_pred=tfidf_res["pred_0_5"], limit=15
    )

    official_eval = None
    if has_official and official_parquet is not None and official_parquet.exists():
        _, _, off_rows = load_official_parquet(official_parquet)
        p_off = clf.predict_proba(X_off)
        off_m_05, off_pred_05 = compute_metrics(y_off, p_off, np.full(len(IDS), 0.5))
        off_m_cal, off_pred_cal = compute_metrics(y_off, p_off, cal_thresholds)

        import joblib as _jl
        tfidf_model = _jl.load(baseline_dir / "tfidf_baseline.joblib")
        p_off_tfidf = tfidf_model["pipeline"].predict_proba([r["description"][:6000] for r in off_rows])
        off_tfidf_05, _ = compute_metrics(y_off, p_off_tfidf, np.full(len(IDS), 0.5))

        # Clean unique subset (matching paired_data_evaluation.py policy)
        label_groups = defaultdict(set)
        for r in off_rows:
            label_groups[normalize(r["description"][:6000])].add(tuple(r["evaluation_labels"]))
        conflicts = {k for k, v in label_groups.items() if len(v) > 1}
        seen_public = {normalize(r["description"][:6000]) for split_name in ("train", "calibration", "validation") for r in data[split_name][2]}
        clean_idx = []
        seen_clean = set()
        for idx, r in enumerate(off_rows):
            k = normalize(r["description"][:6000])
            if k in conflicts or k in seen_public or k in seen_clean:
                continue
            seen_clean.add(k)
            clean_idx.append(idx)
        clean_idx = np.asarray(clean_idx)
        clean_m_05, _ = compute_metrics(y_off[clean_idx], p_off[clean_idx], np.full(len(IDS), 0.5))
        clean_m_cal, _ = compute_metrics(y_off[clean_idx], p_off[clean_idx], cal_thresholds)
        clean_tfidf_05, _ = compute_metrics(y_off[clean_idx], p_off_tfidf[clean_idx], np.full(len(IDS), 0.5))
        official_eval = {
            "official_208": {"qwen_0_5": off_m_05, "qwen_calibrated": off_m_cal, "tfidf_0_5": off_tfidf_05},
            "official_clean_unique_114": {"qwen_0_5": clean_m_05, "qwen_calibrated": clean_m_cal, "tfidf_0_5": clean_tfidf_05},
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "qwen3_sgd.joblib"
    joblib.dump(
        {
            "classifier": clf,
            "ids": IDS,
            "thresholds": cal_thresholds,
            "encoder_model": MODEL_NAME,
            "prompt": CLASSIFICATION_PROMPT,
        },
        artifact_path,
    )
    np.savez_compressed(
        output_dir / "predictions.npz",
        gold=y_val,
        prob=p_val,
        pred_0_5=pred_05,
        pred_calibrated=pred_cal,
        tfidf_pred_0_5=tfidf_res["pred_0_5"],
    )
    (output_dir / "errors_sample.json").write_text(
        json.dumps({"total_error_rows": total_error_rows, "samples": errors_sample}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result = {
        "architecture": f"{MODEL_NAME} (frozen) + OneVsRestClassifier(SGDClassifier)",
        "encoder": encoder_meta,
        "classifier_params": clf.estimator.get_params(),
        "calibrated_thresholds": dict(zip(map(str, IDS), map(lambda x: round(float(x), 4), cal_thresholds))),
        "sgd_fit_wall_seconds": fit_wall,
        "sgd_fit_cpu_seconds": fit_cpu,
        "sgd_single_ad_infer_seconds": {
            "n_samples": len(sgd_lats),
            "mean": float(np.mean(sgd_lats)),
            "median": float(np.median(sgd_lats)),
        },
        "train_eval_peak_process_rss_mib": peak_rss_mib(),
        "classifier_artifact_bytes": artifact_path.stat().st_size,
        "validation_729": {
            "qwen_0_5": m_05,
            "qwen_calibrated": m_cal,
            "tfidf_baseline_0_5": tfidf_res["thresholds_0_5"],
            "tfidf_baseline_calibrated": tfidf_res["thresholds_calibrated"],
            "tfidf_single_ad_infer_seconds": tfidf_res["single_ad_infer_seconds"],
            "paired_bootstrap_qwen05_vs_tfidf05": boot_vs_05,
            "paired_bootstrap_qwencal_vs_tfidf05": boot_cal_vs_05,
            "paired_bootstrap_qwencal_vs_tfidfcal": boot_cal_vs_cal,
            "qwen_cal_fixes_vs_tfidf05": int(((tfidf_res["pred_0_5"] != y_val) & (pred_cal == y_val)).sum()),
            "qwen_cal_breaks_vs_tfidf05": int(((tfidf_res["pred_0_5"] == y_val) & (pred_cal != y_val)).sum()),
        },
        "official_test": official_eval,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
    }
    (output_dir / "measurement.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, default=Path("output/ngram-comparison/splits"))
    p.add_argument("--output-dir", type=Path, default=Path("output/qwen3-embedding-sgd"))
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--baseline-dir", type=Path, default=Path("output/tfidf-feature-experiments/char_wb"))
    default_parquet = Path("datasets/public/public-test-dataset.parquet")
    if not default_parquet.exists():
        default_parquet = Path("mvp/innoglobalhack_avito/innoglobalhack_avito/public-test-dataset.parquet")
    p.add_argument("--official-parquet", type=Path, default=default_parquet)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-length", type=int, default=8192)
    p.add_argument("--device", default=None)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--train-eval-only", action="store_true")
    p.add_argument("--force-encode", action="store_true")
    p.add_argument("--alpha", type=float, default=SGD_PARAMS["alpha"])
    args = p.parse_args()

    cache_dir = args.cache_dir or (args.output_dir / "cache")
    if not args.train_eval_only:
        encode_splits(
            data_dir=args.data_dir,
            cache_dir=cache_dir,
            official_parquet=args.official_parquet if args.official_parquet.exists() else None,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            device=args.device,
            force=args.force_encode,
        )
    if not args.encode_only:
        train_and_evaluate(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            cache_dir=cache_dir,
            baseline_dir=args.baseline_dir,
            official_parquet=args.official_parquet if args.official_parquet.exists() else None,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
