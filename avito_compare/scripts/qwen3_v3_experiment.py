#!/usr/bin/env python3
"""Evaluate Qwen3-Embedding-0.6B + SGDClassifier vs TF-IDF + SGDClassifier on own_v3."""
import argparse
import inspect
import json
import platform
import time
from pathlib import Path

import numpy as np

from .qwen3_embedding_experiment import (
    CLASSIFICATION_PROMPT,
    IDS,
    MODEL_NAME,
    SGD_PARAMS,
    compute_metrics,
    load_official_parquet,
    normalize,
    paired_bootstrap_ci,
    peak_rss_mib,
    sha256,
)


def adapt_v3_rows(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    adapted = []
    for r in rows:
        source = r.get("mcId", 101)
        offered = sorted(set(r.get("offered_microcategories", r.get("labels", []))) & set(IDS))
        if "found_microcategories" in r:
            expected = sorted({e["mcId"] for e in r["found_microcategories"]} & set(IDS))
        else:
            expected = sorted(set(r["labels"]) - ({source} if source in IDS else set()))
        if sorted(set(offered) - ({source} if source in IDS else set())) != expected:
            raise ValueError(f"Inconsistent offered/found labels for item {r.get('item_id')}")
        adapted.append({
            "description": r["description"],
            "mcId": source,
            "labels": offered,
            "evaluation_labels": expected,
            "group_id": r.get("group_id", normalize(r["description"])),
            "item_id": r.get("item_id"),
            "construction": r.get("construction"),
            "length_bin": r.get("length_bin"),
        })
    return adapted


def rows_to_xy(rows):
    y_train = np.asarray([[int(mid in r["labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    y_eval = np.asarray([[int(mid in r["evaluation_labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    sources = np.asarray([r["mcId"] for r in rows], dtype=np.int16)
    return y_train, y_eval, sources


def apply_source_exclusion(probs, thresholds, sources):
    yp = (probs >= thresholds).astype(np.int8)
    for i, src in enumerate(sources):
        if src in IDS:
            yp[i, IDS.index(int(src))] = 0
    return yp


def encode_v3(v3_dir: Path, cache_dir: Path, batch_size: int | None = None, max_seq_length: int = 8192, device: str | None = None, force: bool = False):
    import torch
    import transformers
    import sentence_transformers
    from sentence_transformers import SentenceTransformer

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / "v3_embeddings.npz"
    meta_path = cache_dir / "v3_encoder_meta.json"
    split_hashes = {name: sha256(v3_dir / f"{name}.jsonl") for name in ("train", "validation", "test", "challenge")}

    if not force and cache_npz.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("split_sha256") == split_hashes and meta.get("prompt") == CLASSIFICATION_PROMPT:
            print(f"Reusing cached v3 embeddings in {cache_dir}", flush=True)
            return cache_npz, meta_path

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    if batch_size is None:
        batch_size = 16 if device == "cuda" else 4

    sig = inspect.signature(SentenceTransformer.__init__).parameters
    kw_name = "processor_kwargs" if "processor_kwargs" in sig else "tokenizer_kwargs"
    model = SentenceTransformer(
        MODEL_NAME,
        device=device,
        model_kwargs={"dtype": dtype},
        **{kw_name: {"padding_side": "left"}},
    )
    model.max_seq_length = max_seq_length

    arrays = {}
    encode_times = {}
    for name in ("train", "validation", "test", "challenge"):
        rows = adapt_v3_rows(v3_dir / f"{name}.jsonl")
        texts = [r["description"][:6000] for r in rows]
        unique_texts = list(dict.fromkeys(texts))
        t0 = time.perf_counter()
        uniq_emb = model.encode(
            unique_texts,
            prompt=CLASSIFICATION_PROMPT,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)
        if device == "cuda":
            torch.cuda.synchronize()
        encode_times[name] = time.perf_counter() - t0
        idx_map = {t: i for i, t in enumerate(unique_texts)}
        emb = uniq_emb[[idx_map[t] for t in texts]]
        y_tr, y_ev, src = rows_to_xy(rows)
        arrays[f"X_{name}"] = emb
        arrays[f"y_train_{name}"] = y_tr
        arrays[f"y_eval_{name}"] = y_ev
        arrays[f"sources_{name}"] = src
        print(f"Encoded v3 {name} ({len(rows)} rows, {len(unique_texts)} unique): {encode_times[name]:.2f}s", flush=True)

    np.savez_compressed(cache_npz, **arrays)
    meta = {
        "model_name": MODEL_NAME,
        "prompt": CLASSIFICATION_PROMPT,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor() or "cpu",
        "dtype": str(dtype),
        "batch_size": batch_size,
        "encode_wall_seconds": encode_times,
        "total_encode_wall_seconds": float(sum(encode_times.values())),
        "encode_peak_process_rss_mib": peak_rss_mib(),
        "split_sha256": split_hashes,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "sentence_transformers": sentence_transformers.__version__,
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return cache_npz, meta_path


def select_alpha_cv(X, y, alphas=(1e-4, 1e-5, 3e-6, 1e-6, 3e-7)):
    from sklearn.linear_model import SGDClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import KFold
    from sklearn.multiclass import OneVsRestClassifier

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for alpha in alphas:
        oof = np.zeros_like(y, dtype=float)
        for tr_i, va_i in kf.split(X):
            clf = OneVsRestClassifier(SGDClassifier(**{**SGD_PARAMS, "alpha": alpha})).fit(X[tr_i], y[tr_i])
            oof[va_i] = clf.predict_proba(X[va_i])
        scores[str(alpha)] = float(f1_score(y, oof >= 0.5, average="micro", zero_division=0))
    best_alpha = float(max(scores, key=scores.get))
    return best_alpha, scores


def calibrate_thresholds(y_true, probs, sources):
    from sklearn.metrics import f1_score

    grid = np.arange(0.20, 0.81, 0.025)
    ths = []
    for i in range(len(IDS)):
        mask = sources != IDS[i]
        best_th = max(grid, key=lambda t: f1_score(y_true[mask, i], probs[mask, i] >= t, zero_division=0))
        ths.append(float(best_th))
    return np.asarray(ths)


def score_split(y_eval, probs, sources, thresholds):
    yp = apply_source_exclusion(probs, thresholds, sources)
    # compute_metrics expects probabilities when passed directly, so pass binary yp with threshold 0.5
    m, _ = compute_metrics(y_eval, yp.astype(float), np.full(len(IDS), 0.5))
    return m, yp


def build_tfidf_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.pipeline import FeatureUnion, Pipeline

    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), min_df=2, max_features=18000, sublinear_tf=True, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=28000, sublinear_tf=True, dtype=np.float32)),
    ])
    return Pipeline([
        ("vectorizer", vec),
        ("classifier", OneVsRestClassifier(SGDClassifier(**{**SGD_PARAMS, "alpha": 0.0001}))),
    ])


def train_and_eval_v3(
    v3_dir: Path,
    public_dir: Path,
    public_cache_dir: Path,
    v3_cache_dir: Path,
    output_dir: Path,
    official_parquet: Path | None = None,
):
    from sklearn.linear_model import SGDClassifier
    from sklearn.multiclass import OneVsRestClassifier

    # Load public raw texts & cached embeddings
    pub_rows = {s: adapt_v3_rows(public_dir / f"{s}.jsonl") for s in ("train", "calibration", "validation")}
    v3_rows = {s: adapt_v3_rows(v3_dir / f"{s}.jsonl") for s in ("train", "validation", "test", "challenge")}

    with np.load(public_cache_dir / "embeddings.npz") as zp, np.load(v3_cache_dir / "v3_embeddings.npz") as zv:
        pub_X = {s: zp[f"X_{s}"] for s in ("train", "calibration", "validation")}
        pub_y = {s: zp[f"y_{s}"] for s in ("train", "calibration", "validation")}
        v3_X = {s: zv[f"X_{s}"] for s in ("train", "validation", "test", "challenge")}
        v3_y_tr = {s: zv[f"y_train_{s}"] for s in ("train", "validation", "test", "challenge")}
        v3_y_ev = {s: zv[f"y_eval_{s}"] for s in ("train", "validation", "test", "challenge")}
        v3_src = {s: zv[f"sources_{s}"] for s in ("train", "validation", "test", "challenge")}
        has_off = "X_official" in zp and "y_official" in zp
        off_X = zp["X_official"] if has_off else None
        off_y = zp["y_official"] if has_off else None

    pub_src = {s: np.full(len(pub_rows[s]), 101, dtype=np.int16) for s in ("train", "calibration", "validation")}
    eval_sets = {
        "public_validation_729": (
            [r["description"][:6000] for r in pub_rows["validation"]],
            pub_X["validation"], pub_y["validation"], pub_src["validation"],
            [r["group_id"] for r in pub_rows["validation"]],
            pub_rows["validation"],
        ),
        "v3_test_600": (
            [r["description"][:6000] for r in v3_rows["test"]],
            v3_X["test"], v3_y_ev["test"], v3_src["test"],
            [r["group_id"] for r in v3_rows["test"]],
            v3_rows["test"],
        ),
        "v3_challenge_144": (
            [r["description"][:6000] for r in v3_rows["challenge"]],
            v3_X["challenge"], v3_y_ev["challenge"], v3_src["challenge"],
            [r["group_id"] for r in v3_rows["challenge"]],
            v3_rows["challenge"],
        ),
    }
    if has_off and official_parquet is not None and official_parquet.exists():
        off_texts, _, off_rows = load_official_parquet(official_parquet)
        eval_sets["official_public_208"] = (
            off_texts, off_X, off_y, np.full(len(off_rows), 101, dtype=np.int16),
            [normalize(t) for t in off_texts], off_rows,
        )

    regimes = {
        "public_only": {
            "X_tr": pub_X["train"], "y_tr": pub_y["train"],
            "txt_tr": [r["description"][:6000] for r in pub_rows["train"]],
            "X_cal": pub_X["calibration"], "y_cal": pub_y["calibration"], "src_cal": pub_src["calibration"],
            "txt_cal": [r["description"][:6000] for r in pub_rows["calibration"]],
        },
        "v3_only": {
            "X_tr": v3_X["train"], "y_tr": v3_y_tr["train"],
            "txt_tr": [r["description"][:6000] for r in v3_rows["train"]],
            "X_cal": v3_X["validation"], "y_cal": v3_y_ev["validation"], "src_cal": v3_src["validation"],
            "txt_cal": [r["description"][:6000] for r in v3_rows["validation"]],
        },
        "public_plus_v3": {
            "X_tr": np.vstack([pub_X["train"], v3_X["train"]]),
            "y_tr": np.vstack([pub_y["train"], v3_y_tr["train"]]),
            "txt_tr": [r["description"][:6000] for r in pub_rows["train"]] + [r["description"][:6000] for r in v3_rows["train"]],
            "X_cal": np.vstack([pub_X["calibration"], v3_X["validation"]]),
            "y_cal": np.vstack([pub_y["calibration"], v3_y_ev["validation"]]),
            "src_cal": np.concatenate([pub_src["calibration"], v3_src["validation"]]),
            "txt_cal": [r["description"][:6000] for r in pub_rows["calibration"]] + [r["description"][:6000] for r in v3_rows["validation"]],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"encoder_meta": json.loads((v3_cache_dir / "v3_encoder_meta.json").read_text()), "regimes": {}}
    v3_errors = {}

    for reg_name, reg in regimes.items():
        best_alpha, cv_scores = select_alpha_cv(reg["X_tr"], reg["y_tr"])
        t0 = time.perf_counter()
        qwen_clf = OneVsRestClassifier(SGDClassifier(**{**SGD_PARAMS, "alpha": best_alpha})).fit(reg["X_tr"], reg["y_tr"])
        qwen_fit_s = time.perf_counter() - t0
        qwen_cal_th = calibrate_thresholds(reg["y_cal"], qwen_clf.predict_proba(reg["X_cal"]), reg["src_cal"])

        t0 = time.perf_counter()
        tfidf_clf = build_tfidf_pipeline().fit(reg["txt_tr"], reg["y_tr"])
        tfidf_fit_s = time.perf_counter() - t0
        tfidf_cal_th = calibrate_thresholds(reg["y_cal"], tfidf_clf.predict_proba(reg["txt_cal"]), reg["src_cal"])

        eval_out = {}
        for eval_name, (texts, X_ev, y_ev, src_ev, groups_ev, raw_ev) in eval_sets.items():
            p_q = qwen_clf.predict_proba(X_ev)
            p_t = tfidf_clf.predict_proba(texts)
            mq_05, ypq_05 = score_split(y_ev, p_q, src_ev, np.full(len(IDS), 0.5))
            mq_cal, ypq_cal = score_split(y_ev, p_q, src_ev, qwen_cal_th)
            mt_05, ypt_05 = score_split(y_ev, p_t, src_ev, np.full(len(IDS), 0.5))
            mt_cal, ypt_cal = score_split(y_ev, p_t, src_ev, tfidf_cal_th)
            eval_out[eval_name] = {
                "qwen_0_5": mq_05,
                "qwen_calibrated": mq_cal,
                "tfidf_0_5": mt_05,
                "tfidf_calibrated": mt_cal,
                "bootstrap_qwen05_vs_tfidf05": paired_bootstrap_ci(y_ev, ypt_05, ypq_05, groups_ev),
                "bootstrap_qwencal_vs_tfidf05": paired_bootstrap_ci(y_ev, ypt_05, ypq_cal, groups_ev),
                "bootstrap_qwencal_vs_tfidfcal": paired_bootstrap_ci(y_ev, ypt_cal, ypq_cal, groups_ev),
            }
            if eval_name == "v3_test_600":
                errs = []
                for i, r in enumerate(raw_ev):
                    if not np.array_equal(y_ev[i], ypq_cal[i]):
                        errs.append({
                            "index": i,
                            "construction": r.get("construction"),
                            "length_bin": r.get("length_bin"),
                            "source_mcId": int(r["mcId"]),
                            "gold": [IDS[j] for j in range(len(IDS)) if y_ev[i, j] == 1],
                            "qwen_pred": [IDS[j] for j in range(len(IDS)) if ypq_cal[i, j] == 1],
                            "tfidf_pred": [IDS[j] for j in range(len(IDS)) if ypt_05[i, j] == 1],
                            "probs": {str(IDS[j]): round(float(p_q[i, j]), 4) for j in range(len(IDS))},
                            "text_preview": r["description"][:600],
                        })
                v3_errors[reg_name] = {"total_error_rows": len(errs), "samples": errs[:15]}

        summary["regimes"][reg_name] = {
            "train_rows": int(len(reg["X_tr"])),
            "calibration_rows": int(len(reg["X_cal"])),
            "qwen_cv_alpha_scores": cv_scores,
            "qwen_selected_alpha": best_alpha,
            "qwen_calibrated_thresholds": dict(zip(map(str, IDS), map(lambda x: round(float(x), 4), qwen_cal_th))),
            "tfidf_calibrated_thresholds": dict(zip(map(str, IDS), map(lambda x: round(float(x), 4), tfidf_cal_th))),
            "qwen_sgd_fit_seconds": qwen_fit_s,
            "tfidf_fit_seconds": tfidf_fit_s,
            "evaluations": eval_out,
        }

    (output_dir / "measurement.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "errors_v3_test.json").write_text(json.dumps(v3_errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v3-dir", type=Path, default=Path("/tmp/innopolis-dataset-v3-review/datasets/own_v3"))
    p.add_argument("--public-dir", type=Path, default=Path("output/ngram-comparison/splits"))
    p.add_argument("--public-cache-dir", type=Path, default=Path("output/qwen3-embedding-sgd/cache"))
    p.add_argument("--output-dir", type=Path, default=Path("output/qwen3-v3-experiment"))
    default_parquet = Path("datasets/public/public-test-dataset.parquet")
    if not default_parquet.exists():
        default_parquet = Path("mvp/innoglobalhack_avito/innoglobalhack_avito/public-test-dataset.parquet")
    p.add_argument("--official-parquet", type=Path, default=default_parquet)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--encode-only", action="store_true")
    p.add_argument("--train-eval-only", action="store_true")
    p.add_argument("--force-encode", action="store_true")
    args = p.parse_args()

    v3_cache_dir = args.output_dir / "cache"
    if not args.train_eval_only:
        encode_v3(args.v3_dir, v3_cache_dir, batch_size=args.batch_size, device=args.device, force=args.force_encode)
    if not args.encode_only:
        train_and_eval_v3(
            v3_dir=args.v3_dir,
            public_dir=args.public_dir,
            public_cache_dir=args.public_cache_dir,
            v3_cache_dir=v3_cache_dir,
            output_dir=args.output_dir,
            official_parquet=args.official_parquet if args.official_parquet.exists() else None,
        )


if __name__ == "__main__":
    main()
