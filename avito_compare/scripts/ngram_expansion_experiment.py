"""Run the word n-gram expansion experiment on the fixed mixed public + v2 splits."""
import argparse
import collections
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from . import prepare_dataset, train_baseline

IDS = train_baseline.IDS

CONFIGS = [
    ("ref_46k", 3, 18000, 28000),
    ("A_1_3_uncapped", 3, None, 28000),
    ("B_1_4_uncapped", 4, None, 28000),
    ("C_1_5_uncapped", 5, None, 28000),
    ("D_1_6_uncapped", 6, None, 28000),
]

EXPECTED_SPLIT_SHA256 = {
    "train": "f75b362db045775708355861447ad3a27f8b6690e24ffbded3efffa17c87eb39",
    "calibration": "020e46a7fe401647b5bd7e3cba8126c128d229a30809242a903b48e742a919ed",
    "validation": "e73489cbe2c2b883946293789d0eea70d90cf0446eb8580664282ccc30822c90",
    "challenge": "7c25264f2125ae03072956e14bf98780ed4899c72906f2115aada3ecd8938241",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rss_mib(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except FileNotFoundError:
        pass
    return 0.0


def compute_metrics(gold: np.ndarray, pred: np.ndarray) -> dict:
    if len(gold) == 0:
        return {"rows": 0}
    tp = int(((gold == 1) & (pred == 1)).sum())
    fp = int(((gold == 0) & (pred == 1)).sum())
    fn = int(((gold == 1) & (pred == 0)).sum())
    tn = int(((gold == 0) & (pred == 0)).sum())
    return {
        "rows": int(len(gold)),
        "micro_f1": float(f1_score(gold, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(gold, pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(gold, pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(gold, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(gold, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(gold, pred, average="macro", zero_division=0)),
        "exact_set_accuracy": float(accuracy_score(gold, pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "empty_predictions": int((pred.sum(axis=1) == 0).sum()),
        "empty_on_positive_rows": int(((pred.sum(axis=1) == 0) & (gold.sum(axis=1) > 0)).sum()),
        "positive_rows": int((gold.sum(axis=1) > 0).sum()),
        "gold_empty_rows": int((gold.sum(axis=1) == 0).sum()),
        "f1_by_category": dict(zip(map(str, IDS), map(float, f1_score(gold, pred, average=None, zero_division=0)))),
    }


def predict_source_aware(pipeline: Pipeline, rows: list[dict], thresholds: np.ndarray):
    texts = [r["description"][:6000] for r in rows]
    probs = pipeline.predict_proba(texts)
    pred = (probs >= thresholds).astype(np.int8)
    gold = np.asarray([[int(mid in r["evaluation_labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    for idx, row in enumerate(rows):
        mc_id = row.get("mcId", 101)
        if mc_id in IDS:
            pred[idx, IDS.index(mc_id)] = 0
    return gold, pred, probs


def run_worker(args) -> None:
    data_dir: Path = args.data_dir
    out_dir: Path = args.run_dir
    if out_dir.exists():
        raise ValueError(f"Output directory already exists: {out_dir}")

    split_hashes = {}
    for name in ("train", "calibration", "validation", "challenge"):
        path = data_dir / f"{name}.jsonl"
        split_hashes[name] = sha256(path)
        if split_hashes[name] != EXPECTED_SPLIT_SHA256[name]:
            raise ValueError(f"Split hash mismatch for {name}: {split_hashes[name]} != {EXPECTED_SPLIT_SHA256[name]}")

    train_rows = read_jsonl(data_dir / "train.jsonl")
    val_rows = read_jsonl(data_dir / "validation.jsonl")
    challenge_rows = read_jsonl(data_dir / "challenge.jsonl")

    x_train = [r["description"][:6000] for r in train_rows]
    y_train = np.asarray([[int(mid in r["labels"]) for mid in IDS] for r in train_rows], dtype=np.int8)

    # Verify zero normalized effective intersections between train and validation/challenge
    train_norm = {prepare_dataset.normalize(t) for t in x_train}
    val_norm = {prepare_dataset.normalize(r["description"][:6000]) for r in val_rows}
    if train_norm & val_norm:
        raise ValueError("Normalized text leakage detected between train and validation")

    word_max = None if args.word_max_features <= 0 else args.word_max_features
    char_max = None if args.char_max_features <= 0 else args.char_max_features

    vec = FeatureUnion([
        ("word", TfidfVectorizer(
            ngram_range=(1, args.word_ngram_max),
            min_df=2,
            max_features=word_max,
            sublinear_tf=True,
            dtype=np.float32,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=3,
            max_features=char_max,
            sublinear_tf=True,
            dtype=np.float32,
        )),
    ])
    clf = OneVsRestClassifier(
        SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
    )
    pipeline = Pipeline([("vectorizer", vec), ("classifier", clf)])

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    pipeline.fit(x_train, y_train)
    fit_wall = time.perf_counter() - wall_start
    fit_cpu = time.process_time() - cpu_start
    fit_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    thresholds = np.full(len(IDS), 0.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_dir / "tfidf_baseline.joblib"
    joblib.dump({"pipeline": pipeline, "ids": IDS, "thresholds": thresholds}, artifact_path)

    val_gold, val_pred, val_prob = predict_source_aware(pipeline, val_rows, thresholds)
    chal_gold, chal_pred, chal_prob = predict_source_aware(pipeline, challenge_rows, thresholds)
    train_eval_gold, train_eval_pred, train_prob = predict_source_aware(pipeline, train_rows, thresholds)
    eval_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    transformers = dict(pipeline.named_steps["vectorizer"].transformer_list)
    word_vocab = transformers["word"].vocabulary_
    char_vocab = transformers["char"].vocabulary_
    word_by_len = collections.Counter(token.count(" ") + 1 for token in word_vocab)

    val_matrix = pipeline.named_steps["vectorizer"].transform(
        [r["description"][:6000] for r in val_rows]
    ).tocsr()
    val_nnz_per_row = np.diff(val_matrix.indptr)
    word_val_matrix = transformers["word"].transform([r["description"][:6000] for r in val_rows]).tocsr()
    word_nnz_per_row = np.diff(word_val_matrix.indptr)

    public_mask = np.asarray([r["dataset"] == "public" for r in val_rows])
    v2_mask = np.asarray([r["dataset"] == "v2" for r in val_rows])
    train_public_mask = np.asarray([r["dataset"] == "public" for r in train_rows])
    train_v2_mask = np.asarray([r["dataset"] == "v2" for r in train_rows])

    measurement = {
        "name": args.name,
        "word_ngram_range": [1, args.word_ngram_max],
        "char_analyzer": "char_wb",
        "char_ngram_range": [3, 5],
        "max_features": {"word": word_max, "char": char_max},
        "sublinear_tf": True,
        "exclude_source_category": True,
        "thresholds": thresholds.tolist(),
        "dimensions": {
            "word": len(word_vocab),
            "char": len(char_vocab),
            "total": len(word_vocab) + len(char_vocab),
            "word_by_ngram_length": {str(k): int(v) for k, v in sorted(word_by_len.items())},
        },
        "sparsity_on_validation": {
            "total_nnz_mean": float(val_nnz_per_row.mean()),
            "total_nnz_median": float(np.median(val_nnz_per_row)),
            "word_nnz_mean": float(word_nnz_per_row.mean()),
            "word_nnz_median": float(np.median(word_nnz_per_row)),
            "word_zero_vectors": int((word_nnz_per_row == 0).sum()),
            "total_zero_vectors": int((val_nnz_per_row == 0).sum()),
        },
        "fit_wall_seconds": fit_wall,
        "fit_cpu_seconds": fit_cpu,
        "fit_peak_process_rss_mib": fit_peak_rss,
        "eval_peak_process_rss_mib": eval_peak_rss,
        "artifact_bytes": artifact_path.stat().st_size,
        "evaluations": {
            "combined_evaluation": compute_metrics(val_gold, val_pred),
            "public_validation": compute_metrics(val_gold[public_mask], val_pred[public_mask]),
            "v2_test": compute_metrics(val_gold[v2_mask], val_pred[v2_mask]),
            "challenge": compute_metrics(chal_gold, chal_pred),
        },
        "train_diagnostic": {
            "combined_train": compute_metrics(train_eval_gold, train_eval_pred),
            "public_train": compute_metrics(train_eval_gold[train_public_mask], train_eval_pred[train_public_mask]),
            "v2_train": compute_metrics(train_eval_gold[train_v2_mask], train_eval_pred[train_v2_mask]),
        },
        "split_sha256": split_hashes,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "estimator_parameters": clf.estimator.get_params(),
        "vectorizer_parameters": {
            name: {k: str(v) if isinstance(v, type) else v for k, v in vec_step.get_params().items()}
            for name, vec_step in transformers.items()
        },
    }
    (out_dir / "measurement.json").write_text(json.dumps(measurement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(
        out_dir / "predictions.npz",
        val_gold=val_gold,
        val_pred=val_pred,
        val_prob=val_prob,
        challenge_gold=chal_gold,
        challenge_pred=chal_pred,
        challenge_prob=chal_prob,
        train_eval_gold=train_eval_gold,
        train_eval_pred=train_eval_pred,
        train_prob=train_prob,
    )
    print(
        f"WORKER_DONE {args.name}: word_dim={len(word_vocab)} total_dim={len(word_vocab)+len(char_vocab)} "
        f"pub_f1={measurement['evaluations']['public_validation']['micro_f1']:.6f} "
        f"v2_f1={measurement['evaluations']['v2_test']['micro_f1']:.6f} "
        f"fit={fit_wall:.2f}s rss={fit_peak_rss:.1f}MiB",
        flush=True,
    )


def run_orchestrator(args) -> None:
    out_root: Path = args.output_dir
    runs_dir = out_root / "runs"
    logs_dir = out_root / "logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    statuses = {}
    summary = {"configs": [], "runs": {}}

    for name, ngram_max, word_max, char_max in CONFIGS:
        run_dir = runs_dir / name
        meas_path = run_dir / "measurement.json"
        if args.resume and meas_path.exists():
            measured = json.loads(meas_path.read_text(encoding="utf-8"))
            summary["runs"][name] = measured
            summary["configs"].append(name)
            print(f"REUSE {name}", flush=True)
            continue
        cmd = [
            sys.executable,
            "-m",
            "avito_compare.scripts.ngram_expansion_experiment",
            "--worker",
            "--name",
            name,
            "--data-dir",
            str(args.data_dir),
            "--run-dir",
            str(run_dir),
            "--word-ngram-max",
            str(ngram_max),
            "--word-max-features",
            str(0 if word_max is None else word_max),
            "--char-max-features",
            str(char_max),
        ]
        started = time.monotonic()
        peak = 0.0
        print(f"START {name}", flush=True)
        with (logs_dir / f"{name}.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                peak = max(peak, rss_mib(proc.pid))
                time.sleep(0.1)
        wall = time.monotonic() - started
        statuses[name] = {
            "exit_code": proc.returncode,
            "monitored_peak_rss_mib": peak,
            "process_wall_seconds": wall,
            "command": cmd,
        }
        if proc.returncode != 0:
            raise RuntimeError(f"Worker {name} failed with exit code {proc.returncode}; see {logs_dir / (name + '.log')}")
        measured = json.loads(meas_path.read_text(encoding="utf-8"))
        measured["monitored_peak_rss_mib"] = peak
        meas_path.write_text(json.dumps(measured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary["runs"][name] = measured
        summary["configs"].append(name)
        print(
            f"DONE {name}: pub_f1={measured['evaluations']['public_validation']['micro_f1']:.6f} "
            f"v2_f1={measured['evaluations']['v2_test']['micro_f1']:.6f} "
            f"dims={measured['dimensions']['total']} fit={measured['fit_wall_seconds']:.2f}s "
            f"rss={measured['fit_peak_process_rss_mib']:.1f}MiB",
            flush=True,
        )

    # Verify ref_46k matches existing mixed char_wb reference
    ref_eval_path = Path("output/mixed-charwb-comparison/evaluation.json")
    if ref_eval_path.exists():
        ref_eval = json.loads(ref_eval_path.read_text(encoding="utf-8"))["models"]["char_wb"]
        repro = summary["runs"]["ref_46k"]["evaluations"]
        for split_key in ("combined_evaluation", "public_validation", "v2_test", "challenge"):
            for metric_key in ("micro_f1", "macro_f1", "micro_precision", "micro_recall", "fp", "fn"):
                val_ref = ref_eval[split_key][metric_key]
                val_new = repro[split_key][metric_key]
                if isinstance(val_ref, float):
                    if abs(val_ref - val_new) > 1e-12:
                        raise ValueError(f"ref_46k mismatch on {split_key}.{metric_key}: {val_new} vs {val_ref}")
                elif val_ref != val_new:
                    raise ValueError(f"ref_46k mismatch on {split_key}.{metric_key}: {val_new} vs {val_ref}")
        summary["ref_46k_verified_against_mixed_charwb"] = True
        print("VERIFIED ref_46k matches output/mixed-charwb-comparison/evaluation.json (char_wb) exactly!", flush=True)

    (out_root / "status.json").write_text(json.dumps(statuses, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_root / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--name", type=str)
    parser.add_argument("--data-dir", type=Path, default=Path("output/ngram-expansion-experiment/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/ngram-expansion-experiment"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--word-ngram-max", type=int, default=3)
    parser.add_argument("--word-max-features", type=int, default=18000)
    parser.add_argument("--char-max-features", type=int, default=28000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
