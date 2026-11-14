#!/usr/bin/env python3
"""Train TF-IDF multi-label classifier, calibrate scalar decision thresholds, and evaluate."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

IDS = (102, 103, 104, 105, 106)


def load(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    X = [r["description"][:6000] for r in rows]
    Y = np.asarray([[int(mid in r["labels"]) for mid in IDS] for r in rows], dtype=np.int8)
    return X, Y


def report(y_true, y_prob, thresholds):
    yp = (y_prob >= thresholds).astype(int)
    return {
        "micro_f1": round(float(f1_score(y_true, yp, average="micro", zero_division=0)), 5),
        "micro_precision": round(float(precision_score(y_true, yp, average="micro", zero_division=0)), 5),
        "micro_recall": round(float(recall_score(y_true, yp, average="micro", zero_division=0)), 5),
        "by_category_f1": {str(mid): round(float(f1_score(y_true[:, i], yp[:, i], zero_division=0)), 5) for i, mid in enumerate(IDS)},
        "actual_positives": dict(zip(map(str, IDS), map(int, y_true.sum(axis=0)))),
        "predicted_positives": dict(zip(map(str, IDS), map(int, yp.sum(axis=0)))),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=Path("data/processed"), type=Path)
    p.add_argument("--output-dir", default=Path("artifacts"), type=Path)
    p.add_argument("--word-ngram-max", type=int, choices=(2, 3), default=2,
                   help="Maximum word n-gram length; default preserves the bigram baseline")
    p.add_argument("--char-analyzer", choices=("char", "char_wb"), default="char")
    p.add_argument("--no-max-features", action="store_true",
                   help="Remove vocabulary caps from both TF-IDF vectorizers")
    p.add_argument("--word-max-features", type=int, default=18000)
    p.add_argument("--char-max-features", type=int, default=28000)
    p.add_argument("--sublinear-tf", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    x_train, y_train = load(args.data_dir / "train.jsonl")
    x_cal, y_cal = load(args.data_dir / "calibration.jsonl")
    x_val, y_val = load(args.data_dir / "validation.jsonl")

    vec = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, args.word_ngram_max), min_df=2, max_features=None if args.no_max_features else args.word_max_features, sublinear_tf=args.sublinear_tf, dtype=np.float32)),
        ("char", TfidfVectorizer(analyzer=args.char_analyzer, ngram_range=(3, 5), min_df=3, max_features=None if args.no_max_features else args.char_max_features, sublinear_tf=args.sublinear_tf, dtype=np.float32)),
    ])
    model = Pipeline([
        ("vectorizer", vec),
        ("classifier", OneVsRestClassifier(SGDClassifier(loss="log_loss", alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42))),
    ])
    model.fit(x_train, y_train)
    p_cal = model.predict_proba(x_cal)
    p_val = model.predict_proba(x_val)
    # Calibrate one threshold per label on calibration set only, never on validation.
    grid = np.arange(0.20, 0.81, 0.025)
    thresholds = []
    for i in range(len(IDS)):
        scores = [f1_score(y_cal[:, i], p_cal[:, i] >= th, zero_division=0) for th in grid]
        thresholds.append(float(grid[int(np.argmax(scores))]))
    thresholds = np.asarray(thresholds)
    result = {
        "train_size": len(x_train), "calibration_size": len(x_cal), "validation_size": len(x_val),
        "thresholds": dict(zip(map(str, IDS), map(lambda x: round(float(x), 3), thresholds))),
        "validation_default_0_5": report(y_val, p_val, 0.5),
        "validation_calibrated": report(y_val, p_val, thresholds),
        "note": "Weak GPT-generated labels. This is not a manually verified gold result.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "ids": IDS, "thresholds": np.full(len(IDS), 0.5)}, args.output_dir / "tfidf_baseline.joblib")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
