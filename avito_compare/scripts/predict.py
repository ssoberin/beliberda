#!/usr/bin/env python3
"""Demo-compatible JSON response with title/text null, based on TF-IDF model."""
import argparse
import json
from pathlib import Path
import joblib


def predict(model, description, source_mc_id):
    if source_mc_id not in range(101, 107):
        raise ValueError("mcId must be one of 101..106")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description is required")
    probs = model["pipeline"].predict_proba([description[:6000]])[0]
    return {"drafts": [{"mcId": mid, "title": None, "text": None}
                       for mid, prob, threshold in zip(model["ids"], probs, model["thresholds"])
                       if mid != source_mc_id and prob >= threshold]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=Path("artifacts/tfidf_baseline.joblib"))
    p.add_argument("--description", required=True)
    p.add_argument("--mcId", type=int, required=True)
    args = p.parse_args()
    model = joblib.load(args.model)  # Only load locally-created trusted joblib files.
    print(json.dumps(predict(model, args.description, args.mcId), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
