#!/usr/bin/env python3
"""Prepare leakage-resistant splits from provided weakly labeled Avito JSONL."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

IDS = (102, 103, 104, 105, 106)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def load_unique(path: Path):
    groups = {}
    counts = Counter()
    invalid = 0
    total = 0
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            description = row.get("description")
            if not isinstance(description, str) or not description.strip():
                invalid += 1
                continue
            labels = set()
            for item in row.get("found_microcategories", []):
                cat = item.get("mcId")
                if cat not in IDS:
                    raise ValueError(f"Invalid mcId {cat!r} on line {n}")
                labels.add(cat)
                counts[cat] += 1
            key = normalize(description)
            existing = groups.get(key)
            if existing is None:
                groups[key] = {"description": description, "labels": sorted(labels), "source_count": 1}
            else:
                if existing["labels"] != sorted(labels):
                    raise ValueError(f"Conflicting labels on normalized duplicate, line {n}")
                existing["source_count"] += 1
    return list(groups.values()), {"source_rows": total, "invalid_rows": invalid, "label_occurrences": dict(counts)}


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", default=Path("data/processed"), type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records, stats = load_unique(args.input)
    if len(records) < 20:
        raise ValueError("Too few unique examples")
    # Splitting unique descriptions makes the same text unable to cross splits.
    train, holdout = train_test_split(records, test_size=0.30, random_state=args.seed)
    calibration, validation = train_test_split(holdout, test_size=0.50, random_state=args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("calibration", calibration), ("validation", validation)):
        write_jsonl(args.output / f"{name}.jsonl", rows)
    result = {**stats, "unique_normalized_descriptions": len(records), "splits": {"train": len(train), "calibration": len(calibration), "validation": len(validation)}, "seed": args.seed}
    (args.output / "stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
