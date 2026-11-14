"""Build expanded split-preserving corpus; no external packages or network."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

from generate import NAMES, SERVICES, generate, make_row, labels
from expanded_cases import expanded_rows, PROTECTED


def validate(splits):
    ids, texts, families, groups = set(), {}, {}, {}
    stats = {}
    for split, rows in splits.items():
        by_group = defaultdict(list)
        by_id = {r["item_id"]: r for r in rows}
        for r in rows:
            assert r["item_id"] not in ids, r["item_id"]
            ids.add(r["item_id"])
            key = " ".join(r["description"].casefold().split())
            assert key not in texts, (r["item_id"], texts.get(key))
            texts[key] = r["item_id"]
            assert 0 < len(r["description"]) <= 12000
            assert r["mcId"] in NAMES
            assert set(r["offered_microcategories"]) <= set(NAMES)
            assert r["found_microcategories"] == labels(set(r["offered_microcategories"]) - {r["mcId"]})
            assert families.setdefault(r["type"], split) == split
            assert groups.setdefault(r["group_id"], split) == split
            by_group[r["group_id"]].append(r)
            if "clean_item_id" in r:
                clean = by_id[r["clean_item_id"]]
                assert clean["augmentation"] == "clean"
                for field in ("mcId", "found_microcategories", "offered_microcategories", "group_id", "variant"):
                    assert r[field] == clean[field]
                assert [x.lower() for x in PROTECTED.findall(r["description"])] == [x.lower() for x in PROTECTED.findall(clean["description"])]
                if r["augmentation"] != "clean":
                    assert r["description"] != clean["description"]
        for group in by_group.values():
            modes = defaultdict(list)
            for row in group:
                modes[row.get("augmentation", "clean")].append(row)
            for pair in modes.values():
                assert len(pair) == 2
                assert pair[0]["mcId"] == pair[1]["mcId"]
                assert pair[0]["found_microcategories"] != pair[1]["found_microcategories"]
        def counts(values):
            return dict(sorted(Counter(values).items()))
        stats[split] = {
            "rows": len(rows), "semantic_groups": len(by_group),
            "empty_targets": sum(not r["found_microcategories"] for r in rows),
            "target_counts": counts(m["mcId"] for r in rows for m in r["found_microcategories"]),
            "source_counts": counts(r["mcId"] for r in rows),
            "target_cardinality": counts(len(r["found_microcategories"]) for r in rows),
            "augmentations": counts(r.get("augmentation", "clean") for r in rows),
            "topics": counts(r.get("topic", "original_semantics") for r in rows),
            "negation_depth": counts(r.get("negation_depth", 0) for r in rows),
        }
        assert set(stats[split]["target_counts"]) == set(NAMES)
    return stats


def build(seed=2026):
    splits = generate(seed)
    extra = expanded_rows(NAMES, SERVICES, make_row)
    rng = random.Random(seed)
    for split, rows in extra.items():
        splits[split].extend(rows)
        rng.shuffle(splits[split])
    return splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("v2"))
    args = parser.parse_args()
    splits = build(args.seed)
    stats = validate(splits)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, rows in splits.items():
        output = args.output_dir / f"{split}.jsonl"
        output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        hashes[output.name] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {"version": 2, "seed": args.seed, "total": sum(len(rows) for rows in splits.values()),
                "splits": stats, "sha256": hashes,
                "lineage": "Includes v1 hard_dataset plus new synthetic scenarios. Do not concatenate v1 and v2. Legacy uncertain corpora and official test are not imported."}
    (args.output_dir / "stats.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": manifest["total"], **{k: v["rows"] for k, v in stats.items()}}))


if __name__ == "__main__":
    main()
