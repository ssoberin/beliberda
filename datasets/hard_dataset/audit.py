"""Read every record in repository datasets; flag structural and review issues.

Semantic flags are review candidates, not automatic relabeling decisions.
Optional pyarrow is required to audit Parquet contents (no conversion to training).
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from generate import NAMES

ROOT = Path(__file__).resolve().parents[2]
IDS = set(range(101, 107))


def normalize(text):
    return " ".join(text.casefold().split())


def audit(root, exclude_v2=False):
    paths = []
    for folder in ("datasets/public", "datasets/distillate_dataset", "datasets/hard_dataset", "datasets/hard_negation_300", "datasets/hard_illiterate_200"):
        paths.extend(p for p in (root / folder).rglob("*") if p.suffix in (".jsonl", ".parquet"))
    if exclude_v2:
        paths = [p for p in paths if "v2" not in p.relative_to(root).parts]
    report = {"files": {}, "findings": [], "scope": "All rows; structural checks plus semantic risk heuristics, not a full human relabeling."}
    seen = defaultdict(list)
    source_ids = set()
    derived = []
    for path in sorted(paths):
        rel = path.relative_to(root).as_posix()
        stats = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": 0, "label_counts": Counter(), "flags": Counter()}
        report["files"][rel] = stats
        def issue(line, row, code, severity="review"):
            stats["flags"][code] += 1
            report["findings"].append({"file": rel, "line": line, "item_id": row.get("item_id"), "code": code, "severity": severity})
        if path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError:
                stats["not_read"] = "pyarrow unavailable"
                continue
            table = pq.read_table(path)
            stats["columns"] = table.column_names
            stats["adapter"] = "Named categories mapped to IDs; implicit source 101 per README-files.txt; row number is audit-only ID."
            mapped = []
            by_name = {name: mid for mid, name in NAMES.items()}
            for number, row in enumerate(table.to_pylist(), 1):
                row["item_id"] = f"parquet-row-{number}"
                row["mcId"] = 101
                row["found_microcategories"] = [{"mcId": by_name.get(name), "label": name} for name in row["categories"]]
                mapped.append(row)
            iterator = enumerate(mapped, 1)
        else:
            def json_rows():
                with path.open(encoding="utf-8") as stream:
                    for line, text in enumerate(stream, 1):
                        if not text.strip():
                            issue(line, {}, "blank_line", "structural")
                            continue
                        try:
                            row = json.loads(text)
                        except json.JSONDecodeError:
                            issue(line, {}, "invalid_json", "structural")
                            continue
                        if not isinstance(row, dict):
                            issue(line, {}, "not_object", "structural")
                            continue
                        yield line, row
            iterator = json_rows()
        local_ids = set()
        for line, row in iterator:
            stats["rows"] += 1
            desc = row.get("description")
            if not isinstance(desc, str) or not desc.strip():
                issue(line, row, "empty_description", "structural")
                continue
            if len(desc) > 12000:
                issue(line, row, "over_api_length", "structural")
            item_id = row.get("item_id")
            if item_id in local_ids:
                issue(line, row, "duplicate_item_id", "structural")
            local_ids.add(item_id)
            is_public_train = path.name == "public-train-annotated.jsonl"
            if is_public_train:
                source_ids.add(item_id)
            if "source_item_id" in row:
                derived.append((rel, line, row))
            source = row.get("mcId", 101 if is_public_train else None)
            if source not in IDS:
                issue(line, row, "missing_or_invalid_source", "structural")
            target = row.get("found_microcategories")
            cats = None
            if target is not None:
                if not isinstance(target, list) or any(not isinstance(x, dict) or x.get("mcId") not in IDS for x in target):
                    issue(line, row, "invalid_labels", "structural")
                    continue
                cats = [x["mcId"] for x in target]
                if any(x.get("label") != NAMES[x["mcId"]] for x in target):
                    issue(line, row, "label_name_mismatch", "structural")
                if len(cats) != len(set(cats)):
                    issue(line, row, "repeated_label", "structural")
                if source in cats:
                    issue(line, row, "source_in_targets", "structural")
                stats["label_counts"].update(cats)
                if "offered_microcategories" in row and set(cats) != set(row["offered_microcategories"]) - {source}:
                    issue(line, row, "offered_target_mismatch", "structural")
                if "offered" in row and set(cats) != (set(row["offered"]) | set(row.get("partial", []))) - {source}:
                    issue(line, row, "legacy_offered_target_mismatch", "structural")
            seen[normalize(desc)].append((rel, line, item_id, source, tuple(sorted(cats)) if cats is not None else None))
            if row.get("confidence") == "medium":
                issue(line, row, "legacy_medium_confidence")
            if row.get("type") == "SRC":
                issue(line, row, "category_101_inferred_from_old_placement")
            if row.get("type") == "ILL":
                if re.search(r"ровняю стены под обои|стены ровные сделаю по уровню", desc, re.I):
                    issue(line, row, "ambiguous_leveling_method")
                if re.search(r"\bн[а-я]?е?\s+делаю|\bн\s+делаю", desc, re.I):
                    # A heuristic only; unrestricted character corruption can damage НЕ.
                    if re.search(r"\bн\s+делаю", desc, re.I):
                        issue(line, row, "possibly_damaged_negation")
            if row.get("type") == "NEG2" and len(row.get("offered", [])) > 1:
                if re.search(r"— не (?:штукатур|плиточник|сантехник|электрик|маляр):", desc):
                    issue(line, row, "profession_template_may_drop_offered_services")
            if row.get("type") == "NEG2" and re.search(r"откос подправить могу", desc):
                issue(line, row, "ambiguous_slope_repair")
        stats["label_counts"] = dict(sorted(stats["label_counts"].items()))
        stats["flags"] = dict(sorted(stats["flags"].items()))
    report["duplicates"] = []
    for entries in seen.values():
        if len(entries) < 2:
            continue
        # Different source categories legitimately change target labels.
        by_source = defaultdict(set)
        for _, _, _, source, labels in entries:
            if labels is not None:
                by_source[source].add(labels)
        report["duplicates"].append({"occurrences": entries, "conflicting_targets_same_source": any(len(v) > 1 for v in by_source.values())})
    report["missing_source_references"] = [
        {"file": rel, "line": line, "item_id": row["item_id"], "source_item_id": row["source_item_id"]}
        for rel, line, row in derived if row["source_item_id"] not in source_ids
    ]
    report["summary"] = {
        "files": len(paths), "rows": sum(s["rows"] for s in report["files"].values()),
        "unread_files": sum("not_read" in s for s in report["files"].values()),
        "structural_findings": sum(f["severity"] == "structural" for f in report["findings"]),
        "review_findings": sum(f["severity"] == "review" for f in report["findings"]),
        "duplicate_text_groups": len(report["duplicates"]),
        "conflicting_text_groups": sum(g["conflicting_targets_same_source"] for g in report["duplicates"]),
        "missing_source_references": len(report["missing_source_references"]),
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("audit_report.json"))
    parser.add_argument("--exclude-v2", action="store_true")
    args = parser.parse_args()
    report = audit(ROOT, args.exclude_v2)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
