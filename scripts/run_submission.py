#!/usr/bin/env python3
"""Convert a CSV batch through the *running* backend into bot-ready CSV.

No model is loaded here. The backend configuration must match the best submit.
The output is written only after every item and every draft succeeds.
"""
import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HEADER = ("testcase", "microcats", "descriptions")
MAX_BYTES = 5 * 1024 * 1024


def input_rows(raw):
    if not 0 < len(raw) <= MAX_BYTES:
        raise ValueError("Input must contain 1–5 MiB or less")
    text = raw.decode("utf-8-sig")
    head = text.splitlines()[0]
    delimiter = next((sep for sep in (",", ";") if "description" in next(
        csv.reader([head], delimiter=sep), [])), None)
    if delimiter is None:
        raise ValueError("CSV header must include description")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
    if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("Duplicate or missing CSV headers")
    if "description" not in reader.fieldnames:
        raise ValueError("CSV header must include description")
    rows = list(reader)
    if not 1 <= len(rows) <= 300:
        raise ValueError("Expected 1–300 advertisements")
    names = set()
    for i, row in enumerate(rows):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f"Row {i + 1}: malformed CSV field count")
        if not row["description"].strip():
            raise ValueError(f"Row {i + 1}: empty description")
        name = row.get("testcase") or f"case-{i + 1:03d}"
        if name in names or len(name) > 200:
            raise ValueError(f"Row {i + 1}: duplicate or long testcase")
        names.add(name)
    return rows


def convert(items, rows):
    if set(items) != set(range(len(rows))):
        raise ValueError("Missing, repeated or unexpected result index")
    table = io.StringIO(newline="")
    writer = csv.writer(table, lineterminator="\r\n")
    writer.writerow(HEADER)
    for index, row in enumerate(rows):
        item = items[index]
        if item.get("error") or item.get("errors") or item.get("incomplete"):
            raise ValueError(f"Row {index + 1}: backend reported an error")
        if row.get("testcase") and item.get("testcase") != row["testcase"]:
            raise ValueError(f"Row {index + 1}: testcase changed")
        ids = item.get("microcats")
        drafts = item.get("drafts")
        if (not isinstance(ids, list) or not isinstance(drafts, list) or
                any(type(mid) is not int or mid not in range(101, 107) for mid in ids) or
                len(ids) != len(set(ids)) or len(drafts) != len(ids)):
            raise ValueError(f"Row {index + 1}: incomplete or malformed drafts")
        mapping = {}
        for draft in drafts:
            if not isinstance(draft, dict) or draft.get("mcId") in mapping:
                raise ValueError(f"Row {index + 1}: duplicate draft ID")
            mapping[draft.get("mcId")] = draft.get("text")
        if set(mapping) != set(ids) or any(not isinstance(mapping[mid], str) or not mapping[mid].strip() for mid in ids):
            raise ValueError(f"Row {index + 1}: missing draft text")
        writer.writerow([row.get("testcase") or f"case-{index + 1:03d}",
                         json.dumps(ids, ensure_ascii=False),
                         json.dumps([mapping[mid] for mid in ids], ensure_ascii=False)])
    return table.getvalue()


def run(raw, rows, url, deadline):
    request = Request(url.rstrip("/") + "/api/create?generateText=true", data=raw,
                      headers={"Content-Type": "text/csv; charset=utf-8", "Accept": "application/x-ndjson"})
    items = {}
    done = None
    start = time.monotonic()
    try:
        with urlopen(request, timeout=120) as response:
            if "application/x-ndjson" not in response.headers.get("Content-Type", ""):
                raise ValueError("Backend did not return NDJSON")
            for line in response:
                if time.monotonic() - start > deadline:
                    raise TimeoutError(f"Exceeded {deadline}s deadline; no CSV written")
                event = json.loads(line)
                kind = event.get("type")
                if kind == "item_done":
                    index = event.get("index")
                    if type(index) is not int or index in items or not 0 <= index < len(rows):
                        raise ValueError("Duplicate or invalid item_done index")
                    items[index] = event
                    print(f"{len(items)}/{len(rows)} completed ({time.monotonic()-start:.1f}s)", file=sys.stderr)
                elif kind == "done":
                    done = event
                    break
    except HTTPError as exc:
        raise ValueError(f"Backend HTTP {exc.code}: {exc.read(800).decode('utf-8', errors='replace')}") from exc
    except URLError as exc:
        raise ValueError(f"Backend unavailable: {exc.reason}") from exc
    if not done or done.get("total") != len(rows) or done.get("completed") != len(rows) or done.get("failed") != 0:
        raise ValueError(f"Batch incomplete or failed: {done}; no CSV written")
    return convert(items, rows), round(time.monotonic() - start, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:18009")
    parser.add_argument("--deadline", type=int, default=900, help="Overall seconds, default 15 minutes")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; use a new filename to avoid stale submissions")
    try:
        raw = args.input.read_bytes()
        rows = input_rows(raw)
        output, elapsed = run(raw, rows, args.base_url, args.deadline)
        args.output.write_text(output, encoding="utf-8", newline="")
    except (OSError, UnicodeError, csv.Error, ValueError, TimeoutError, json.JSONDecodeError) as exc:
        parser.exit(1, f"No submission written: {exc}\n")
    print(f"Wrote {len(rows)} rows to {args.output} in {elapsed}s")


if __name__ == "__main__":
    main()
