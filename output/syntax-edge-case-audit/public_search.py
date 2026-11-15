"""Reproduce candidate search in original public data; no training or gold edits.

Candidate matching is deliberately broad. It is NOT an automatic semantic labeler.
Run from the repository root with the experiment Python environment.
"""
import hashlib
import json
import re
from pathlib import Path

import duckdb

OUT = Path('output/syntax-edge-case-audit')
PATTERN = re.compile(
    r'\bне\b|\bтолько\b|кроме|исключ|отказ|бер[её]мся|занимаемся', re.I
)


def main():
    train_path = Path('datasets/public/public-train-annotated.jsonl')
    test_path = Path('datasets/public/public-test-dataset.parquet')
    train = [json.loads(s) for s in train_path.read_text().splitlines() if s.strip()]
    con = duckdb.connect()
    cursor = con.execute('select * from read_parquet(?)', [str(test_path)])
    columns = [x[0] for x in cursor.description]
    test = [dict(zip(columns, values)) for values in cursor.fetchall()]
    summary = {}
    for name, rows in [('train', train), ('test', test)]:
        seen = set()
        candidates = []
        for i, row in enumerate(rows):
            text = row['description']
            if text in seen:
                continue
            seen.add(text)
            matches = list(PATTERN.finditer(text))
            if matches:
                candidates.append({
                    'row_idx': i, 'description': text,
                    'snippets': [text[max(0, m.start()-100):m.end()+150] for m in matches],
                    'gold': row.get('categories', row.get('found_microcategories')),
                })
        (OUT / f'public_{name}_search.json').write_text(
            json.dumps(candidates, ensure_ascii=False, indent=2) + '\n')
        summary[name] = dict(rows=len(rows), unique_descriptions=len(seen),
                             broad_search_candidates=len(candidates))
    summary['sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [train_path, test_path]}
    summary['limitations'] = ('Lexical retrieval plus expert review; absence of an exact '
                              'analogue does not prove absence of all semantic paraphrases.')
    (OUT / 'public_search_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
