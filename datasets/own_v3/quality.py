"""Read-only quality checks, with report output limited to the new corpus."""
from collections import Counter, defaultdict
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
import re

from generate import ROOT, validate

def shingles(text):
    words = re.findall(r'[а-яёa-z0-9]+', text.casefold())
    return {tuple(words[i:i + 5]) for i in range(max(1, len(words) - 4))}

def near_duplicates(splits, threshold=.8):
    """Exact prefix-filtered token-5-gram Jaccard join across splits.

    Shared lexical prefixes yield candidates; verify every candidate against
    the complete sets. No embedding model and no claim of semantic deduplication.
    """
    entries = [(split, row['item_id'], shingles(row['description']))
               for split, rows in splits.items() for row in rows]
    frequency = Counter(s for _, _, values in entries for s in values)
    index = defaultdict(list)
    matches = []
    for i, (split, item_id, values) in enumerate(entries):
        ordered = sorted(values, key=lambda s: (frequency[s], s))
        prefix = ordered[:len(values) - math.ceil(threshold * len(values)) + 1]
        candidates = {j for s in prefix for j in index[s]}
        for j in sorted(candidates):
            other_split, other_id, other = entries[j]
            if split == other_split: continue
            if min(len(values), len(other)) < threshold * max(len(values), len(other)): continue
            score = len(values & other) / len(values | other)
            if score >= threshold:
                matches.append({'a': item_id, 'b': other_id, 'jaccard': round(score, 4)})
        for s in prefix: index[s].append(i)
    return matches

def main():
    splits = {s: [json.loads(line) for line in (ROOT / f'{s}.jsonl').read_text().splitlines()]
              for s in ('train', 'validation', 'test')}
    validate(splits)
    matches = near_duplicates(splits)
    report = {'scope': 'Only own_v3 train/validation/test; no public data read.',
              'input_sha256': {s: hashlib.sha256((ROOT / f'{s}.jsonl').read_bytes()).hexdigest() for s in splits},
              'token_ngram': 5, 'jaccard_threshold': .8,
              'cross_split_near_duplicate_pairs': len(matches), 'examples': matches[:30],
              'semantic_independence_proven': False}
    (ROOT / 'quality.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if matches: raise SystemExit('Cross-split near-duplicates need review')

if __name__ == '__main__':
    main()
