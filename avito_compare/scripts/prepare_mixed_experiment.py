"""Combine deduplicated public and v2 ready splits without moving held-out rows."""
import argparse
import hashlib
import json
from pathlib import Path

from .prepare_dataset import normalize


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', type=Path, default=Path('output/ngram-comparison/splits'))
    parser.add_argument('--v2', type=Path, default=Path('output/v2-charwb-comparison/splits'))
    parser.add_argument('--output', type=Path, default=Path('output/mixed-charwb-comparison/splits'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    combined, hashes, counts = {}, {}, {}
    for name in ('train', 'calibration', 'validation'):
        combined[name] = []
        counts[name] = {}
        for dataset, folder in (('public', args.public), ('v2', args.v2)):
            path = folder / f'{name}.jsonl'
            rows = read(path)
            hashes[f'{dataset}/{name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
            counts[name][dataset] = len(rows)
            for row in rows:
                copied = dict(row, dataset=dataset)
                if dataset == 'public':
                    copied['mcId'] = 101
                    copied['evaluation_labels'] = copied['labels']
                combined[name].append(copied)
    combined['challenge'] = [dict(row, dataset='v2') for row in read(args.v2 / 'challenge.jsonl')]
    keys = {}
    for name, rows in combined.items():
        keys[name] = {normalize(row['description'][:6000]) for row in rows}
        if len(keys[name]) != len(rows):
            raise ValueError(f'Duplicate effective description inside {name}')
        for row in rows:
            if set(row['labels']) - {row['mcId']} != set(row['evaluation_labels']):
                raise ValueError('Inconsistent source-aware labels')
    names = list(combined)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if keys[a] & keys[b]:
                raise ValueError(f'Text leakage: {a} and {b}')
    args.output.mkdir(parents=True)
    for name, rows in combined.items():
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    stats = {'seed': {'public_split': 42, 'v2_supplied_split': 2026}, 'sgd_random_state': 42,
             'splits': {name: len(rows) for name, rows in combined.items()}, 'composition': counts,
             'input_sha256': hashes, 'resplit': False, 'reweighting': False,
             'normalized_effective_description_intersections': 0,
             'public_deduplication': 'Existing casefold/whitespace deduplication: 10000 raw rows -> 4855 unique; original 70/15/15 split preserved.',
             'v2_grouping': 'Ready-made group/type-disjoint splits preserved; no v2 test or challenge rows used for training.',
             'note': 'Synthetic data is an addition to public real data. Public weak labels remain unchanged. Near-duplicate public templates can remain.'}
    (args.output / 'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
