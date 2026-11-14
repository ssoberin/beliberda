"""Adapt ready-made v2 splits to the existing trainer without resplitting."""
import argparse
import hashlib
import json
from pathlib import Path

from .prepare_dataset import normalize

IDS = set(range(102, 107))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('datasets/hard_dataset/v2'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a fresh output directory')
    mapping = {'train': 'train', 'validation': 'calibration', 'test': 'validation', 'challenge': 'challenge'}
    converted, text_sets, groups, hashes = {}, {}, {}, {}
    for source, target in mapping.items():
        path = args.input / f'{source}.jsonl'
        hashes[source] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        out = []
        for row in rows:
            offered = set(row['offered_microcategories'])
            found = {item['mcId'] for item in row['found_microcategories']}
            if found != offered - {row['mcId']}:
                raise ValueError(f'Inconsistent labels: {row["item_id"]}')
            out.append({'description': row['description'], 'labels': sorted(offered & IDS),
                        'mcId': row['mcId'], 'evaluation_labels': sorted(found & IDS), 'item_id': row['item_id']})
        converted[target] = out
        text_sets[source] = {normalize(row['description'][:6000]) for row in rows}
        if len(text_sets[source]) != len(rows):
            raise ValueError(f'Duplicate effective texts in {source}')
        groups[source] = {row['group_id'] for row in rows}
    names = list(mapping)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            if text_sets[a] & text_sets[b] or groups[a] & groups[b]:
                raise ValueError(f'Text/group leakage between {a} and {b}')
    args.output.mkdir(parents=True)
    for name, rows in converted.items():
        (args.output / f'{name}.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    source_stats = json.loads((args.input / 'stats.json').read_text())
    result = {'seed': source_stats['seed'], 'sgd_random_state': 42, 'source_sha256': hashes,
              'source_splits': {source: len(converted[target]) for source, target in mapping.items()},
              'splits': {name: len(rows) for name, rows in converted.items()},
              'mapping': mapping, 'training_labels': 'offered_microcategories intersect 102..106',
              'evaluation_labels': 'found_microcategories intersect 102..106; source category excluded',
              'text_and_group_intersections': 0, 'resplit': False}
    (args.output / 'stats.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
