import copy
import hashlib
import json
import random
import unittest

from challenge import CASES, build_challenge
from generate import ROOT, build, corrupt, FRAMES, IDS, labels, PARTICLES, validate
from lexicon import CONTEXT, detail_bank, service_bank
from workflows import workflow_bank, SHORT_SERVICES, SHORT_FRAMES
from quality import near_duplicates

class OwnDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved = {s: [json.loads(line) for line in (ROOT / f'{s}.jsonl').read_text().splitlines()]
                     for s in ('train', 'validation', 'test')}

    def test_saved_files_and_exact_quotas(self):
        validate(self.saved)
        for rows in self.saved.values():
            # All six sources occur at every cardinality, including zero/five.
            for k in range(6):
                self.assertEqual({r['mcId'] for r in rows if len(r['found_microcategories']) == k}, set(IDS))
            counts = [sum(c in [m['mcId'] for m in r['found_microcategories']] for r in rows) for c in IDS]
            self.assertLessEqual(max(counts) - min(counts), 3)
            self.assertEqual({r['mcId'] for r in rows if not r['found_microcategories']}, set(IDS))

    def test_reproducibility_including_stored_bytes(self):
        stats = json.loads((ROOT / 'stats.json').read_text())
        rebuilt = build(stats['seed'])
        for split, rows in rebuilt.items():
            data = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
            self.assertEqual(data, (ROOT / f'{split}.jsonl').read_bytes())
        for name, sha in stats['sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), sha)

    def test_held_out_surface_language(self):
        pools = {}
        for split in self.saved:
            pools[split] = set(CONTEXT[split])
            pools[split].update(s for phrases in service_bank(split).values() for s in phrases)
            pools[split].update(s for phrases in detail_bank(split).values() for s in phrases)
            pools[split].update(s for phrases in FRAMES[split].values() for s in phrases)
            pools[split].update(s for phrases in workflow_bank(split).values() for s in phrases)
            pools[split].update(SHORT_SERVICES[split].values())
            pools[split].update(SHORT_FRAMES[split].values())
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse(pools[a] & pools[b])

    def test_noise_preserves_negation_and_scope(self):
        text = 'Не только поклейку обоев предлагаем, но и штукатурку. Раньше розеток не делали, теперь делаем. Ни трубы, ни краны не меняем.'
        for mode in ('spelling', 'punctuation'):
            changed = corrupt(text, mode, random.Random(42))
            self.assertNotEqual(changed, text)
            self.assertEqual(PARTICLES.findall(changed), PARTICLES.findall(text))

    def test_validation_rejects_bad_labels_and_stale_evidence(self):
        damaged = copy.deepcopy(self.saved)
        damaged['train'][0]['found_microcategories'] = labels([damaged['train'][0]['mcId']])
        with self.assertRaises(AssertionError): validate(damaged)
        damaged = copy.deepcopy(self.saved)
        damaged['test'][0]['evidence'][0]['start'] += 1
        with self.assertRaises(AssertionError): validate(damaged)

    def test_validation_rejects_cross_split_duplicates(self):
        damaged = copy.deepcopy(self.saved)
        damaged['validation'][0]['description'] = damaged['train'][0]['description']
        with self.assertRaises(AssertionError): validate(damaged)

    def test_challenge_source_counterfactuals(self):
        rows = build_challenge()
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({r['description'] for r in rows}), 24)
        for i, (_, offered, _) in enumerate(CASES):
            group = rows[i * 6:(i + 1) * 6]
            self.assertEqual({r['mcId'] for r in group}, set(IDS))
            for row in group:
                self.assertEqual(row['found_microcategories'], labels(set(offered) - {row['mcId']}))
        self.assertFalse({r['description'] for r in rows} & {r['description'] for rs in self.saved.values() for r in rs})
        self.assertEqual(rows[0]['offered_microcategories'], [105])
        self.assertEqual(rows[6 * 16]['offered_microcategories'], [])

    def test_similarity_join_detects_near_copies(self):
        text = ' '.join(f'слово{i}' for i in range(60))
        changed = text.replace('слово28', 'другое')
        data = {'train': [{'item_id': 'a', 'description': text}],
                'test': [{'item_id': 'b', 'description': changed}]}
        self.assertEqual(len(near_duplicates(data)), 1)
        data['test'][0]['description'] = 'Совсем другая короткая фраза'
        self.assertFalse(near_duplicates(data))

    def test_service_refusals_are_not_hidden_in_technical_paragraphs(self):
        import re
        bad = re.compile(r'не (?:выполня\w*|предлага\w*|включа\w*|вход\w*|заявля\w*)', re.I)
        for rows in self.saved.values():
            for row in rows:
                self.assertNotIn('Прежде не занимались такой работой, как', row['description'])
                for e in row['evidence']:
                    if e['kind'] == 'technical_detail':
                        self.assertIsNone(bad.search(e['quote']), row['item_id'])
        damaged = copy.deepcopy(self.saved)
        row = next(r for r in damaged['train'] if any(e['kind'] == 'technical_detail' for e in r['evidence']))
        e = next(e for e in row['evidence'] if e['kind'] == 'technical_detail')
        text = e['quote'].replace('.', '. Механизированное оштукатуривание здесь не заявляется.', 1)
        row['description'] += '\n' + text
        e.update(quote=text, start=len(row['description']) - len(text), end=len(row['description']))
        with self.assertRaises(AssertionError): validate(damaged)

    def test_short_cases_keep_scope_difficulty_at_high_cardinality(self):
        for rows in self.saved.values():
            short = [r for r in rows if r['length_bin'] == 'short']
            self.assertGreater(sum(r['construction'] != 'direct' for r in short), len(short) * .3)
            for k in (3, 4, 5):
                subset = [r for r in short if len(r['found_microcategories']) == k]
                self.assertTrue(subset)
                self.assertTrue(any(any(e['status'] == 'not_offered' for e in r['evidence']) for r in subset))
                self.assertTrue(any(r['construction'] != 'direct' for r in subset))

    def test_long_content_is_not_mostly_generic_order_terms(self):
        for rows in self.saved.values():
            for r in rows:
                if r['length_bin'] in ('long', 'extended'):
                    self.assertLessEqual(sum(e['end'] - e['start'] for e in r['context_spans']) / len(r['description']), .35)

if __name__ == '__main__':
    unittest.main()
