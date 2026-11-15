"""Invariants for the deterministic grouped split, independent of fitted models."""
import importlib.util
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('postoyanka_experiment', HERE / 'experiment.py')
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old = {name: experiment.read(experiment.PUBLIC / f'{file}.jsonl') for name, file in
                   [('train', 'train'), ('validation', 'calibration'), ('test', 'validation')]}
        cls.new = experiment.read(experiment.NEW)
        cls.splits = experiment.split_new(cls.new, cls.old)

    def test_all_rows_accounted_for_and_stable(self):
        self.assertEqual(sum(map(len, self.splits.values())), len(self.new))
        second = experiment.split_new(self.new, self.old)
        for name in self.splits:
            self.assertEqual([r['item_id'] for r in self.splits[name]],
                             [r['item_id'] for r in second[name]])

    def test_groups_do_not_cross_partitions(self):
        keys = {}
        for name, rows in self.splits.items():
            for row in rows:
                meta = row.get('meta', {})
                group_keys = [('text', experiment.norm(row['description']))]
                if meta.get('core'):
                    group_keys.append(('core', experiment.norm(meta['core'])))
                if 'source_item_id' in meta:
                    group_keys.append(('source', meta['source_item_id']))
                for key in group_keys:
                    self.assertEqual(keys.setdefault(key, name), name)

    def test_old_holdout_not_used_in_new_train(self):
        old_texts = {experiment.norm(r['description']) for r in
                     self.old['validation'] + self.old['test']}
        self.assertTrue(all(experiment.norm(r['description']) not in old_texts
                            for r in self.splits['train']))


if __name__ == '__main__':
    unittest.main()
