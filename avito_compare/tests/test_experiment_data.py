"""Regression checks for source-aware labels and leakage-safe v2 adaptation."""
import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from avito_compare.scripts import prepare_v2_experiment


class V2AdaptationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root/'source'
        self.source.mkdir()
        self.output = self.root/'prepared'
        self.rows = {
            'train': self.row('train', 'Меняем трубы и розетки.', 104, [104,105]),
            'validation': self.row('validation', 'Штукатурим стены по маякам.', 101, [102]),
            'test': self.row('test', 'Красим стены и выполняем ремонт под ключ.', 106, [101,106]),
            'challenge': self.row('challenge', 'Электрику не делаем.', 101, []),
        }
        (self.source/'stats.json').write_text(json.dumps({'seed':2026}))

    @staticmethod
    def row(name, description, source, offered):
        return {'item_id':name, 'group_id':name, 'description':description,
                'mcId':source, 'offered_microcategories':offered,
                'found_microcategories':[{'mcId':mid} for mid in offered if mid != source]}

    def write(self):
        for name,row in self.rows.items():
            (self.source/f'{name}.jsonl').write_text(json.dumps(row,ensure_ascii=False)+'\n')

    def run_adapter(self):
        self.write()
        argv=['prepare_v2','--input',str(self.source),'--output',str(self.output)]
        with patch.object(sys,'argv',argv), contextlib.redirect_stdout(io.StringIO()):
            prepare_v2_experiment.main()

    def test_offered_labels_keep_source_for_training_and_exclude_it_for_scoring(self):
        self.write()
        before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        self.run_adapter()
        train=json.loads((self.output/'train.jsonl').read_text())
        self.assertEqual(train['labels'],[104,105])
        self.assertEqual(train['evaluation_labels'],[105])
        test=json.loads((self.output/'validation.jsonl').read_text())
        self.assertEqual(test['labels'],[106])
        self.assertEqual(test['evaluation_labels'],[])
        self.assertEqual(test['item_id'],'test')
        calibration=json.loads((self.output/'calibration.jsonl').read_text())
        self.assertEqual(calibration['item_id'],'validation')
        after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.source.iterdir()}
        self.assertEqual(before,after)

    def test_rejects_inconsistent_found_labels(self):
        self.rows['train']['found_microcategories']=[{'mcId':104}]
        with self.assertRaisesRegex(ValueError,'Inconsistent labels'):
            self.run_adapter()
        self.assertFalse(self.output.exists())

    def test_rejects_normalized_description_leakage(self):
        self.rows['validation']['description']='  МЕНЯЕМ  трубы и розетки.\n'
        with self.assertRaisesRegex(ValueError,'leakage'):
            self.run_adapter()

    def test_rejects_family_leakage_even_with_different_text(self):
        self.rows['validation']['group_id']='train'
        with self.assertRaisesRegex(ValueError,'leakage'):
            self.run_adapter()

    def test_refuses_to_overwrite_an_existing_split(self):
        self.output.mkdir()
        sentinel=self.output/'keep.txt'
        sentinel.write_text('existing experiment')
        with self.assertRaisesRegex(ValueError,'fresh output'):
            self.run_adapter()
        self.assertEqual(sentinel.read_text(),'existing experiment')


if __name__=='__main__':
    unittest.main()
