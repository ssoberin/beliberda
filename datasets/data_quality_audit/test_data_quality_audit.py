import json
import unittest
from pathlib import Path

from data_quality_audit import OUT, theoretical_ceiling
from score_human_labels import calculate


class DataQualityTests(unittest.TestCase):
    def test_ceiling_for_conflicting_duplicate(self):
        group=[{'gold_mcId':[102]},{'gold_mcId':[]}]
        result=theoretical_ceiling([group])
        self.assertAlmostEqual(result['micro_f1'],2/3,places=7)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))

    def test_blind_export_excludes_gold_predictions(self):
        with (OUT/'annotation/public_blind.jsonl').open(encoding='utf-8') as stream:
            blind=[json.loads(s) for s in stream]
        self.assertEqual(len(blind),97)
        self.assertEqual(sum(len(x['selected_indices']) for x in blind),150)
        for r in blind:
            self.assertEqual(set(r),{'item_id','description','source_mcId','human_labels',
                                     'human_comment','original_indices','selected_indices'})
            self.assertIsNone(r['human_labels'])
            self.assertEqual(r['source_mcId'],101)

    def test_scorer_rejects_unfilled_labels(self):
        with self.assertRaises(ValueError):
            calculate([{'item_id':'a','human_labels':None,'source_mcId':101,
                        'original_indices':[0]}],
                      [{'item_id':'a','original_indices':[0],
                        'selected_indices':[0],
                        'original_rows':[{'index':0,'gold_mcId':[],
                                          'predicted_mcId':[]}]}])

if __name__=='__main__':unittest.main()
