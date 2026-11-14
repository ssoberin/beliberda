import copy
import json
import unittest
from build import ROOT, build, validate, normalize


class ContextPairTests(unittest.TestCase):
    def test_contract_and_different_answers_in_each_pair(self):
        validate(build())

    def test_saved_data_is_reproducible(self):
        expected=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in build())
        self.assertEqual(expected,(ROOT/'data.jsonl').read_text())

    def test_rejects_same_answer_pair(self):
        rows=copy.deepcopy(build());rows[1]['offered_microcategories']=rows[0]['offered_microcategories']
        rows[1]['found_microcategories']=rows[0]['found_microcategories']
        end=rows[1]['discriminating_span']['end']
        rows[1]['evidence']=[{'mcId':c,'start':0,'end':end,'quote':rows[1]['description'][:end]} for c in rows[1]['offered_microcategories']]
        with self.assertRaises(AssertionError):validate(rows)

    def test_rejects_stale_evidence(self):
        rows=copy.deepcopy(build());rows[0]['discriminating_span']['start']+=1
        with self.assertRaises(AssertionError):validate(rows)

    def test_rejects_duplicate_input(self):
        rows=copy.deepcopy(build());rows[1]['description']=' '+rows[0]['description'].upper()+'\n'
        with self.assertRaises(AssertionError):validate(rows)

    def test_new_control_is_reproducible_and_group_disjoint(self):
        from test_cases import build_test
        train=build();controls=build_test()
        self.assertEqual(len(controls),20)
        self.assertEqual(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in controls),(ROOT/'test_20.jsonl').read_text())
        self.assertFalse({normalize(r['description']) for r in train}&{normalize(r['description']) for r in controls})
        self.assertFalse({r['group_id'] for r in train}&{r['group_id'] for r in controls})
        for i in range(0,20,2):
            self.assertNotEqual(controls[i]['offered_microcategories'],controls[i+1]['offered_microcategories'])


if __name__=='__main__':unittest.main()
