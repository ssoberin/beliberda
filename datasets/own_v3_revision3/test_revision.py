import json,unittest,collections
from pathlib import Path
from build import ROOT,SCENARIOS,SCOPE,build,encoded,norm,read
class RevisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=read(ROOT/'train.jsonl');cls.new=read(ROOT/'replacements_250.jsonl')
        cls.manifest=json.loads((ROOT/'replacement_manifest.json').read_text())
    def test_fixed_size_and_unique_effective_text(self):
        self.assertEqual(len(self.rows),3000);self.assertEqual(len(self.new),250)
        self.assertEqual(len({norm(r['description'][:6000]) for r in self.rows}),3000)
    def test_group_variants_and_contrast(self):
        groups=collections.defaultdict(list)
        for r in self.new:groups[r['group_id']].append(r)
        self.assertEqual(len(groups),50)
        for rows in groups.values():
            self.assertEqual(len(rows),5)
            self.assertEqual(sum(bool(set(r['offered_microcategories'])&SCOPE) for r in rows),3)
            self.assertEqual({r['variant'] for r in rows},set(range(1,6)))
    def test_source_contract_and_evidence(self):
        for row in self.new:
            self.assertEqual([c['mcId'] for c in row['found_microcategories']],[c for c in row['offered_microcategories'] if c!=row['mcId']])
            self.assertEqual({e['mcId'] for e in row['evidence']},set(row['offered_microcategories']))
            for e in row['evidence']:self.assertEqual(row['description'][e['start']:e['end']],e['quote'])
            self.assertNotIn(row['description'][-1:],['+'])
    def test_manifest_matches_actual_replacements(self):
        for m,r in zip(self.manifest,self.new):
            self.assertEqual(m['new_item_id'],r['item_id'])
            self.assertEqual(self.rows[m['train_index']],r)
        self.assertEqual(len({m['old_item_id'] for m in self.manifest}),250)
    def test_saved_variants_match_authored_content(self):
        for i,case in enumerate(SCENARIOS):
            for j,marked in enumerate(case['variants']):
                r=self.new[i*5+j]
                self.assertEqual(r['description'],case['context']+'\n\n'+marked[2:])
                self.assertEqual(set(r['offered_microcategories'])&SCOPE,{case['category']} if marked[0]=='+' else set())
if __name__=='__main__':unittest.main()
