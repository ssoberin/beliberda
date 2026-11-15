import unittest

import experiment as e


class FakeTokenizer:
    cls_token_id=101
    sep_token_id=102
    def encode(self,text,add_special_tokens=False,truncation=False):
        return list(range(int(text)))

class ExperimentTests(unittest.TestCase):
    def test_head_tail_preserves_bounds(self):
        ids=e.encode(FakeTokenizer(),'900')
        self.assertEqual(len(ids),768)
        self.assertEqual(ids[:4],[101,0,1,2])
        self.assertEqual(ids[-4:],[897,898,899,102])

    def test_labels_and_excluded_source_category(self):
        rows=[{'description':'a','labels':[102,103]},
              {'description':'b','mcId':103,'found_microcategories':[{'mcId':103},{'mcId':104}]},
              {'description':'c','categories':['Электрика']}]
        self.assertEqual(e.targets(rows).tolist(),[[1,1,0,0,0],[0,0,1,0,0],[0,0,0,1,0]])

if __name__=='__main__':unittest.main()
