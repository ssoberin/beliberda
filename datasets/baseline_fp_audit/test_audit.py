import unittest

import numpy as np

from audit import conflicts, norm
from rules import apply


class AuditTests(unittest.TestCase):
    def test_conflicting_identical_texts(self):
        rows=[{'description':' Трубы\nи смесители ', 'labels':[104]},
              {'description':'трубы и смесители', 'labels':[]},
              {'description':'другой текст','labels':[]}]
        c,details,groups=conflicts(rows)
        self.assertEqual(len(details),1)
        self.assertEqual(set(c),{0,1})
        self.assertEqual(details[0]['changing_categories'],[104])

    def test_guard_only_104_105(self):
        rows=[{'description':'Выполняем ремонт под ключ.'},
              {'description':'Устанавливаем розетки и смесители.'}]
        pred=np.ones((2,5),dtype=bool)
        result=apply(rows,pred)
        self.assertEqual(result[0].tolist(),[True,True,False,False,True])
        self.assertEqual(result[1].tolist(),[True,True,True,True,True])

if __name__=='__main__':unittest.main()
