import csv
import io
import json
import unittest

from run_submission import convert, input_rows


class SubmissionTest(unittest.TestCase):
    def test_quoted_multiline_and_order(self):
        raw = ('\ufefftestcase,description\ncase-001,"Штукатурим,\nставим раковины"\ncase-002,Другое\n').encode()
        rows = input_rows(raw)
        result = convert({0: {'index': 0, 'testcase': 'case-001', 'microcats': [101, 104],
                              'drafts': [{'mcId': 104, 'text': 'Сантехника'}, {'mcId': 101, 'text': 'Ремонт'}]},
                          1: {'index': 1, 'testcase': 'case-002', 'microcats': [], 'drafts': []}}, rows)
        parsed = list(csv.DictReader(io.StringIO(result)))
        self.assertEqual(list(parsed[0]), ['testcase', 'microcats', 'descriptions'])
        self.assertEqual(json.loads(parsed[0]['microcats']), [101, 104])
        self.assertEqual(json.loads(parsed[0]['descriptions']), ['Ремонт', 'Сантехника'])
        self.assertEqual(json.loads(parsed[1]['descriptions']), [])

    def test_reject_missing_text_and_duplicate_id(self):
        rows = input_rows(b'testcase,description\ncase-001,text\n')
        for bad in ({'microcats': [104], 'drafts': []},
                    {'microcats': [104, 104], 'drafts': [{'mcId': 104, 'text': 'x'}] * 2},
                    {'microcats': [104], 'drafts': [{'mcId': 104, 'text': None}]},
                    {'microcats': None, 'drafts': []}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                convert({0: bad}, rows)

    def test_reject_duplicate_testcase(self):
        with self.assertRaises(ValueError):
            input_rows(b'testcase,description\na,x\na,y\n')


if __name__ == '__main__':
    unittest.main()
