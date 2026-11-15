import json
import tempfile
import unittest
from pathlib import Path

from storage import (answer_path,completeness,export_jsonl,load_answers,
                     save_answer,tasks,validate_labels)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.task_dir=self.root/'annotation'
        self.task_dir.mkdir()
        public=[{'item_id':'public-1','description':'Плитку не кладём, сантехнику делаем.',
                 'source_mcId':101,'human_labels':None,'human_comment':'',
                 'original_indices':[2,3],'selected_indices':[2,3]}]
        train=[{'item_id':'train-1','description':'Проверка работы мастера',
                'current_labels':[104,105],'review_status':'',
                'corrected_labels':None,'human_comment':'', 'original_index':4}]
        for name,rows in [('public_blind.jsonl',public),('train_review.jsonl',train)]:
            (self.task_dir/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        self.answer_dir=self.root/'human_labels'

    def tearDown(self):self.temp.cleanup()

    def test_blind_projection_never_exposes_old_labels(self):
        public=tasks('public',self.task_dir)
        train=tasks('train',self.task_dir)
        self.assertEqual(public[0]['source_mcId'],101)
        self.assertNotIn('current_labels',train[0])
        self.assertNotIn('review_status',train[0])
        self.assertEqual(train[0]['source_mcId'],101)

    def test_save_load_restore_independence_and_export(self):
        items=tasks('public',self.task_dir)
        self.assertEqual(completeness(items,{})['missing_ids'],['public-1'])
        save_answer('public','annotator_1',items[0],[104],'Не плитка',items,self.answer_dir)
        save_answer('public','annotator_2',items[0],[], '',items,self.answer_dir)
        restored=load_answers('public','annotator_1',items,self.answer_dir)
        other=load_answers('public','annotator_2',items,self.answer_dir)
        self.assertEqual(restored['public-1']['human_labels'],[104])
        self.assertEqual(restored['public-1']['human_comment'],'Не плитка')
        self.assertEqual(other['public-1']['human_labels'],[])
        self.assertEqual(completeness(items,restored)['saved'],1)
        exported=json.loads(export_jsonl('public','annotator_1',items,self.answer_dir).decode())
        self.assertEqual(exported['original_indices'],[2,3])
        self.assertNotIn('gold_mcId',exported)
        self.assertNotIn('predicted_mcId',exported)
        self.assertEqual(answer_path('public','annotator_1',self.answer_dir).stat().st_mode&0o777,0o600)

    def test_edit_overwrites_only_own_answer_and_empty_is_complete(self):
        items=tasks('public',self.task_dir)
        save_answer('public','annotator_1',items[0],[103],'',items,self.answer_dir)
        save_answer('public','annotator_1',items[0],[],'Проверено: нет',items,self.answer_dir)
        rows=answer_path('public','annotator_1',self.answer_dir).read_text().splitlines()
        self.assertEqual(len(rows),1)
        answer=load_answers('public','annotator_1',items,self.answer_dir)['public-1']
        self.assertEqual(answer['human_labels'],[])
        self.assertEqual(completeness(items,{'public-1':answer})['missing_ids'],[])

    def test_invalid_labels_rejected(self):
        for labels in (None,[101],[102,102],['102'],[True]):
            with self.subTest(labels=labels),self.assertRaises(ValueError):
                validate_labels(labels)

    def test_streamlit_page_renders_without_private_data(self):
        from streamlit.testing.v1 import AppTest
        at=AppTest.from_file(str(Path(__file__).with_name('app.py'))).run(timeout=25)
        self.assertEqual(len(at.exception),0)
        self.assertIn('Разметка услуг Авито',[x.value for x in at.title])
        self.assertEqual({'Предыдущее','Сохранить','Следующее'},
                         {x.label for x in at.button if x.label in
                          {'Предыдущее','Сохранить','Следующее'}})
        rendered=' '.join(x.value for x in at.markdown)
        self.assertNotIn('predicted_mcId',rendered)
        self.assertNotIn('current_labels',rendered)

if __name__=='__main__':unittest.main()
