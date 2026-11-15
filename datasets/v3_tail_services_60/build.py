"""Build the separately versioned train addition and untouched holdout."""
import hashlib,json
from pathlib import Path
from catalog import TRAIN,TEST
from long_context import TRAIN_CONTEXT,TEST_CONTEXT
P=Path(__file__).resolve().parent
NAMES={102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',105:'Электрика',106:'Поклейка обоев и малярные работы'}
def build():
 reviews=[]
 for split,items,n in [('train',TRAIN,60),('test',TEST,20)]:
  assert len(items)==n,(split,len(items))
  rows=[]
  for i,(focus,gold,text,reason) in enumerate(items,1):
   context=(TRAIN_CONTEXT if split=='train' else TEST_CONTEXT).get(i-1)
   if context:
    body,sep,tail=text.rpartition('\n\n');assert sep
    text=body+'\n\n'+context+'\n\n'+tail
   assert set(gold)<=set(NAMES) and 80<len(text)<4500
   ident=f'tail60-{split}-{i:03d}'
   labels=[{'mcId':c,'label':NAMES[c]} for c in sorted(gold)]
   rows.append({'item_id':ident,'group_id':ident,'description':text,'mcId':101,'source_mcId':101,'offered_microcategories':sorted(gold),'found_microcategories':labels,'gold':sorted(gold)})
   reviews.append({'item_id':ident,'split':split,'focus':focus,'gold':sorted(gold),'rationale':reason,'annotation_status':'authored_and_expert_self_reviewed_not_independent_official_gold'})
  filename='train_addition.jsonl' if split=='train' else 'test_holdout.jsonl'
  (P/filename).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 assert len({text for _,_,text,_ in TRAIN+TEST})==80
 (P/'annotation_review.json').write_text(json.dumps(reviews,ensure_ascii=False,indent=2)+'\n')
 print({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.glob('*.jsonl')})
if __name__=='__main__':build()
