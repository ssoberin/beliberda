"""Compose ads from authored briefs and commercial supporting paragraphs.

Supporting conditions recur across ads; this is audited explicitly rather than
claimed to be 200 independently hand-written full narratives.
"""
import collections,hashlib,json,random
from pathlib import Path
from catalog import CASES
from support import PARAGRAPHS
P=Path(__file__).resolve().parent
BINS=[(80,200,20),(201,500,50),(501,1200,60),(1201,2500,50),(2501,4500,20)]
def build():
 rng=random.Random(42);assign=[(lo,hi) for lo,hi,n in BINS for _ in range(n)];rng.shuffle(assign)
 rows=[];review=[]
 for i,((topic,cats,title,detail),(lo,hi)) in enumerate(zip(CASES,assign),1):
  text=title+'. '+detail
  # Only standard transaction conditions; never introduce an offered target trade.
  desired=lo+int((hi-lo)*(.3+.35*rng.random()))
  pool=rng.sample(PARAGRAPHS,len(PARAGRAPHS))
  for paragraph in pool:
   if len(text)>=desired:break
   if len(text)+2+len(paragraph)<=hi:text+='\n\n'+paragraph.strip()
  assert lo<=len(text)<=hi,(i,len(text),lo,hi)
  uid=f'v3-hn200-{i:03d}'
  row={'item_id':uid,'group_id':uid,'description':text,'mcId':101,'source_mcId':101,'offered_microcategories':[],'found_microcategories':[],'gold':[]}
  rows.append(row)
  review.append({'item_id':uid,'topic':topic,'potential_false_positive_categories':cats,'actual_offer':title,'decisive_scope':detail,'why_gold_empty':f'Фактическое предложение: {title}. Граница предложения: {detail} Упоминания категорий {cats} относятся к объекту, товару, документации, чужому результату или отменённой услуге; собственного предложения работ этих категорий нет.','annotation_status':'author_labeled_self_reviewed_not_independent_gold','length_bin':[lo,hi]})
 assert len(rows)==200 and len({r['description'] for r in rows})==200
 return rows,review
def main():
 rows,review=build()
 for name,data in [('train_addition.jsonl',rows),('annotation_review.json',review)]:
  (P/name).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in data) if name.endswith('jsonl') else json.dumps(data,ensure_ascii=False,indent=2)+'\n')
 print(collections.Counter(tuple(r['length_bin']) for r in review))
if __name__=='__main__':main()
