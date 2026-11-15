"""Read-only text-based realism screening; flags are expert review candidates."""
import collections,hashlib,json,re
from pathlib import Path
ROOT=Path('output/v3-realism-screen');BASE=Path('/tmp/innopolis-dataset-v3-review/datasets/own_v3')
PATTERNS={
'price_explanation':r'отказ относится к бесплатной работе, а не к самой услуге',
'repeated_team':r'лично руководитель не выполняет|бригадир лично не выезжает|хотя владелец сам не выполняет',
'repeated_faq':r'спрашивают:|вопрос из чата:|в переписке уточняют',
'repeated_offer_correction':r'в старом объявлении писали|раньше не брались за|отказ выполнять|сейчас в перечень добавили|сообщение о том|прежде не брались выполнять',
'repeated_price':r'бесплатно .{1,120}? не делаем; по согласованной смете|не обещаем выполнить .{1,120}? бесплатно, но по смете|за оплату выполняем',
'explicit_stale_tags':r'в старых тегах осталось|поисковый запрос .{1,160}? устарел|попали в поисковые теги случайно',
}
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def norm(s):return ' '.join(s.casefold().split())
def assess(row):
 text=norm(row['description']);counts={k:len(re.findall(p,text)) for k,p in PATTERNS.items()}
 structural=[];offered=set(row['offered_microcategories']);source=row['mcId'];found={c['mcId'] for c in row['found_microcategories']}
 if found!=offered-{source}:structural.append('found_offered_source_mismatch')
 for e in row.get('evidence',[]):
  if row['description'][e['start']:e['end']]!=e['quote']:structural.append('bad_evidence_span');break
 positive={e['mcId'] for e in row.get('evidence',[]) if e.get('status')=='offered'}
 negative={e['mcId'] for e in row.get('evidence',[]) if e.get('status')=='not_offered'}
 if positive&negative:structural.append('same_category_in_positive_and_negative_status_requires_review')
 reasons=[]
 if counts['price_explanation']:reasons.append('meta_explanation_of_negation')
 for k in ['repeated_team','repeated_faq','repeated_offer_correction','repeated_price']:
  if counts[k]>=3:reasons.append(k+'_ge3')
 tier='quarantine_structural' if structural else 'rewrite_candidate' if reasons else 'review_stale_tag' if counts['explicit_stale_tags'] else 'no_high_priority_flag'
 return {'item_id':row['item_id'],'group_id':row.get('group_id'),'source':source,'offered':sorted(offered),'found':sorted(found),'tier':tier,'reasons':structural+reasons+(['explicit_stale_tags'] if counts['explicit_stale_tags'] else []),'text_counts':counts,'description':row['description'],'characters':len(row['description'])}
def main():
 summary={};catalog={};hashes={}
 paths={s:BASE/f'{s}.jsonl' for s in ['train','validation','test','challenge']}
 paths['revision3_new250']=Path('datasets/own_v3_revision3/replacements_250.jsonl')
 paths['revision3_train']=Path('datasets/own_v3_revision3/train.jsonl')
 for split,path in paths.items():
  hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest();rows=read(path);items=[assess(r) for r in rows]
  labelgroups=collections.defaultdict(set)
  for r in rows:labelgroups[(norm(r['description']),r['mcId'])].add(tuple(c['mcId'] for c in r['found_microcategories']))
  conflicting={k for k,v in labelgroups.items() if len(v)>1}
  summary[split]={'rows':len(rows),'unique_descriptions':len({norm(r['description']) for r in rows}),'tiers':dict(collections.Counter(r['tier'] for r in items)),'reasons':dict(collections.Counter(x for r in items for x in r['reasons'])),'conflicting_same_text_same_source_groups':len(conflicting),'by_generator_construction':dict(collections.Counter(r.get('construction','not_available') for r in rows))}
  catalog[split]=items
  print(split,json.dumps(summary[split],ensure_ascii=False),flush=True)
 (ROOT/'summary.json').write_text(json.dumps({'method':'Text-based expert heuristics, no predictions used; structural checks do not establish full semantic consistency. >=3 repeated constructions and explicit logical self-explanations trigger rewrite review, not automatic deletion. Stale-tag cases separate.','sha256':hashes,'sets':summary},ensure_ascii=False,indent=2)+'\n')
 (ROOT/'candidates.json').write_text(json.dumps({k:[r for r in v if r['tier']!='no_high_priority_flag'] for k,v in catalog.items()},ensure_ascii=False,indent=2)+'\n')
 # Print representative flagged texts, no model scores or label-dependent selection.
 for split in ['train','test','revision3_new250']:
  seen=set()
  for r in catalog[split]:
   for reason in r['reasons']:
    if reason not in seen:
     print('\nEXAMPLE',split,r['item_id'],r['offered'],r['reasons'],'\n',r['description'][:2100],flush=True);seen.add(reason)
if __name__=='__main__':main()
