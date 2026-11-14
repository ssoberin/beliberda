"""Summarize measured coverage; lexical flags are proxies, not semantic annotations."""
import json,re,collections
from pathlib import Path
from avito_compare.scripts.evaluate_v2_experiment import metrics,IDS
import numpy as np
P=Path('output/public-test-coverage-audit');data=json.loads((P/'examples.json').read_text());summary=json.loads((P/'summary.json').read_text())
patterns={'затирка швов':(r'затир\w*\s+(?:межплиточ\w*\s+)?шв',103),'крыльцо':(r'крыльц',103),'замена плитки':(r'(?:замен\w*.{0,35}плит|плит\w*.{0,35}замен)',103),'колеровка':(r'колеров',106),'подбор цвета':(r'подбор\w*\s+(?:краск|цвет|оттен)',106),'явный только комплекс / отказ от отдельных задач':(r'только\s+(?:комплекс|под\s+ключ)|отдельн\w*\s+(?:мелк\w*\s+)?(?:работ\w*|задач\w*|заказ\w*)\s+не\s+бер',None)}
coverage={}
for name,(pat,category) in patterns.items():
 coverage[name]={}
 for split,rows in data.items():
  hit=[r for r in rows if re.search(pat,r['description'],re.I|re.S)]
  coverage[name][split]={'rows':len(hit),'with_target_label':sum(category in r['gold'] for r in hit) if category else None,'empty_gold':sum(not r['gold'] for r in hit),'fp':sum(r['fp'] for r in hit),'fn':sum(r['fn'] for r in hit)}
summary['lexical_topic_coverage']=coverage
summary['error_breakdown']={}
for split,rows in data.items():
 summary['error_breakdown'][split]={k:{'rows':sum(f(r) for r in rows),'fp':sum(r['fp'] for r in rows if f(r)),'fn':sum(r['fn'] for r in rows if f(r))} for k,f in {'empty_gold':lambda r:not r['gold'],'nonempty_gold':lambda r:bool(r['gold']),'conflicting_test_group':lambda r:r['conflicting_test_labels']}.items()}
summary['source_sha256']={}
import hashlib
for path in [Path('datasets/public/public-test-dataset.parquet'),*[Path('output/ngram-comparison/splits')/f'{s}.jsonl' for s in ['train','calibration','validation']]]:
 summary['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
# Store only public test exemplars and compact per-row validation diagnostics in Git-friendly outputs.
(P/'test_examples.json').write_text(json.dumps(data['test'],ensure_ascii=False,indent=2)+'\n')
(P/'validation_neighbors.json').write_text(json.dumps([{k:r[k] for k in ['index','gold','predicted','nearest_similarity','nearest_train_index','fp','fn','unknown_word_fraction','unknown_bigram_fraction']} for r in data['validation']],ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'coverage':coverage,'errors':summary['error_breakdown']},ensure_ascii=False,indent=2))
