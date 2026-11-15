"""Read-only natural tail-variant audit using frozen C_syntax; no text perturbation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
import collections, gzip, hashlib, json, pickle, re, sys
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool
sys.path.insert(0,str(Path.cwd()))
import runpy
helpers=runpy.run_path('output/public-test-audit/audit.py',run_name='audit_helpers')
sets=helpers['load_sets'](); norm=helpers['norm']
P=Path('output/public-tag-tail-audit');P.mkdir(exist_ok=True,parents=True)
model_path=Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')
before=hashlib.sha256(model_path.read_bytes()).hexdigest()
saved=pd.read_csv('output/catboost-optimization/final_predictions.csv')
saved=saved[saved.model=='C_syntax_baseline__fixed']
pred={}
for split,key in [('pub_validation','public_validation_729'),('official_208','public_test_208')]:
 for idx,g in saved[saved.dataset==split].groupby('row_idx'):
  r=sets[key][int(idx)];pred[(key,int(idx))]={'prediction':sorted(g.loc[g.prediction==1,'cat_id'].astype(int).tolist()),'probabilities':{str(int(x.cat_id)):float(x.probability) for x in g.itertuples()}}
# Calibration cache contains frozen full-head inference probabilities, not training OOF.
with open('output/catboost-optimization/cache/selection.pkl','rb') as f: cache=pickle.load(f)
cal=cache['cal'].copy();cal=cal[cal.row_idx<728]
for col in cache['categorical']:cal[col]=cal[col].astype(str)
model=CatBoostClassifier();model.load_model(str(model_path))
cal['probability']=model.predict_proba(Pool(cal[cache['cols']],cat_features=cache['categorical']),thread_count=2)[:,1]
for idx,g in cal.groupby('row_idx'):
 pred[('public_calibration_728',int(idx))]={'prediction':sorted(g.loc[g.probability>=.5,'cat_id'].astype(int).tolist()),'probabilities':{str(int(x.cat_id)):float(x.probability) for x in g.itertuples()}}

seo=re.compile(r'(?im)^\s*[^\w\n]{0,5}(?:(?:для|ключи для) поиска|ключевые (?:слова|запросы)(?: для поиска)?|поисковые запросы|также нас ищут(?: как)?)\s*[:：]')
meta=re.compile(r'(?im)^\s*(?:характеристик[аи]|параметры(?: товара)?|место на складе|артикул)\s*[:：]')
def cut(text):
 hits=[(m.start(),'explicit_seo') for m in seo.finditer(text)]+[(m.start(),'attributes') for m in meta.finditer(text)]
 hits=[(i,t) for i,t in hits if i>=80]
 if hits:
  i,t=min(hits);return text[:i].strip(),text[i:].strip(),t
 # Unmarked keyword paragraphs are candidates, not automatically meaningless noise.
 for m in list(re.finditer(r'\n\s*\n',text))[-1:]:
  tail=text[m.end():].strip()
  if m.start()>=80 and len(tail)>=100 and tail.count(',')+tail.count(';')>=10 and len(tail.split())>=25 and tail.count('\n')<=2:
   return text[:m.start()].strip(),tail,'unmarked_list_candidate'
 return text,'','none'

all_rows=[];stats={}
for key,rows in sets.items():
 counts=collections.Counter();unique=collections.defaultdict(set)
 for r in rows:
  body,tail,kind=cut(r['description']);counts[kind]+=1;unique[kind].add(norm(r['description']))
  all_rows.append({**r,'split':key,'body':body,'tail':tail,'tail_type':kind,**pred.get((key,r['row']),{})})
 stats[key]={'rows':len(rows),'detected_rows':dict(counts),'unique_descriptions_by_type':{k:len(v) for k,v in unique.items()}}
groups=collections.defaultdict(list)
for r in all_rows:groups[norm(r['body'])].append(r)
variants=[]
for rs in groups.values():
 if not any(r['tail'] for r in rs):continue
 full=collections.defaultdict(list)
 for r in rs:full[norm(r['description'])].append(r)
 if len(full)<2:continue
 vs=[]
 for value in full.values():
  exemplar=value[0]
  golds=sorted({tuple(r['gold']) for r in value})
  available=[r for r in value if 'prediction' in r]
  ps=sorted({tuple(r['prediction']) for r in available})
  vs.append({'description':exemplar['description'],'body':exemplar['body'],'tail':exemplar['tail'],'tail_type':exemplar['tail_type'],'golds':[list(x) for x in golds],'references':[{'split':r['split'],'row':r['row']} for r in value],'scored_references':[{'split':r['split'],'row':r['row'],'prediction':r['prediction'],'probabilities':r['probabilities']} for r in available],'predictions':[list(x) for x in ps]})
 gold_union={tuple(g) for v in vs for g in v['golds']}
 scored=[v for v in vs if v['predictions']]
 prediction_union={tuple(g) for v in scored for g in v['predictions']}
 variants.append({'n_variants':len(vs),'gold_changes':len(gold_union)>1,'within_variant_gold_conflict':any(len(v['golds'])>1 for v in vs),'scored_variants':len(scored),'prediction_changes':len(scored)>1 and len(prediction_union)>1,'variants':vs})
variants.sort(key=lambda g:(g['prediction_changes'],g['gold_changes'],g['scored_variants']),reverse=True)
summary={'method':'Natural exact normalized-body matches; unchanged full inputs; frozen C_syntax threshold 0.5; no training or counterfactual predictions','model_sha256':before,'stats':stats,'body_groups_with_distinct_tails':len(variants),'groups_with_gold_changes':sum(g['gold_changes'] for g in variants),'groups_with_at_least_two_scored_variants':sum(g['scored_variants']>=2 for g in variants),'groups_with_prediction_changes':sum(g['prediction_changes'] for g in variants)}
assert before==hashlib.sha256(model_path.read_bytes()).hexdigest()
(P/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
(P/'natural_variants.json').write_text(json.dumps(variants,ensure_ascii=False,indent=2)+'\n')
(P/'natural_variants.json.gz').write_bytes(gzip.compress((P/'natural_variants.json').read_bytes(),mtime=0))
print(json.dumps(summary,ensure_ascii=False,indent=2))
for i,g in enumerate(variants):
 if not g['prediction_changes']:continue
 print('GROUP',i,{k:v for k,v in g.items() if k!='variants'})
 for v in g['variants']:
  if v['predictions']:print(json.dumps({k:v[k] for k in ['tail_type','tail','golds','predictions','scored_references']},ensure_ascii=False))
