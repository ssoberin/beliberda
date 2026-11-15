#!/usr/bin/env python3
"""Reproducible, data-only EDA for the combined public + V3 training corpus.

Run from the repository root with /tmp/innopolis-qwen-emb-venv/bin/python
output/combined-public-v3-eda/eda.py
"""
from __future__ import annotations
import json, re, hashlib, sys, os
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'cache'/'mplconfig'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/combined-public-v3-eda'; OUT.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
LABELS=[102,103,104,105,106]
NAMES={102:'Штукатурка',103:'Плитка',104:'Сантехника',105:'Электрика',106:'Обои и малярка'}
PATHS={
 'public_train':ROOT/'output/ngram-comparison/splits/train.jsonl',
 'v3_train':ROOT/'datasets/own_v3_clean/train.jsonl',
 'addition60':ROOT/'datasets/v3_tail_services_60/train_addition.jsonl',
 'public_calibration':ROOT/'output/ngram-comparison/splits/calibration.jsonl',
 'v3_validation':ROOT/'datasets/own_v3_clean/validation.jsonl',
 'public_validation':ROOT/'output/ngram-comparison/splits/validation.jsonl',
 'v3_test':ROOT/'datasets/own_v3_clean/test.jsonl',
 'official208':ROOT/'datasets/public/public-test-dataset.parquet',
 'addition_holdout20':ROOT/'datasets/v3_tail_services_60/test_holdout.jsonl',
 'public_raw10000':ROOT/'datasets/public/public-train-annotated.jsonl',
}
def read_jsonl(p):
 return [json.loads(s) for s in p.open(encoding='utf-8') if s.strip()]
def text_of(r): return str(r.get('description') or '').strip()
def _ids(x):
 if isinstance(x,dict): x=list(x)
 return sorted({int(v.get('mcId')) if isinstance(v,dict) else int(v) for v in (x or []) if str(v.get('mcId') if isinstance(v,dict) else v).isdigit() and int(v.get('mcId') if isinstance(v,dict) else v) in LABELS})
def labels_of(r):
 # Classification target: additional found/gold labels only. Never use offered_microcategories
 # as target because it may include the actual input/source category (101..106).
 if 'found_microcategories' in r: return _ids(r.get('found_microcategories'))
 if 'labels' in r: return _ids(r.get('labels'))
 return _ids(r.get('gold'))
def offered_labels_of(r):
 return _ids(r.get('offered_microcategories')) if 'offered_microcategories' in r else None
def normalize(t): return re.sub(r'[^\w]+',' ',t.casefold(),flags=re.UNICODE).strip()
def normalize_ws(t): return ' '.join(t.casefold().split())
def stats(xs):
 a=np.asarray(xs,dtype=float)
 return {'n':int(len(a)),'mean':round(float(a.mean()),2) if len(a) else 0,'median':round(float(np.median(a)),2) if len(a) else 0,'p90':round(float(np.quantile(a,.9)),2) if len(a) else 0,'p95':round(float(np.quantile(a,.95)),2) if len(a) else 0,'max':int(a.max()) if len(a) else 0}
def load_json_sets():
 return {k:read_jsonl(p) for k,p in PATHS.items() if p.suffix=='.jsonl'}
def record(r,domain):
 t=text_of(r); y=labels_of(r)
 src=r.get('source_mcId',r.get('mcId',101))
 return {'domain':domain,'title':str(r.get('title') or ''),'text':t,'labels':y,'offered_labels':offered_labels_of(r),'offered_includes_source':bool(r.get('offered_microcategories') is not None and src in r.get('offered_microcategories',[])),'source_mcId':int(src),'id':r.get('item_id',r.get('group_id'))}

def main():
 sets=load_json_sets()
 # Build the actual train/calibration and independent controls exactly as specified.
 pub=[record(r,'public3398') for r in sets['public_train']]
 v3=[record(r,'V3clean2921') for r in sets['v3_train']]
 add=[record(r,'addition60') for r in sets['addition60']]
 train=pub+v3+add; baseline=pub+v3
 cal=[record(r,'public calibration') for r in sets['public_calibration']]+[record(r,'V3 validation') for r in sets['v3_validation']]
 controls={k:[record(r,k) for r in sets[key]] for k,key in [('public validation','public_validation'),('V3 test','v3_test'),('new holdout20','addition_holdout20')]}
 # official parquet via DuckDB (pandas parquet engines are not assumed).
 import duckdb
 p=str(PATHS['official208']).replace("'","''")
 offrows=duckdb.connect().execute(f"select title, description, categories from read_parquet('{p}')").fetchall()
 name_to_id={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,'Электрика':105,'Поклейка обоев и малярные работы':106}
 controls['official208']=[{'domain':'official208','title':a or '', 'text':(b or '').strip(),'labels':sorted({name_to_id[x] for x in (c or []) if x in name_to_id}),'source_mcId':101,'categories':c or []} for a,b,c in offrows]
 domains={'public3398':pub,'V3clean2921':v3,'addition60':add}
 norm_groups=defaultdict(list)
 for r in train: norm_groups[normalize(r['text'])].append(r)
 exact_groups=[v for k,v in norm_groups.items() if k]
 exact_conflicts=[g for g in exact_groups if len({tuple(x['labels']) for x in g})>1]
 duplicate_excess=sum(len(g)-1 for g in exact_groups)
 ws_groups=defaultdict(list)
 for r in train: ws_groups[normalize_ws(r['text'])].append(r)
 ws_exact_groups=[v for k,v in ws_groups.items() if k]
 ws_conflicts=[g for g in ws_exact_groups if len({tuple(x['labels']) for x in g})>1]
 # Text similarity graph: sparse/batched cosine to avoid an all-pairs dense matrix.
 texts=[r['text'] for r in train]
 vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=30000)
 X=vec.fit_transform(texts)
 # Blockwise similarity; only retain >= .85 edges, no classifier/model fitting.
 edge_i=[];edge_j=[]; near=[]
 for st in range(0,len(train),160):
  sims=(X[st:st+160]@X.T).toarray()
  for bi,row in enumerate(sims):
   i=st+bi; row[i]=-1
   inds=np.flatnonzero(row>=.85)
   for j in inds:
    if j<=i: continue
    edge_i.append(i);edge_j.append(int(j))
    if len(near)<10000: near.append((float(row[j]),i,int(j)))
 graph=csr_matrix((np.ones(2*len(edge_i)),(edge_i+edge_j,edge_j+edge_i)),shape=(len(train),len(train)))
 ncomp,comp=connected_components(graph,directed=False)
 sizes=Counter(comp.tolist()); components=[g for g in sizes.values() if g>1]
 near.sort(reverse=True)
 # Cross-split near matches against train, with the same train-fitted word TF-IDF space.
 cross_near={}; cross_near_examples={}
 train_X=X
 for split_name,rows in [('calibration',cal),*controls.items()]:
  Z=vec.transform([r['text'] for r in rows]); maxima=np.zeros(len(rows),dtype=float); argmax=np.zeros(len(rows),dtype=int)
  for st in range(0,len(rows),128):
   sims=(Z[st:st+128]@train_X.T).toarray()
   if sims.size:
    ids=sims.argmax(axis=1); vals=sims[np.arange(len(ids)),ids]
    maxima[st:st+len(ids)]=vals;argmax[st:st+len(ids)]=ids
  cross_near[split_name]={'rows':len(rows),'rows_max_cosine_ge_0_85':int((maxima>=.85).sum()),'rows_max_cosine_ge_0_95':int((maxima>=.95).sum()),'max_cosine':round(float(maxima.max()),4) if len(maxima) else None,'median_nearest_cosine':round(float(np.median(maxima)),4) if len(maxima) else None}
  ix=np.argsort(maxima)[-min(5,len(maxima)):][::-1]
  cross_near_examples[split_name]=[{'cosine':round(float(maxima[j]),4),'split_text':rows[int(j)]['text'],'train_domain':train[int(argmax[j])]['domain'],'train_text':train[int(argmax[j])]['text']} for j in ix]
 # Per-domain raw and label metrics.
 domain_metrics={}
 for name,rows in domains.items():
  card=[len(r['labels']) for r in rows]; lfreq=Counter(l for r in rows for l in r['labels'])
  combos=Counter(tuple(r['labels']) for r in rows)
  src_cond={}
  for src in sorted({r['source_mcId'] for r in rows}):
   sr=[r for r in rows if r['source_mcId']==src]
   src_cond[str(src)]={'rows':len(sr),'label_cardinality':{str(k):sum(len(r['labels'])==k for r in sr) for k in range(6)},'label_frequency':{str(l):sum(l in r['labels'] for r in sr) for l in LABELS},'combination_frequency':{' + '.join(map(str,k)) if k else '[]':v for k,v in sorted(Counter(tuple(r['labels']) for r in sr).items())}}
  offered_rows=[r for r in rows if r['offered_labels'] is not None]
  domain_metrics[name]={'rows':len(rows),'empty_gold':sum(not r['labels'] for r in rows),'empty_gold_rate':round(sum(not r['labels'] for r in rows)/len(rows),4),'label_cardinality':{str(k):card.count(k) for k in range(6)},'label_frequency':{str(k):lfreq[k] for k in LABELS},'combination_frequency':{' + '.join(map(str,k)) if k else '[]':v for k,v in sorted(combos.items())},'source_mcId_frequency':dict(Counter(str(r['source_mcId']) for r in rows)),'source_conditional_target_distribution':src_cond,'offered_audit':{'rows_with_offered_field':len(offered_rows),'offered_includes_source_rows':sum(r['offered_includes_source'] for r in offered_rows),'offered_additional_cardinality':{str(k):sum(len(r['offered_labels'])==k for r in offered_rows) for k in range(6)},'offered_equals_found_rows':sum(set(r['offered_labels'])==set(r['labels']) for r in offered_rows)},'length_chars':stats([len(r['text']) for r in rows]),'length_words':stats([len(re.findall(r'\w+',r['text'],re.UNICODE)) for r in rows]),'fixed_chunks':stats([len(chunk_ad_fixed(r['text'])) for r in rows]),'unique_normalized_texts':len({normalize(r['text']) for r in rows})}
 def summarize(rows):
  return {'rows':len(rows),'empty_gold':sum(not r['labels'] for r in rows),'empty_gold_rate':round(sum(not r['labels'] for r in rows)/len(rows),4),'label_cardinality':{str(k):sum(len(r['labels'])==k for r in rows) for k in range(6)},'label_frequency':{str(l):sum(l in r['labels'] for r in rows) for l in LABELS},'length_chars':stats([len(r['text']) for r in rows]),'length_words':stats([len(re.findall(r'\w+',r['text'],re.UNICODE)) for r in rows]),'fixed_chunks':stats([len(chunk_ad_fixed(r['text'])) for r in rows])}
 # Duplicates within and across splits; exact normalized overlap.
 splitsets={'train':train,'calibration':cal}
 splitsets.update(controls)
 norms={k:Counter(normalize(r['text']) for r in rs) for k,rs in splitsets.items()}
 cross={}
 cross_ws={}
 keys=list(norms)
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   inter=set(norms[a])&set(norms[b]);
   if inter: cross[f'{a} x {b}']={'unique_texts':len(inter),'rows_a':sum(norms[a][x] for x in inter),'rows_b':sum(norms[b][x] for x in inter)}
 ws_norms={k:Counter(normalize_ws(r['text']) for r in rs) for k,rs in splitsets.items()}
 for i,a in enumerate(keys):
  for b in keys[i+1:]:
   inter=set(ws_norms[a])&set(ws_norms[b])
   if inter: cross_ws[f'{a} x {b}']={'unique_texts':len(inter),'rows_a':sum(ws_norms[a][x] for x in inter),'rows_b':sum(ws_norms[b][x] for x in inter)}
 # public raw to actual partitions: item-level identity/normalized text reconstruction.
 raw=sets['public_raw10000']; raw_norm=Counter(normalize(text_of(r)) for r in raw); raw_ws_norm=Counter(normalize_ws(text_of(r)) for r in raw)
 pub_union=pub+ [record(r,'public calibration') for r in sets['public_calibration']]+[record(r,'public validation') for r in sets['public_validation']]
 raw_to_pub={normalize(r['text']) for r in pub_union}; raw_ws_to_pub={normalize_ws(r['text']) for r in pub_union}
 # Transparent heuristic feature flags; these are lexical indicators, not truth labels.
 flags={
  'question_mark':re.compile(r'\?'), 'negation_word':re.compile(r'\b(?:не|нет|ни|без|кроме|отказ|не выполня|не делаем|не предоставля)\b',re.I),
  'third_party_word':re.compile(r'\b(?:ваш(?:а|и|его|ему|им)?|друг(?:ой|ая|ие)|чуж(?:ой|ая|ие)|сторонн(?:ий|яя|ие)|дизайнер|прораб|заказчик|собственник|бригада|подрядчик|мастер)\b',re.I),
  'project_material_word':re.compile(r'\b(?:проект|дизайн-проект|материал|плитк\w*|обо\w*|краск\w*|смесител\w*|кабел\w*|смет\w*)\b',re.I),
  'seo_marker':re.compile(r'\b(?:ключев(?:ые|ых) запрос|для поиска|теги|поисков(?:ые|ых) запрос|\#\w+)\b',re.I),
  'material_sale_or_catalog':re.compile(r'\b(?:прода[её]м|продам|магазин|остатк\w*|каталог|ассортимент|доставк\w*|товар\w*|купить)\b',re.I),
  'project_only_or_design':re.compile(r'\b(?:составл\w* план|проект\w*|черт[её]ж|дизайнер\w*|передам черт[её]ж|для согласования)\b',re.I),
  'explicit_other_specialist':re.compile(r'\b(?:ваш(?:им|ему|а|и)? (?:электрик|монтажник|мастер)|выполняет ваш|делает ваш|работы выполняет (?:он|она|друг)|не наша услуга|не предоставля\w*)\b',re.I),
  'service_tail_marker':re.compile(r'(?:^|\n\s*\n)(?:.{0,80})(?:услуг|работ|выполня|предлага|предостав|монтаж|установ|уклад|покле|штукатур|подключ|разводк)[^\n]{0,180}[.!?]?\s*$',re.I|re.S),
 }
 flag_metrics={}
 for name,rows in domains.items():
  flag_metrics[name]={k:sum(bool(rx.search(r['text'])) for r in rows) for k,rx in flags.items()}
 # Official conflicts across identical normalized descriptions.
 og=defaultdict(list)
 for i,r in enumerate(controls['official208']): og[normalize(r['text'])].append(i)
 # (Known conflict audit is based on description only; retain observed count here too.)
 report={
  'provenance':{'paths':{k:str(v.relative_to(ROOT)) for k,v in PATHS.items()},'counts':{k:len(v) for k,v in sets.items()},'train_B':len(train),'train_baseline_without_addition':len(baseline),'calibration':len(cal),'controls':{k:len(v) for k,v in controls.items()},'official208_read_via':'DuckDB'},
  'domains':domain_metrics,
  'baseline_without_addition':summarize(baseline),
  'calibration_and_control_gold':{k:summarize(v) for k,v in [('calibration',cal),*controls.items()]},
  'combined':{'rows':len(train),'label_frequency':{str(l):sum(l in r['labels'] for r in train) for l in LABELS},'label_cardinality':{str(k):sum(len(r['labels'])==k for r in train) for k in range(6)},'combination_frequency':{' + '.join(map(str,k)) if k else '[]':v for k,v in sorted(Counter(tuple(r['labels']) for r in train).items())},'exact_punctuation_insensitive_unique_nonempty':len(exact_groups),'exact_punctuation_insensitive_duplicate_excess':duplicate_excess,'exact_punctuation_insensitive_conflicting_groups':len(exact_conflicts),'exact_whitespace_only_unique_nonempty':len(ws_exact_groups),'exact_whitespace_only_duplicate_excess':len(train)-len(ws_exact_groups),'exact_whitespace_only_conflicting_groups':len(ws_conflicts),'near_similarity_threshold':.85,'near_edges_ge_threshold':len(edge_i),'near_components_gt1':len(components),'near_components_rows':sum(components),'near_component_count_histogram':dict(sorted(Counter(components).items())),'near_group_approx_independent_rows':int(ncomp),'unique_normalized_texts_by_domain':{k:v['unique_normalized_texts'] for k,v in domain_metrics.items()}},
  'cross_split_exact_punctuation_insensitive':cross,'cross_split_exact_whitespace_only':cross_ws,'cross_split_near_train_tfidf':cross_near,'heuristic_flags':flag_metrics,
  'method':{'gold':'found_microcategories when present, otherwise public labels or addition gold; only additional IDs 102-106 retained; offered_microcategories reported separately and never used as gold','source_category':'use explicit source_mcId, else V3 mcId; public source101 is assigned by documented convention because source is absent from those records','official208_gold':'categories strings mapped to 102-106 by official category names','normalization':'Unicode casefold; replace every run of non-word characters with one space; description only','near_duplicate':'word TF-IDF unigram+bigrams, min_df=2, max_features=30000, sklearn default token pattern and default sublinear_tf=False; cosine >= .85 connected components; used only for approximate lexical/template families, not semantic identity','cross_split_near':'same train-fitted TF-IDF space; each held-out row compared to train in blocks; counts rows whose maximum train cosine meets thresholds','chunker':'avito_compare/scripts/fixed_chunker.py::chunk_ad_fixed defaults (target_chars=300,max_chars=420)','heuristics':'regex counts describe presence of lexical cues only; false positives and false negatives expected; no detector is treated as a gold label','no_classifier_training':True},
  'public_raw_reconstruction':{'rows':len(raw),'unique_punctuation_insensitive':len(raw_norm),'unique_whitespace_only':len(raw_ws_norm),'unique_punctuation_insensitive_found_in_public_partitions':len(raw_to_pub & set(raw_norm)),'raw_rows_punctuation_insensitive_matching_public_partitions':sum(raw_norm[x] for x in (raw_to_pub & set(raw_norm))),'unique_whitespace_only_found_in_public_partitions':len(raw_ws_to_pub & set(raw_ws_norm)),'raw_rows_whitespace_only_matching_public_partitions':sum(raw_ws_norm[x] for x in (raw_ws_to_pub & set(raw_ws_norm))),'note':'Raw 10k is a provenance comparator and is not added to training.'},
 'official_description_duplicate_groups':sum(len(v)>1 for v in og.values()),'official_description_duplicate_rows_in_groups':sum(len(v) for v in og.values() if len(v)>1),
 }
 (OUT/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 # Near-neighbor sample for inspection and fixed review sample, seeded & stratified by domain / gold-empty vs nonempty.
 def seeded_sample(rows,n,seed):
  rng=np.random.default_rng(seed); idx=rng.permutation(len(rows))[:min(n,len(rows))]; return [rows[int(i)] for i in idx]
 sample=[]
 for dom,rows in domains.items():
  for empt,n in [(False,3),(True,2)]:
   bucket=[r for r in rows if (not r['labels'])==empt]
   for r in seeded_sample(bucket,n,20261115+len(sample)):
    sample.append({'domain':dom,'id':r['id'],'labels':r['labels'],'title':r['title'],'text':r['text'],'sampling_stratum':'gold non-empty' if not empt else 'gold empty'})
 (OUT/'fixed_review_sample.json').write_text(json.dumps({'seed_base':20261115,'selection':'5 per training domain: 3 gold-nonempty + 2 gold-empty where available; deterministic sequential RNG seeds','items':sample},ensure_ascii=False,indent=2),encoding='utf-8')
 # Near neighbors (up to 100) and exact conflicting groups.
 def compact(r): return {'domain':r['domain'],'source_mcId':r['source_mcId'],'item_id':r.get('id'),'labels':r['labels'],'text':r['text']}
 (OUT/'near_duplicate_examples.json').write_text(json.dumps([{'cosine':round(s,4),'a':compact(train[i]),'b':compact(train[j])} for s,i,j in near[:30]],ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'cross_split_near_examples.json').write_text(json.dumps(cross_near_examples,ensure_ascii=False,indent=2),encoding='utf-8')
 (OUT/'exact_conflicts.json').write_text(json.dumps([[compact(r) for r in g] for g in exact_conflicts[:25]],ensure_ascii=False,indent=2),encoding='utf-8')
 # Graphics.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
 fig,ax=plt.subplots(figsize=(9,4.7)); ds=list(domains); x=np.arange(len(ds)); width=.13
 for j,l in enumerate(LABELS): ax.bar(x+(j-2)*width,[domain_metrics[d]['label_frequency'][str(l)]/len(domains[d])*100 for d in ds],width,label=f'{l} {NAMES[l]}')
 ax.set_xticks(x,ds);ax.set_ylabel('Доля объявлений с меткой, %');ax.set_title('Частота меток по источникам (метки многозначные)');ax.legend(ncol=3,fontsize=8);fig.tight_layout();fig.savefig(OUT/'label_rates.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(8,4.5)); fields=['length_chars','length_words','fixed_chunks']; labels=['Символы','Слова','Чанки fixed_chunker'];
 vals=[[domain_metrics[d][f]['median'] for d in ds] for f in fields];
 xx=np.arange(len(ds));w=.24
 for i,(lab,v) in enumerate(zip(labels,vals)): ax.bar(xx+(i-1)*w,v,w,label=lab)
 ax.set_xticks(xx,ds);ax.set_ylabel('Медиана');ax.set_title('Длины текстов по доменам');ax.legend();fig.tight_layout();fig.savefig(OUT/'lengths_by_domain.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,4.5)); a=[domain_metrics[d]['empty_gold_rate']*100 for d in ds]; b=[domain_metrics[d]['label_cardinality'].get('1',0)/len(domains[d])*100 for d in ds];
 ax.bar(np.arange(len(ds))-.18,a,.36,label='Пустые gold');ax.bar(np.arange(len(ds))+.18,b,.36,label='Ровно одна метка');ax.set_xticks(np.arange(len(ds)),ds);ax.set_ylabel('% строк');ax.set_title('Пустые метки и кардинальность');ax.legend();fig.tight_layout();fig.savefig(OUT/'gold_cardinality.png',dpi=160);plt.close(fig)
 print(json.dumps({'out':str(OUT),'train':len(train),'calibration':len(cal),'domain_metrics':domain_metrics,'combined':report['combined'],'cross':cross,'flags':flag_metrics,'raw':report['public_raw_reconstruction']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
