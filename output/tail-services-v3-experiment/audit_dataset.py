"""Text-only duplicate audit; never use predictions or existing test labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,json,sys
from pathlib import Path
import duckdb,joblib,numpy as np
from sklearn.metrics.pairwise import cosine_similarity
sys.path.insert(0,str(Path.cwd()))
P=Path('output/tail-services-v3-experiment');P.mkdir(parents=True,exist_ok=True)
D=Path('datasets/v3_tail_services_60')
def load(p):return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]
def norm(s):return ' '.join(s.casefold().replace('ё','е').split())
def main():
 tr=load(D/'train_addition.jsonl');te=load(D/'test_holdout.jsonl');new=tr+te
 assert len(tr)==60 and len(te)==20
 assert all(r['mcId']==r['source_mcId']==101 and r['gold']==[x['mcId'] for x in r['found_microcategories']]==r['offered_microcategories'] for r in new)
 assert len({norm(r['description']) for r in new})==80
 pools={}
 for name,p in [('pub_train','output/ngram-comparison/splits/train.jsonl'),('pub_calibration','output/ngram-comparison/splits/calibration.jsonl'),('pub_validation','output/ngram-comparison/splits/validation.jsonl'),('v3_train','datasets/own_v3_clean/train.jsonl'),('v3_validation','datasets/own_v3_clean/validation.jsonl'),('v3_test','datasets/own_v3_clean/test.jsonl'),('context20','datasets/context_pairs_100/test_20.jsonl')]:pools[name]=[r['description'] for r in load(p)]
 pools['official_208']=[t for t, in duckdb.connect().execute('select description from read_parquet(?)',['datasets/public/public-test-dataset.parquet']).fetchall()]
 pipe=joblib.load('output/syntax-aware-experiment/models/tfidf_baseline.joblib');vec=pipe[:-1]
 texts=[r['description'] for r in new];x=vec.transform(texts);inside=cosine_similarity(x);np.fill_diagonal(inside,-1)
 nearest={};exact={}
 for name,old in pools.items():
  eq={norm(t) for t in old};exact[name]=sum(norm(t) in eq for t in texts)
  sim=cosine_similarity(x,vec.transform(old));ix=sim.argmax(1)
  nearest[name]=[{'item_id':r['item_id'],'row_idx':int(j),'similarity':float(sim[i,j]),'neighbor_description':old[j]} for i,(r,j) in enumerate(zip(new,ix))]
 assert sum(exact.values())==0
 # Retain all potentially close pairs for semantic inspection; no auto label changes.
 pairs=[{'left':new[i]['item_id'],'right':new[j]['item_id'],'similarity':float(inside[i,j]),'left_text':texts[i],'right_text':texts[j]} for i in range(len(new)) for j in range(i) if inside[i,j]>=.5]
 assert float(inside[:60,60:].max())<.85,'New train/holdout near duplicate needs replacement before freeze'
 stats={}
 for name,rows in [('train60',tr),('holdout20',te)]:
  lens=np.array([len(r['description']) for r in rows]);reviews=json.loads((D/'annotation_review.json').read_text());rev={a['item_id']:a for a in reviews}
  stats[name]={'rows':len(rows),'empty_gold':sum(not r['gold'] for r in rows),'category_counts':{str(c):sum(c in r['gold'] for r in rows) for c in range(102,107)},'focus':dict(collections.Counter(rev[r['item_id']]['focus'] for r in rows)),'length':{'min':int(lens.min()),'median':float(np.median(lens)),'max':int(lens.max())}}
 out={'stats':stats,'exact_overlaps':exact,'max_internal_similarity':float(inside.max()),'max_new_train_holdout_similarity':float(inside[:60,60:].max()),'near_pairs_over_0_5':pairs,'max_similarity_by_existing_split':{name:max(a['similarity'] for a in arr) for name,arr in nearest.items()},'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'train_addition.jsonl',D/'test_holdout.jsonl',D/'annotation_review.json']}}
 (P/'dataset_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');(P/'nearest_neighbors.json').write_text(json.dumps(nearest,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({k:v for k,v in out.items() if k!='near_pairs_over_0_5'},ensure_ascii=False,indent=2));print('Internal review pairs',len(pairs))
if __name__=='__main__':main()
