"""Restore original C_syntax training features and leakage-free OOF. No test scoring."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import run_syntax_experiment as se
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.eval_public_tests_syntax import reconstruct_v2_chunk_map
from avito_compare.scripts.qwen3_v3_experiment import build_tfidf_pipeline
from avito_compare.scripts.syntax_features import NON_SYNTAX_META_FEATURES,SYNTAX_ONLY_FEATURES,SYNTAX_CAT_FEATURES
P=Path('output/catboost-optimization')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
 with (P/'cache'/name).open('wb') as f:pickle.dump(x,f)
def main():
 chunks,emb,query=reconstruct_v2_chunk_map();emb=emb.astype(np.float32)
 z=np.load('output/cat-aware-v2/cache/chunks_embeddings.npz');off=z['ad_chunk_offsets'];ix=z['ad_chunk_indices']
 raw=emb@query.T;sims=np.column_stack([raw[:,j*3:(j+1)*3].max(1) for j in range(6)])
 splits={f'pub_{s}':cae.load_public_rows(Path('output/ngram-comparison/splits')/f'{s}.jsonl') for s in ['train','calibration']}
 splits.update({f'v3_{s}':cae.load_v3_rows_full(Path('datasets/own_v3_clean')/f'{s}.jsonl') for s in ['train','validation']})
 rows={'train':splits['pub_train']+splits['v3_train'],'cal':splits['pub_calibration']+splits['v3_validation']}
 uids={'train':np.concatenate([z['uid_pub_train'],z['uid_v3_train']]),'cal':np.concatenate([z['uid_pub_calibration'],z['uid_v3_validation']])}
 feats={name:cae.build_category_features_for_uids(u,off,ix,emb,sims,return_selection=False)[0] for name,u in uids.items()}
 y,src=cae._rows_xy_for_ids(rows['train'],cae.SCOPE_102_106,use_eval=False)
 groups,diag=se.compute_dup_groups([r['description'] for r in rows['train']]);assert len(rows['train'])==6319 and diag['n_components']==5484
 print('Original groups',diag,flush=True)
 oof_path=P/'cache/original_oof.npz'
 if oof_path.exists():oof=np.load(oof_path)['oof'];alphas=json.loads((P/'oof_diagnostics.json').read_text())['alphas']
 else:
  start=time.monotonic();oof,alphas=se.fit_leakage_free_oof_sgd(feats['train'],y,groups)
  np.savez(oof_path,oof=oof,groups=groups)
  (P/'oof_diagnostics.json').write_text(json.dumps({'groups':diag,'alphas':alphas,'seconds':time.monotonic()-start,'same_baseline_groups':True},indent=2))
 print('Original Qwen OOF restored',flush=True)
 heads=joblib.load('output/syntax-aware-experiment/models/sgd_full_heads.joblib')['heads']
 pcal=np.column_stack([heads[c].predict_proba(feats['cal'][c])[:,1] for c in cae.SCOPE_102_106])
 with Path('output/syntax-aware-experiment/cache/chunk_syntax_cache.pkl').open('rb') as f:cache=pickle.load(f)
 frames={}
 for name in rows:
  yy,ss=cae._rows_xy_for_ids(rows[name],cae.SCOPE_102_106,use_eval=True)
  frames[name]=se.build_pair_dataframe(name,rows[name],uids[name],off,ix,sims,chunks,yy,ss,oof if name=='train' else pcal,cache['chunk_sig_cache'],cache['chunk_global_cache'],groups if name=='train' else None)
 frames['train']['domain']=np.where(frames['train'].row_idx<3398,'public','v3')
 frames['cal']['domain']=np.where(frames['cal'].row_idx<728,'public','v3')
 cols=['p_sgd']+NON_SYNTAX_META_FEATURES+SYNTAX_ONLY_FEATURES
 save('selection.pkl',{'train':frames['train'],'cal':frames['cal'],'rows':rows,'cols':cols,'categorical':SYNTAX_CAT_FEATURES,'groups':groups})
 # One shared group-protected partition for TF-IDF; vectorizer fitted inside each fold.
 tf_path=P/'cache/tfidf_oof.npy'
 if tf_path.exists():ptf=np.load(tf_path)
 else:
  ptf=np.zeros_like(oof);texts=np.asarray([r['description'] for r in rows['train']]);folds=[]
  cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
  for fold,(tr,va) in enumerate(cv.split(texts,y.sum(1),groups)):
   assert not set(groups[tr])&set(groups[va]);pipe=build_tfidf_pipeline();pipe.fit(texts[tr].tolist(),y[tr]);ptf[va]=pipe.predict_proba(texts[va].tolist());folds.append({'train':tr.tolist(),'holdout':va.tolist()});print('TF-IDF OOF fold',fold+1,flush=True)
  np.save(tf_path,ptf);(P/'tfidf_oof_folds.json').write_text(json.dumps(folds))
 fulltf=joblib.load('output/syntax-aware-experiment/models/tfidf_baseline.joblib');ctf=fulltf.predict_proba([r['description'] for r in rows['cal']])
 for name,mat in [('train',ptf),('cal',ctf)]:
  df=frames[name];j=df.cat_id.astype(int).to_numpy()-102;df['p_tfidf']=mat[df.row_idx.to_numpy(),j]
 save('selection.pkl',{'train':frames['train'],'cal':frames['cal'],'rows':rows,'cols':cols,'categorical':SYNTAX_CAT_FEATURES,'groups':groups})
 # Evaluation records are in a separate file and not consumed by search.
 ev=pd.read_csv('output/syntax-aware-experiment/syntax_features.csv')
 prob=pd.read_csv('output/syntax-aware-experiment/predictions.csv')
 ev=ev.merge(prob[['split','row_idx','cat_id','p_tfidf','p_C','pred_C_05']],on=['split','row_idx','cat_id'],validate='one_to_one')
 official=pd.read_csv('output/syntax-aware-experiment/public_tests_predictions.csv');official=official[official.split=='official_208'].copy()
 evalframes={n:ev[ev.split==n].copy() for n in ['pub_validation','v3_test']};evalframes['official_208']=official
 from avito_compare.scripts.qwen3_embedding_experiment import load_official_parquet
 _,_,orows=load_official_parquet(Path('datasets/public/public-test-dataset.parquet'))
 evalrows={'pub_validation':cae.load_public_rows(Path('output/ngram-comparison/splits/validation.jsonl')),'v3_test':cae.load_v3_rows_full(Path('datasets/own_v3_clean/test.jsonl')),'official_208':orows}
 for name,rs in evalrows.items():
  probs=fulltf.predict_proba([r['description'] for r in rs]);df=evalframes[name];df['p_tfidf']=probs[df.row_idx.to_numpy(),df.cat_id.astype(int).to_numpy()-102]
 from collections import defaultdict
 labelsets=defaultdict(set)
 norm=lambda t:' '.join(t.casefold().split())
 for r in orows:labelsets[norm(r['description'])].add(tuple(r['evaluation_labels']))
 conflicting={t for t,v in labelsets.items() if len(v)>1};excluded=[i for i,r in enumerate(orows) if norm(r['description']) in conflicting]
 assert len(excluded)==56
 (P/'public_test_exclusions.json').write_text(json.dumps({'policy':'Exclude every row in normalized-description groups with conflicting gold; no label edits, no train-overlap removal in this clean definition. Same fixed list for every model.','excluded_row_indices_zero_based':excluded,'remaining':208-len(excluded),'conflicting_groups':len(conflicting)},indent=2))
 save('final_eval.pkl',{'frames':evalframes,'rows':evalrows,'clean_exclusions':excluded})
 (P/'input_manifest.json').write_text(json.dumps({'train_ads':6319,'calibration_ads':1328,'no_hard_negative_addition':True,'threshold_default':.5,'grouping':diag,'features':cols,'categorical':SYNTAX_CAT_FEATURES,'source_sha256':{str(p):sha(p) for p in [Path('datasets/own_v3_clean/train.jsonl'),Path('datasets/own_v3_clean/validation.jsonl'),Path('datasets/own_v3_clean/test.jsonl'),Path('output/ngram-comparison/splits/train.jsonl'),Path('output/ngram-comparison/splits/calibration.jsonl'),Path('avito_compare/scripts/fixed_chunker.py'),Path('avito_compare/scripts/syntax_features.py'),Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')]}},indent=2))
 print('Prepared original train/cal and separate frozen eval cache.',flush=True)
if __name__=='__main__':main()
