"""Train only augmented C_syntax using existing retrieval, OOF and syntax functions.
Evaluation is a separate, one-shot script after successful training validation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts import run_syntax_experiment as se
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.syntax_features import SyntaxAnalyzer,GLOBAL_CANDIDATE_RE,extract_chunk_syntax_signals,detect_chunk_global_restriction,NON_SYNTAX_META_FEATURES,SYNTAX_ONLY_FEATURES,SYNTAX_CAT_FEATURES
P=Path('output/hard-negatives-v3-experiment')
def main():
 config=json.loads((P/'config.json').read_text())
 assert hashlib.sha256(Path('datasets/v3_hard_negatives_200/train_addition.jsonl').read_bytes()).hexdigest()==config['dataset_sha256']
 with (P/'cache/prepared.pkl').open('rb') as f:prepared=pickle.load(f)
 splits=prepared['splits'];lookup=prepared['lookup'];queries=prepared['queries']
 new=json.loads((P/'cache/delta_chunks.json').read_text());delta=np.load(P/'cache/delta_checkpoint.npz')['embs'];assert len(new)==len(delta)
 lookup.update(zip(new,delta))
 chunks=list(lookup);ctoid={c:i for i,c in enumerate(chunks)};emb=np.asarray(list(lookup.values()),dtype=np.float32)
 raw=emb@queries.T;sims=np.column_stack([raw[:,j*3:(j+1)*3].max(1) for j in range(6)])
 allrows={**splits};allrows['train']=splits['pub_train']+splits['v3_train']+splits['hard_negative_train'];allrows['calibration']=splits['pub_calibration']+splits['v3_validation']
 ads=[];admap={};offsets=[0];indices=[];uids={};features={}
 for name,rows in allrows.items():
  ids=[]
  for r in rows:
   txt=r['description']
   if txt not in admap:
    admap[txt]=len(ads);ads.append(txt);indices.extend(ctoid[c] for c in chunk_ad_fixed(txt));offsets.append(len(indices))
   ids.append(admap[txt])
  uids[name]=np.asarray(ids)
 offsets=np.asarray(offsets);indices=np.asarray(indices)
 for name,ids in uids.items():features[name]=cae.build_category_features_for_uids(ids,offsets,indices,emb,sims,return_selection=False)[0]
 train=allrows['train'];cal=allrows['calibration'];assert len(train)==6519 and len(cal)==1328
 ytr,src=cae._rows_xy_for_ids(train,cae.SCOPE_102_106,use_eval=False)
 yev,_=cae._rows_xy_for_ids(train,cae.SCOPE_102_106,use_eval=True)
 groups,diag=se.compute_dup_groups([r['description'] for r in train]);print('Groups',diag,flush=True)
 # Explicitly store disjoint outer fold memberships for every category.
 fold_manifest={}
 for j,cid in enumerate(cae.SCOPE_102_106):
  folds=[]
  for tr,va in se.StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42).split(np.zeros(len(train)),ytr[:,j],groups):
   assert not set(groups[tr])&set(groups[va]);folds.append({'train_indices':tr.tolist(),'validation_indices':va.tolist()})
  fold_manifest[str(cid)]=folds
 (P/'oof_folds.json').write_text(json.dumps(fold_manifest))
 print('Recomputing all nested grouped OOF probabilities...',flush=True);start=time.monotonic()
 oof,alphas=se.fit_leakage_free_oof_sgd(features['train'],ytr,groups)
 assert np.isfinite(oof).all()
 print('OOF complete',time.monotonic()-start,flush=True)
 heads={};head_diag={}
 for j,cid in enumerate(cae.SCOPE_102_106):
  heads[cid],alpha,_,_=cae._fit_cv_binary_head(features['train'][cid],ytr[:,j]);head_diag[str(cid)]=float(alpha)
 joblib.dump({'heads':heads,'ids':cae.SCOPE_102_106,'diag':head_diag},P/'models/sgd_heads_B.joblib')
 with Path('output/syntax-aware-experiment/cache/chunk_syntax_cache.pkl').open('rb') as f:old=pickle.load(f)
 signals=dict(old['chunk_sig_cache']);globals_=dict(old['chunk_global_cache']);analyzer=None
 pairs_needed=set();global_needed=set()
 for name in ['train','calibration','pub_validation','v3_test','official_208']:
  for i,uid in enumerate(uids[name]):
   ix=indices[offsets[uid]:offsets[uid+1]]
   for cidx in ix:
    if GLOBAL_CANDIDATE_RE.search(chunks[cidx]):global_needed.add(chunks[cidx])
   for cid in cae.SCOPE_102_106:
    if allrows[name][i].get('mcId')==cid:continue
    rank=np.argsort(-sims[ix,cae.ALL_IDS.index(cid)],kind='stable')[:3]
    pairs_needed.update((chunks[ix[p]],cid) for p in rank)
 print('Missing syntax pairs',len(pairs_needed-signals.keys()),flush=True)
 for i,(txt,cid) in enumerate(sorted(pairs_needed-signals.keys())):
  if analyzer is None:analyzer=SyntaxAnalyzer()
  signals[(txt,cid)]=extract_chunk_syntax_signals(analyzer,txt,cid)
  if i%100==0:print('syntax',i,flush=True)
 for txt in sorted(global_needed-globals_.keys()):
  if analyzer is None:analyzer=SyntaxAnalyzer()
  globals_[txt]=detect_chunk_global_restriction(analyzer,txt)
 frames={}
 for name in ['train','calibration','pub_validation','v3_test','official_208']:
  rows=allrows[name];gold,sources=cae._rows_xy_for_ids(rows,cae.SCOPE_102_106,use_eval=True)
  probs=oof if name=='train' else np.column_stack([heads[cid].predict_proba(features[name][cid])[:,1] for cid in cae.SCOPE_102_106])
  frames[name]=se.build_pair_dataframe(name,rows,uids[name],offsets,indices,sims,chunks,gold,sources,probs,signals,globals_,dup_groups=groups if name=='train' else None)
 cols=['p_sgd']+NON_SYNTAX_META_FEATURES+SYNTAX_ONLY_FEATURES
 for frame in frames.values():assert not frame[cols].isna().any().any()
 print('Fitting existing C_syntax CatBoost...',flush=True)
 cb=CatBoostClassifier(**se.CATBOOST_PARAMS)
 cb.fit(Pool(frames['train'][cols],frames['train']['y_true'],cat_features=SYNTAX_CAT_FEATURES),eval_set=Pool(frames['calibration'][cols],frames['calibration']['y_true'],cat_features=SYNTAX_CAT_FEATURES),use_best_model=True)
 cb.save_model(str(P/'models/catboost_C_syntax_B.cbm'))
 # Save features/probabilities without computing or viewing test metrics.
 with (P/'cache/trained.pkl').open('wb') as f:pickle.dump({'frames':frames,'rows':allrows,'cols':cols,'groups':groups},f)
 (P/'training_validation.json').write_text(json.dumps({'status':'passed','train_rows':len(train),'calibration_rows':len(cal),'oof_all_recomputed':True,'group_diagnostics':diag,'oof_alphas':alphas,'head_alphas':head_diag,'best_iteration':cb.get_best_iteration(),'parameters':se.CATBOOST_PARAMS,'columns':cols,'categorical_columns':SYNTAX_CAT_FEATURES,'training_seconds':time.monotonic()-start},indent=2))
 print('TRAINING VALIDATED; no final test metrics computed.',flush=True)
if __name__=='__main__':main()
