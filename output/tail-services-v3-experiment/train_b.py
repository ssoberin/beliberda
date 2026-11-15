"""Train only augmented C_syntax using existing retrieval, OOF and syntax functions.
Evaluation is a separate, one-shot script after successful training validation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
from sklearn.base import clone
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts import run_syntax_experiment as se
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.syntax_features import SyntaxAnalyzer,GLOBAL_CANDIDATE_RE,extract_chunk_syntax_signals,detect_chunk_global_restriction,NON_SYNTAX_META_FEATURES,SYNTAX_ONLY_FEATURES,SYNTAX_CAT_FEATURES
P=Path('output/tail-services-v3-experiment');(P/'models').mkdir(exist_ok=True,parents=True)
def main():
 config=json.loads((P/'config.json').read_text())
 assert hashlib.sha256(Path('datasets/v3_tail_services_60/train_addition.jsonl').read_bytes()).hexdigest()==config['dataset_sha256']
 assert hashlib.sha256(Path('datasets/v3_tail_services_60/test_holdout.jsonl').read_bytes()).hexdigest()==config['new_holdout_sha256']
 for p,h in config['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
 assert hashlib.sha256(Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm').read_bytes()).hexdigest()==config['baseline_model_sha256']
 assert hashlib.sha256(Path('output/syntax-aware-experiment/models/sgd_full_heads.joblib').read_bytes()).hexdigest()==config['baseline_heads_sha256']
 with (P/'cache/prepared.pkl').open('rb') as f:prepared=pickle.load(f)
 splits=prepared['splits'];lookup=prepared['lookup'];queries=prepared['queries']
 new=json.loads((P/'cache/delta_chunks.json').read_text());delta=np.load(P/'cache/delta_checkpoint.npz')['embs'];assert len(new)==len(delta)
 lookup.update(zip(new,delta))
 chunks=list(lookup);ctoid={c:i for i,c in enumerate(chunks)};emb=np.asarray(list(lookup.values()),dtype=np.float32)
 raw=emb@queries.T;sims=np.column_stack([raw[:,j*3:(j+1)*3].max(1) for j in range(6)])
 allrows={**splits};allrows['train']=splits['pub_train']+splits['v3_train']+splits['tail_train'];allrows['calibration']=splits['pub_calibration']+splits['v3_validation']
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
 train=allrows['train'];cal=allrows['calibration'];assert len(train)==6379 and len(cal)==1328
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
 np.savez_compressed(P/'cache/oof_B.npz',oof=oof,groups=groups)
 (P/'oof_alphas.json').write_text(json.dumps(alphas))
 assert np.isfinite(oof).all()
 print('OOF complete',time.monotonic()-start,flush=True)
 heads={};head_diag={};baseline_heads=joblib.load('output/syntax-aware-experiment/models/sgd_full_heads.joblib')['heads']
 for j,cid in enumerate(cae.SCOPE_102_106):
  heads[cid]=clone(baseline_heads[cid]).fit(features['train'][cid],ytr[:,j]);assert heads[cid].get_params()==baseline_heads[cid].get_params();head_diag[str(cid)]=float(heads[cid].alpha)
 joblib.dump({'heads':heads,'ids':cae.SCOPE_102_106,'diag':head_diag},P/'models/sgd_heads_B.joblib')
 with Path('output/syntax-aware-experiment/cache/chunk_syntax_cache.pkl').open('rb') as f:old=pickle.load(f)
 signals=dict(old['chunk_sig_cache']);globals_=dict(old['chunk_global_cache']);analyzer=None
 pairs_needed=set();global_needed=set()
 for name in ['train','calibration','pub_validation','v3_test','official_208','tail_holdout20']:
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
 for name in ['train','calibration','pub_validation','v3_test','official_208','tail_holdout20']:
  rows=allrows[name];gold,sources=cae._rows_xy_for_ids(rows,cae.SCOPE_102_106,use_eval=True)
  probs=oof if name=='train' else np.column_stack([heads[cid].predict_proba(features[name][cid])[:,1] for cid in cae.SCOPE_102_106])
  frames[name]=se.build_pair_dataframe(name,rows,uids[name],offsets,indices,sims,chunks,gold,sources,probs,signals,globals_,dup_groups=groups if name=='train' else None)
 cols=['p_sgd']+NON_SYNTAX_META_FEATURES+SYNTAX_ONLY_FEATURES
 for frame in frames.values():assert not frame[cols].isna().any().any()
 # Full baseline inference on the new holdout: same frozen input features, different head weights.
 baseline_holdout=frames['tail_holdout20'].copy()
 for cid in cae.SCOPE_102_106:
  mask=baseline_holdout.cat_id.astype(int)==cid
  baseline_holdout.loc[mask,'p_sgd']=baseline_heads[cid].predict_proba(features['tail_holdout20'][cid])[:,1][baseline_holdout.loc[mask,'row_idx'].to_numpy()]
 print('Fitting existing C_syntax CatBoost...',flush=True)
 cb=CatBoostClassifier(**se.CATBOOST_PARAMS)
 cb.fit(Pool(frames['train'][cols],frames['train']['y_true'],cat_features=SYNTAX_CAT_FEATURES),eval_set=Pool(frames['calibration'][cols],frames['calibration']['y_true'],cat_features=SYNTAX_CAT_FEATURES),use_best_model=True)
 cb.save_model(str(P/'models/catboost_C_syntax_B.cbm'))
 # Save features/probabilities without computing or viewing test metrics.
 with (P/'cache/trained.pkl').open('wb') as f:pickle.dump({'frames':frames,'rows':allrows,'cols':cols,'groups':groups,'baseline_holdout':baseline_holdout},f)
 (P/'training_validation.json').write_text(json.dumps({'status':'passed','full_head_parameters_match_baseline':True,'train_rows':len(train),'calibration_rows':len(cal),'oof_all_recomputed':True,'group_diagnostics':diag,'oof_alphas':alphas,'head_alphas':head_diag,'best_iteration':cb.get_best_iteration(),'parameters':se.CATBOOST_PARAMS,'columns':cols,'categorical_columns':SYNTAX_CAT_FEATURES,'training_seconds':time.monotonic()-start},indent=2))
 print('TRAINING VALIDATED; no final test metrics computed.',flush=True)
if __name__=='__main__':main()
