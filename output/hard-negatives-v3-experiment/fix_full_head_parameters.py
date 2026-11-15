"""Training validation: retain baseline full-head parameters, reuse compatible new OOF.

The existing helper auto-selects alpha for full heads. For this strict data-only
A/B, full heads must instead clone baseline parameters. No test scores are read.
Grouped nested OOF was fully recomputed on B and remains unchanged/compatible.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import json,pickle,sys
from pathlib import Path
import joblib,numpy as np
from sklearn.base import clone
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts.fixed_chunker import chunk_ad_fixed
from avito_compare.scripts.run_syntax_experiment import CATBOOST_PARAMS
from avito_compare.scripts.syntax_features import SYNTAX_CAT_FEATURES
P=Path('output/hard-negatives-v3-experiment')
assert not (P/'metrics.json').exists(),'Must run before final evaluation'
with (P/'cache/prepared.pkl').open('rb') as f:pre=pickle.load(f)
with (P/'cache/trained.pkl').open('rb') as f:s=pickle.load(f)
lookup=pre['lookup'];delta=np.load(P/'cache/delta_checkpoint.npz')['embs'];lookup.update(zip(json.loads((P/'cache/delta_chunks.json').read_text()),delta))
chunks=list(lookup);ctoi={t:i for i,t in enumerate(chunks)};emb=np.asarray(list(lookup.values()),dtype=np.float32)
raw=emb@pre['queries'].T;sims=np.column_stack([raw[:,j*3:(j+1)*3].max(1) for j in range(6)])
features={}
for name in ['train','calibration','pub_validation','v3_test','official_208']:
 rows=s['rows'][name];indices=[];offsets=[0]
 for r in rows:
  indices.extend(ctoi[t] for t in chunk_ad_fixed(r['description']));offsets.append(len(indices))
 features[name]=cae.build_category_features_for_uids(np.arange(len(rows)),np.asarray(offsets),np.asarray(indices),emb,sims,return_selection=False)[0]
ytr,_=cae._rows_xy_for_ids(s['rows']['train'],cae.SCOPE_102_106,use_eval=False)
baseline=joblib.load('output/syntax-aware-experiment/models/sgd_full_heads.joblib');heads={}
for j,cid in enumerate(cae.SCOPE_102_106):
 heads[cid]=clone(baseline['heads'][cid]).fit(features['train'][cid],ytr[:,j])
 assert heads[cid].get_params()==baseline['heads'][cid].get_params()
joblib.dump({'heads':heads,'ids':cae.SCOPE_102_106,'diag':{str(c):h.alpha for c,h in heads.items()}},P/'models/sgd_heads_B.joblib')
for name in ['calibration','pub_validation','v3_test','official_208']:
 frame=s['frames'][name]
 for cid in cae.SCOPE_102_106:
  ix=frame['cat_id'].astype(int)==cid
  frame.loc[ix,'p_sgd']=heads[cid].predict_proba(features[name][cid])[:,1][frame.loc[ix,'row_idx'].to_numpy()]
cols=s['cols'];train=s['frames']['train'];cal=s['frames']['calibration']
cb=CatBoostClassifier(**CATBOOST_PARAMS)
cb.fit(Pool(train[cols],train['y_true'],cat_features=SYNTAX_CAT_FEATURES),eval_set=Pool(cal[cols],cal['y_true'],cat_features=SYNTAX_CAT_FEATURES),use_best_model=True)
cb.save_model(str(P/'models/catboost_C_syntax_B.cbm'))
with (P/'cache/trained.pkl').open('wb') as f:pickle.dump(s,f)
v=json.loads((P/'training_validation.json').read_text());v.update(full_head_parameters_match_baseline=True,head_alphas={str(c):h.alpha for c,h in heads.items()},best_iteration=cb.get_best_iteration(),pre_evaluation_validation_correction='Full-head CV replaced with baseline parameter clones; current B grouped nested OOF values retained unchanged, no old OOF mixed.')
(P/'training_validation.json').write_text(json.dumps(v,indent=2)+'\n')
print('VALIDATED: exact baseline full-head parameters; no test metrics computed.')
