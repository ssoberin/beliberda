"""Independent count/OOF/integrity checks after the single final evaluation."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
P=Path('output/tail-services-v3-experiment')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 config=json.loads((P/'config.json').read_text());freeze=json.loads((P/'frozen_inputs.json').read_text());m=json.loads((P/'metrics.json').read_text())
 for p,h in {**config['source_sha256'],**freeze['files']}.items():assert sha(p)==h,p
 assert sha('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')==config['baseline_model_sha256']
 assert sha('output/syntax-aware-experiment/models/sgd_full_heads.joblib')==config['baseline_heads_sha256']
 z=np.load(P/'cache/oof_B.npz');oof=z['oof'];groups=z['groups'];assert oof.shape==(6379,5) and np.isfinite(oof).all()
 folds=json.loads((P/'oof_folds.json').read_text())
 for cid,fs in folds.items():
  seen=[]
  for f in fs:
   tr=np.asarray(f['train_indices']);va=np.asarray(f['validation_indices']);assert not set(groups[tr])&set(groups[va]);assert not set(tr)&set(va);assert len(tr)+len(va)==6379;seen.extend(va)
  assert sorted(seen)==list(range(6379))
 pred=pd.read_csv(P/'predictions.csv')
 for split,n in [('pub_validation',729),('v3_test',600),('official_208',208),('tail_holdout20',20)]:
  f=pred[pred.split==split];assert len(f)==n*5;assert not f.duplicated(['row_idx','cat_id']).any()
  assert all(f[f.active==0][c].eq(0).all() for c in ['gold','pred_A','pred_B','p_A','p_B'])
  for model in ['A','B']:
   p=f['pred_'+model];y=f.gold;tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum());r=m[split][model]
   assert (tp,fp,fn)==(r['tp'],r['fp'],r['fn']);assert abs(2*tp/max(1,2*tp+fp+fn)-r['micro_f1'])<1e-12
   assert np.array_equal((f['p_'+model]>=.5).astype(int),p)
   for cid in range(102,107):
    g=f[f.cat_id==cid];gg=g.gold;pp=g['pred_'+model];c=r['per_category'][str(cid)]
    assert int(((gg==0)&(pp==1)).sum())==c['fp'];assert int(((gg==1)&(pp==0)).sum())==c['fn']
  assert m[split]['transitions']['net_gain']==m[split]['A']['fp']+m[split]['A']['fn']-m[split]['B']['fp']-m[split]['B']['fn']
 out={'status':'passed','frozen_inputs_unchanged':True,'baseline_weights_unchanged':True,'old_dataset_hashes_unchanged':True,'outer_oof_group_disjoint':True,'oof_rows':6379,'new_train_ads':60,'new_holdout_ads':20,'threshold':.5,'metrics_independently_recounted':True,'no_baseline_replacement':True}
 (P/'verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
