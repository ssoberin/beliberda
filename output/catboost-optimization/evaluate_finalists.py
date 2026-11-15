"""Score calibration-selected, frozen finalists once. No training or threshold selection."""
import hashlib,json,pickle,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()));sys.path.insert(0,str(Path(__file__).resolve().parent))
from search import transform
from avito_compare.scripts.run_syntax_experiment import probs_to_eval_matrix,bootstrap_f1_delta,error_transition_counts
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics
from avito_compare.scripts import qwen3_category_aware_experiment as cae
P=Path('output/catboost-optimization')
def grouped_bootstrap(gold,a,b,texts):
 groups={}
 for i,t in enumerate(texts):groups.setdefault(' '.join(t.casefold().split()),[]).append(i)
 counts=[]
 for ids in groups.values():
  counts.append([[int(((pred[ids]==1)&(gold[ids]==1)).sum()),int(((pred[ids]==1)&(gold[ids]==0)).sum()),int(((pred[ids]==0)&(gold[ids]==1)).sum())] for pred in [a,b]])
 counts=np.asarray(counts);rng=np.random.default_rng(42);d=[]
 for _ in range(2000):
  x=counts[rng.integers(0,len(counts),len(counts))].sum(0)
  f=2*x[:,0]/np.maximum(1,2*x[:,0]+x[:,1]+x[:,2]);d.append(f[1]-f[0])
 return {'groups':len(groups),'ci95_low':float(np.percentile(d,2.5)),'ci95_high':float(np.percentile(d,97.5)),'delta_mean':float(np.mean(d))}
def main():
 if (P/'final_metrics.json').exists():raise RuntimeError('Finalist scoring already complete: do not tune after evaluation')
 frozen=json.loads((P/'finalists.json').read_text());assert frozen['frozen_before_test_scoring']
 for name,h in frozen['model_sha256'].items():assert hashlib.sha256((P/'models'/f'{name}.cbm').read_bytes()).hexdigest()==h
 manifest=json.loads((P/'input_manifest.json').read_text())
 for path,h in manifest['source_sha256'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h,path
 with (P/'cache/final_eval.pkl').open('rb') as f:ev=pickle.load(f)
 experiments={r['spec']['name']:r for r in json.loads((P/'experiments.json').read_text())}
 specs=[{'experiment':'C_syntax_baseline','mode':'fixed','thresholds':[.5]*5,'spec':{}}]+frozen['finalists']
 output={};predrecords=[];comparisons=[];probstore={};baseline_y={};rowsstore={}
 fulltf=joblib.load('output/syntax-aware-experiment/models/tfidf_baseline.joblib')
 for dataset,df in ev['frames'].items():
  mat=fulltf.predict_proba([r['description'] for r in ev['rows'][dataset]])
  df['p_tfidf']=mat[df.row_idx.to_numpy(),df.cat_id.astype(int).to_numpy()-102]
 for item in specs:
  name=item['experiment']+'__'+item['mode'];spec=item['spec'];th=np.asarray(item['thresholds'])
  model=CatBoostClassifier();model.load_model('output/syntax-aware-experiment/models/catboost_C_syntax.cbm' if item['experiment']=='C_syntax_baseline' else str(P/'models'/f"{item['experiment']}.cbm"))
  scores={};probs={};preds={}
  for dataset,original in ev['frames'].items():
   df,cols,cats=transform(original,manifest['features'],manifest['categorical'],spec)
   assert cols==model.feature_names_,(name,cols,model.feature_names_)
   start=time.perf_counter();p=model.predict_proba(Pool(df[cols],cat_features=cats))[:,1];seconds=time.perf_counter()-start
   rows=ev['rows'][dataset];gold,src=cae._rows_xy_for_ids(rows,cae.SCOPE_102_106,use_eval=True)
   mat=probs_to_eval_matrix(df,p,len(rows),src,cae.SCOPE_102_106);m,yp=compute_detailed_metrics(gold,mat,th,src,cae.SCOPE_102_106)
   m.update(m['confusion']);m['prediction_seconds']=seconds
   empty=gold.sum(1)==0;m['fp_on_empty_gold']=int(yp[empty].sum());m['fn_on_nonempty_gold']=int(((gold==1)&(yp==0)).sum())
   scores[dataset]=m;probs[dataset]=mat;preds[dataset]=yp;rowsstore[dataset]=rows
   if item['experiment']=='C_syntax_baseline':
    baseline_y[dataset]=yp
    saved=df.pred_C_05.to_numpy();actual=(p>=.5).astype(int)
    assert np.array_equal(saved,actual),('Baseline predictions differ',dataset,int((saved!=actual).sum()))
   else:
    baseline=baseline_y[dataset];trans=error_transition_counts(gold,baseline,yp,src,cae.SCOPE_102_106)
    trans['new_erroneous_empty_ads']=int(((baseline.sum(1)>0)&(yp.sum(1)==0)&(gold.sum(1)>0)).sum())
    m['transitions']=trans;m['bootstrap_delta']=bootstrap_f1_delta(gold,baseline,yp,src,cae.SCOPE_102_106,n_boot=2000)
    if dataset=='official_208':m['grouped_bootstrap_delta']=grouped_bootstrap(gold,baseline,yp,[r['description'] for r in rows])
    for i,r in enumerate(rows):
     errors_before=int((baseline[i]!=gold[i]).sum());errors_after=int((yp[i]!=gold[i]).sum())
     if not np.array_equal(baseline[i],yp[i]):comparisons.append({'model':name,'dataset':dataset,'row_idx':i,'source_mcId':int(src[i]),'description':r['description'],'gold':np.asarray(cae.SCOPE_102_106)[gold[i].astype(bool)].tolist(),'baseline':np.asarray(cae.SCOPE_102_106)[baseline[i].astype(bool)].tolist(),'candidate':np.asarray(cae.SCOPE_102_106)[yp[i].astype(bool)].tolist(),'errors_before':errors_before,'errors_after':errors_after})
   for i,r in df.iterrows():
    j=cae.SCOPE_102_106.index(int(r.cat_id));row=int(r.row_idx)
    predrecords.append({'model':name,'dataset':dataset,'row_idx':row,'source_mcId':int(src[row]),'cat_id':int(r.cat_id),'gold':int(gold[row,j]),'probability':float(mat[row,j]),'threshold':float(th[j]),'prediction':int(yp[row,j])})
   if dataset=='official_208':
    keep=np.array([i not in ev['clean_exclusions'] for i in range(len(rows))]);clean,cy=compute_detailed_metrics(gold[keep],mat[keep],th,src[keep],cae.SCOPE_102_106);clean.update(clean['confusion']);scores['public_test_clean']=clean
   if dataset=='pub_validation':
    import re
    mask=np.array([bool(re.search(r'под\s+ключ|капитальн\w*\s+ремонт|комплексн\w*\s+ремонт',r['description'],re.I)) for r in rows])&(gold.sum(1)>=2)
    sub,_=compute_detailed_metrics(gold[mask],mat[mask],th,src[mask],cae.SCOPE_102_106);scores['public_complex_multiservice']=sub
  output[name]={'selection':item,'metrics':scores,'tree_count':model.tree_count_,'model_bytes':Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm').stat().st_size if item['experiment']=='C_syntax_baseline' else (P/'models'/f"{item['experiment']}.cbm").stat().st_size};probstore[name]=probs
  print(name,{k:round(v['micro_f1'],6) for k,v in scores.items()},flush=True)
 (P/'final_metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n');pd.DataFrame(predrecords).to_csv(P/'final_predictions.csv',index=False)
 (P/'changed_examples.json').write_text(json.dumps(comparisons,ensure_ascii=False,indent=2)+'\n')
 with (P/'cache/final_probabilities.pkl').open('wb') as f:pickle.dump(probstore,f)
 # Attribution only after finalists are frozen/scored; never fed back into search.
 original=ev['frames']['official_208'];df,cols,cats=transform(original,manifest['features'],manifest['categorical'],{})
 base=CatBoostClassifier();base.load_model('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')
 pool=Pool(df[cols],cat_features=cats);shap=base.get_feature_importance(pool,type='ShapValues')[:,:-1]
 fp=(df.pred_C_05.to_numpy()==1)&(df.y_true.to_numpy()==0);tp=(df.pred_C_05.to_numpy()==1)&(df.y_true.to_numpy()==1)
 stats=[{'feature':c,'mean_shap_on_fp':float(shap[fp,j].mean()),'mean_absolute_shap_on_fp':float(np.abs(shap[fp,j]).mean()),'mean_shap_on_tp':float(shap[tp,j].mean())} for j,c in enumerate(cols)]
 (P/'public_test_shap.json').write_text(json.dumps(sorted(stats,key=lambda x:-x['mean_absolute_shap_on_fp']),ensure_ascii=False,indent=2)+'\n')
 print('Final scoring complete. No new optimization after viewing these metrics.',flush=True)
if __name__=='__main__':main()
