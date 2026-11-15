"""Calibration-only CatBoost search. Never loads final_eval.pkl or public test labels."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import copy,hashlib,json,pickle,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts.run_syntax_experiment import CATBOOST_PARAMS
P=Path('output/catboost-optimization')
def transform(df,cols,cats,spec):
 df=df.copy();cols=[c for c in cols if c not in spec.get('drop',[])];cats=[c for c in cats if c in cols]
 if spec.get('tfidf'):cols=cols+['p_tfidf']
 if spec.get('cat_mode')=='numeric':df['cat_id']=df.cat_id.astype(int);cats.remove('cat_id')
 if spec.get('cat_mode')=='onehot':
  for cid in range(102,107):df[f'cat_{cid}']=(df.cat_id.astype(int)==cid).astype(int)
  cols=[c for c in cols if c!='cat_id']+[f'cat_{cid}' for cid in range(102,107)];cats.remove('cat_id')
 for c in cats:df[c]=df[c].astype(str)
 assert not df[cols].isna().any().any()
 return df,cols,cats

def counts(y,pred):
 tp=int(((y==1)&pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
 return {'f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn}
def cal_metrics(df,p,ths):
 y=df.y_true.to_numpy();cid=df.cat_id.astype(int).to_numpy()-102;pred=p>=np.asarray(ths)[cid];domains=df.domain.to_numpy()
 out={d:counts(y[domains==d],pred[domains==d]) for d in ['public','v3']}
 out['all']=counts(y,pred);out['score']=.65*out['public']['f1']+.35*out['v3']['f1']
 out['exact_match']=float(pd.Series(y==pred).groupby(df.row_idx.to_numpy()).all().mean())
 return out

def thresholds(df,p):
 grid=np.round(np.arange(.25,.751,.025),3)
 global_t=max(grid,key=lambda t:(cal_metrics(df,p,[t]*5)['score'],-abs(t-.5)))
 cats=df.cat_id.astype(int).to_numpy();y=df.y_true.to_numpy();dom=df.domain.to_numpy();per=[]
 for cid in range(102,107):
  def objective(t):
   scores=[]
   for d in ['public','v3']:
    mask=(cats==cid)&(dom==d);scores.append(counts(y[mask],p[mask]>=t)['f1'])
   return .65*scores[0]+.35*scores[1],-abs(t-.5)
  per.append(float(max(grid,key=objective)))
 return {'fixed':[.5]*5,'global':[float(global_t)]*5,'per_category':per}

def main():
 with (P/'cache/selection.pkl').open('rb') as f:s=pickle.load(f)
 tr=s['train'];cal=s['cal'];params0={**CATBOOST_PARAMS,'verbose':False,'allow_writing_files':False}
 plan=[{'name':'baseline_refit'},*({'name':f'depth_{d}','params':{'depth':d}} for d in [5,6,8]),
  *({'name':f'l2_{v}','params':{'l2_leaf_reg':v}} for v in [1,3,10,20,50]),
  *({'name':f'lr_{v}','params':{'learning_rate':v}} for v in [.025,.035,.075,.1]),
  *({'name':f'random_{v}','params':{'random_strength':v}} for v in [0,2,5,10]),
  *({'name':f'bagging_{v}','params':{'bootstrap_type':'Bayesian','bagging_temperature':v}} for v in [0,.5,2]),
  *({'name':'drop_'+c,'drop':[c]} for c in ['syn_global_turnkey_only','syn_global_partial_refusal','syn_global_outside_top3']),
  {'name':'drop_all_global','drop':['syn_global_turnkey_only','syn_global_partial_refusal','syn_global_outside_top3']},
  *({'name':f'v3_weight_{v}','v3_weight':v} for v in [.25,.5,.75,1.5,2]),
  {'name':'cat_numeric','cat_mode':'numeric'},{'name':'cat_onehot','cat_mode':'onehot'},
  {'name':'tfidf','tfidf':True},{'name':'tfidf_depth5','tfidf':True,'params':{'depth':5}},
  {'name':'tfidf_no_turnkey','tfidf':True,'drop':['syn_global_turnkey_only']}]
 results=[]
 if (P/'experiments.json').exists():results=json.loads((P/'experiments.json').read_text())
 completed={r['spec']['name'] for r in results}
 def run(spec):
  name=spec['name']
  if name in completed:return
  signature=lambda spec:json.dumps({'params':{**params0,**spec.get('params',{})},'drop':sorted(spec.get('drop',[])),'v3_weight':spec.get('v3_weight',1),'cat_mode':spec.get('cat_mode','categorical'),'tfidf':spec.get('tfidf',False)},sort_keys=True)
  if any(signature(spec)==signature(r['spec']) for r in results):return
  df,cols,cats=transform(tr,s['cols'],s['categorical'],spec);cf,_,_=transform(cal,s['cols'],s['categorical'],spec)
  w=np.where(tr.domain=='v3',spec.get('v3_weight',1),1).astype(float);w/=w.mean()
  params={**params0,**spec.get('params',{})};start=time.monotonic()
  model=CatBoostClassifier(**params);model.fit(Pool(df[cols],df.y_true,cat_features=cats,weight=w),eval_set=Pool(cf[cols],cf.y_true,cat_features=cats),use_best_model=True)
  model.save_model(str(P/'models'/f'{name}.cbm'));p=model.predict_proba(Pool(cf[cols],cat_features=cats))[:,1]
  if name=='baseline_refit':
   original=CatBoostClassifier();original.load_model('output/syntax-aware-experiment/models/catboost_C_syntax.cbm')
   ref=original.predict_proba(Pool(cf[cols],cat_features=cats))[:,1]
   diff=float(np.max(np.abs(ref-p)))
   (P/'baseline_reproduction.json').write_text(json.dumps({'max_abs_calibration_probability_difference':diff,'old_trees':original.tree_count_,'reproduced_trees':model.tree_count_},indent=2))
   assert diff<1e-8,('Baseline training reconstruction mismatch',diff)
  modes=thresholds(cf,p)
  r={'spec':spec,'parameters':params,'features':cols,'categorical':cats,'best_iteration':model.get_best_iteration(),'fit_seconds':time.monotonic()-start,'calibration':{mode:{'thresholds':ths,'metrics':cal_metrics(cf,p,ths)} for mode,ths in modes.items()}}
  results.append(r);completed.add(name);np.save(P/'cache'/f'cal_{name}.npy',p)
  (P/'experiments.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
  print(len(results),name,'best_iteration',r['best_iteration'],'cal public/v3 fixed',r['calibration']['fixed']['metrics']['public']['f1'],r['calibration']['fixed']['metrics']['v3']['f1'],flush=True)
 (P/'selection_protocol.json').write_text(json.dumps({'selection_inputs':['train6319','calibration1328'],'no_test_inputs':True,'score':'.65*public_calibration_F1+.35*v3_calibration_F1','threshold_grid':[.25,.75,.025],'stages':'34 initial parameter/data/ablation specs, then 12 adaptive calibration-only combinations; no test scoring before finalists freeze.','finalist_slots':['balanced calibration score','public calibration champion with v3 floor','low-FP candidate without added TF-IDF and with recall floors','best fixed0.5 candidate; distinct trained experiments per slot'],'floors':'public calibration F1>=baseline-.003; v3 calibration F1>=baseline-.005 where feasible','no_auto_promotion':True},indent=2))
 for spec in plan:run(spec)
 initial=[r for r in results if not r['spec']['name'].startswith('adaptive_')]
 baseline=initial[0]['calibration']['fixed']['metrics']
 # Adapt only from calibration outcomes: combine strong configurations and test depth/l2/weights nearby.
 def best_score(r):return max(x['metrics']['score'] for x in r['calibration'].values())
 leaders=sorted(initial,key=best_score,reverse=True)[:3]
 adaptive=[]
 for i,r in enumerate(leaders):
  for j,patch in enumerate([{'params':{'l2_leaf_reg':15}}, {'params':{'random_strength':2}}, {'v3_weight':.5}, {'params':{'depth':6,'learning_rate':.035,'l2_leaf_reg':15}}]):
   spec=copy.deepcopy(r['spec']);spec['name']=f'adaptive_{i}_{j}';spec['params']={**spec.get('params',{}),**patch.get('params',{})}
   for k,v in patch.items():
    if k!='params':spec[k]=v
   adaptive.append(spec)
 (P/'adaptive_plan.json').write_text(json.dumps({'leaders_selected_on_calibration':[r['spec']['name'] for r in leaders],'specs':adaptive},indent=2))
 for spec in adaptive:run(spec)
 for r in results:
  if 'effective_parameters' not in r:
   trained=CatBoostClassifier();trained.load_model(str(P/'models'/f"{r['spec']['name']}.cbm"));r['effective_parameters']=trained.get_all_params()
 (P/'experiments.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 candidates=[]
 for r in results:
  for mode,score in r['calibration'].items():
   m=score['metrics'];candidates.append({'experiment':r['spec']['name'],'mode':mode,'thresholds':score['thresholds'],'calibration':m,'spec':r['spec']})
 feasible=[r for r in candidates if r['calibration']['public']['f1']>=baseline['public']['f1']-.003 and r['calibration']['v3']['f1']>=baseline['v3']['f1']-.005]
 assert feasible
 chosen=[]
 def choose(slot,pool,key):
  pool=[r for r in pool if not any(x['experiment']==r['experiment'] for x in chosen)]
  if not pool:return
  row=max(pool,key=key);identity=(row['experiment'],row['mode'])
  if not any((x['experiment'],x['mode'])==identity for x in chosen):chosen.append({**row,'selection_slot':slot})
 choose('balanced',feasible,lambda r:r['calibration']['score'])
 choose('public',feasible,lambda r:(r['calibration']['public']['f1'],r['calibration']['v3']['f1']))
 precision=[r for r in feasible if r['calibration']['public']['recall']>=baseline['public']['recall']-.005 and r['calibration']['v3']['recall']>=baseline['v3']['recall']-.01]
 precision_light=[r for r in precision if not r['spec'].get('tfidf')]
 choose('low_fp_no_extra_tfidf',precision_light or precision or feasible,lambda r:(-r['calibration']['public']['fp']-r['calibration']['v3']['fp'],r['calibration']['score']))
 fixed=[r for r in feasible if r['mode']=='fixed']
 if fixed:choose('best_fixed_0.5',fixed,lambda r:r['calibration']['score'])
 payload={'frozen_before_test_scoring':True,'experiments_count':len(results),'baseline_calibration':baseline,'finalists':chosen,'experiment_results_sha256':hashlib.sha256((P/'experiments.json').read_bytes()).hexdigest(),'model_sha256':{r['experiment']:hashlib.sha256((P/'models'/f"{r['experiment']}.cbm").read_bytes()).hexdigest() for r in chosen}}
 (P/'finalists.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
 table=[]
 for r in results:
  for mode,v in r['calibration'].items():
   m=v['metrics'];table.append({'experiment':r['spec']['name'],'mode':mode,'public_f1':m['public']['f1'],'v3_f1':m['v3']['f1'],'weighted_score':m['score'],'public_fp':m['public']['fp'],'public_fn':m['public']['fn'],'v3_fp':m['v3']['fp'],'v3_fn':m['v3']['fn'],'fit_seconds':r['fit_seconds']})
 pd.DataFrame(table).to_csv(P/'calibration_metrics.csv',index=False)
 print('FROZEN FINALISTS',[(r['experiment'],r['mode']) for r in chosen],flush=True)
if __name__=='__main__':main()
