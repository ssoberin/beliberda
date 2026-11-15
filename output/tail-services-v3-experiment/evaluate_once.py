"""Single frozen A/B evaluation. No fits, threshold selection, or data edits."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import collections,hashlib,json,pickle,re,sys
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts import run_syntax_experiment as se
from avito_compare.scripts.syntax_features import SYNTAX_CAT_FEATURES
P=Path('output/tail-services-v3-experiment');IDS=cae.SCOPE_102_106
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
 tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
 per={}
 for j,c in enumerate(IDS):
  a=int(((y[:,j]==1)&(p[:,j]==1)).sum());b=int(((y[:,j]==0)&(p[:,j]==1)).sum());d=int(((y[:,j]==1)&(p[:,j]==0)).sum())
  per[str(c)]={'f1':2*a/max(1,2*a+b+d),'precision':a/max(1,a+b),'recall':a/max(1,a+d),'tp':a,'fp':b,'fn':d,'support':int(y[:,j].sum())}
 empty=y.sum(1)==0
 return {'micro_f1':2*tp/max(1,2*tp+fp+fn),'macro_f1':float(np.mean([r['f1'] for r in per.values()])),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn,'exact_match':float((y==p).all(1).mean()),'wrong_ads':int((y!=p).any(1).sum()),'fp_on_empty_gold':int(p[empty].sum()),'empty_gold_ads_with_fp':int((p[empty].sum(1)>0).sum()),'fn_on_nonempty_gold':fn,'empty_gold_ads':int(empty.sum()),'per_category':per}
def transitions(y,a,b):
 ea=(a!=y).sum(1);eb=(b!=y).sum(1)
 return {**se.error_transition_counts(y,a,b,np.full(len(y),101),IDS),'ads_fully_fixed':int(((ea>0)&(eb==0)).sum()),'ads_newly_wrong':int(((ea==0)&(eb>0)).sum()),'ads_with_fewer_errors':int((eb<ea).sum()),'ads_with_more_errors':int((eb>ea).sum()),'new_erroneous_empty_on_nonempty_gold':int(((a.sum(1)>0)&(b.sum(1)==0)&(y.sum(1)>0)).sum())}
def clustered_ci(y,a,b,texts,n_boot=4000):
 groups=collections.defaultdict(list)
 for i,t in enumerate(texts):groups[' '.join(t.casefold().split())].append(i)
 stats=[]
 for ix in groups.values():
  z=[]
  for p in [a,b]:z.extend([((y[ix]==1)&(p[ix]==1)).sum(),((y[ix]==0)&(p[ix]==1)).sum(),((y[ix]==1)&(p[ix]==0)).sum()])
  stats.append(z)
 stats=np.asarray(stats,dtype=float);rng=np.random.default_rng(42);deltas=[]
 for _ in range(n_boot):
  s=stats[rng.integers(0,len(stats),size=len(stats))].sum(0)
  deltas.append(2*s[3]/max(1,2*s[3]+s[4]+s[5])-2*s[0]/max(1,2*s[0]+s[1]+s[2]))
 return {'method':'paired bootstrap by normalized exact-description groups','groups':len(stats),'n_boot':n_boot,'ci95_low':float(np.percentile(deltas,2.5)),'ci95_high':float(np.percentile(deltas,97.5)),'fraction_positive':float(np.mean(np.array(deltas)>0))}
def score(y,pa,pb,rows):
 a=(pa>=.5).astype(np.int8);b=(pb>=.5).astype(np.int8)
 out={'rows':len(y),'A':metrics(y,a),'B':metrics(y,b),'transitions':transitions(y,a,b),'bootstrap_delta_micro_f1':clustered_ci(y,a,b,[r['description'] for r in rows])}
 out['delta']={k:out['B'][k]-out['A'][k] for k in ['micro_f1','macro_f1','precision','recall','fp','fn','tp','exact_match']}
 return out,a,b
def main():
 assert not (P/'metrics.json').exists(),'Final evaluation already exists; no iterative scoring'
 freeze=json.loads((P/'frozen_inputs.json').read_text())
 for p,h in freeze['files'].items():assert sha(p)==h,(p,'changed after freeze')
 config=json.loads((P/'config.json').read_text());validation=json.loads((P/'training_validation.json').read_text());assert validation['status']=='passed' and validation['full_head_parameters_match_baseline']
 for p,h in config['source_sha256'].items():assert sha(p)==h
 baseline_path=Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm');assert sha(baseline_path)==config['baseline_model_sha256']
 with (P/'cache/trained.pkl').open('rb') as f:s=pickle.load(f)
 base=CatBoostClassifier();base.load_model(str(baseline_path));aug=CatBoostClassifier();aug.load_model(str(P/'models/catboost_C_syntax_B.cbm'))
 assert base.feature_names_==aug.feature_names_==s['cols']
 saved=pd.read_csv('output/catboost-optimization/final_predictions.csv');saved=saved[saved.model=='C_syntax_baseline__fixed']
 results={};predrows=[];examples=[];matrices={}
 for name in ['pub_validation','v3_test','official_208','tail_holdout20']:
  rows=s['rows'][name];frame=s['frames'][name];y,src=cae._rows_xy_for_ids(rows,IDS,use_eval=True)
  pb=se.probs_to_eval_matrix(frame,aug.predict_proba(Pool(frame[s['cols']],cat_features=SYNTAX_CAT_FEATURES),thread_count=2)[:,1],len(rows),src,IDS)
  pa=np.zeros_like(pb)
  if name=='tail_holdout20':
   af=s['baseline_holdout'];pa=se.probs_to_eval_matrix(af,base.predict_proba(Pool(af[s['cols']],cat_features=SYNTAX_CAT_FEATURES),thread_count=2)[:,1],len(rows),src,IDS)
  else:
   for r in saved[saved.dataset==name].itertuples():
    j=IDS.index(int(r.cat_id));assert y[int(r.row_idx),j]==r.gold;pa[int(r.row_idx),j]=r.probability
  res,a,b=score(y,pa,pb,rows);results[name]=res;matrices[name]=(y,pa,pb)
  # Own-source cells are masked by construction in y/probability matrices.
  assert all(src[i] not in IDS or (y[i,IDS.index(src[i])]==0 and pa[i,IDS.index(src[i])]==0 and pb[i,IDS.index(src[i])]==0) for i in range(len(rows)))
  for i,row in enumerate(rows):
   ea=int((a[i]!=y[i]).sum());eb=int((b[i]!=y[i]).sum())
   if (a[i]!=b[i]).any() or name=='tail_holdout20':examples.append({'split':name,'row_idx':i,'item_id':row.get('item_id'),'source_mcId':int(src[i]),'description':row['description'],'gold':[c for j,c in enumerate(IDS) if y[i,j]],'A':[c for j,c in enumerate(IDS) if a[i,j]],'B':[c for j,c in enumerate(IDS) if b[i,j]],'p_A':pa[i].tolist(),'p_B':pb[i].tolist(),'errors_A':ea,'errors_B':eb})
   for j,cid in enumerate(IDS):predrows.append({'split':name,'row_idx':i,'source_mcId':int(src[i]),'cat_id':cid,'active':int(src[i]!=cid),'gold':int(y[i,j]),'p_A':float(pa[i,j]),'pred_A':int(a[i,j]),'p_B':float(pb[i,j]),'pred_B':int(b[i,j])})
  print(name,'F1',res['A']['micro_f1'],'->',res['B']['micro_f1'],'FP',res['A']['fp'],'->',res['B']['fp'],'FN',res['A']['fn'],'->',res['B']['fn'],flush=True)
 # Same pre-existing conflict exclusion for A/B; never relabel official rows.
 official=s['rows']['official_208'];groups=collections.defaultdict(list)
 for i,r in enumerate(official):groups[' '.join(r['description'].casefold().split())].append(i)
 excluded=sorted(i for ix in groups.values() if len({tuple(official[k]['evaluation_labels']) for k in ix})>1 for i in ix)
 assert len(excluded)==56
 keep=np.array([i for i in range(208) if i not in set(excluded)]);y,pa,pb=matrices['official_208']
 results['public_test_clean']=score(y[keep],pa[keep],pb[keep],[official[i] for i in keep])[0];results['public_test_clean']['excluded_rows']=excluded
 # Descriptive existing-test slice, no data selection or tuning.
 name='pub_validation';rows=s['rows'][name];y,pa,pb=matrices[name]
 keep=np.array([i for i,r in enumerate(rows) if y[i].sum()>=2 and re.search(r'под\s+ключ|капитальн\w*\s+ремонт|комплексн\w*\s+ремонт',r['description'],re.I)])
 if len(keep):results['pub_validation_complex_repair']=score(y[keep],pa[keep],pb[keep],[rows[i] for i in keep])[0]
 results['provenance']={'threshold':.5,'data_only_single_candidate':True,'training_validation_passed':True,'baseline_sha256':sha(baseline_path),'candidate_sha256':sha(P/'models/catboost_C_syntax_B.cbm'),'known_baseline_f1':{'pub_validation':.956124,'v3_test':.915916,'official_208':.819767}}
 for name,value in results['provenance']['known_baseline_f1'].items():assert abs(results[name]['A']['micro_f1']-value)<1e-6
 (P/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');pd.DataFrame(predrows).to_csv(P/'predictions.csv',index=False);(P/'error_examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
 print('Final evaluation saved once; baseline unchanged.',flush=True)
if __name__=='__main__':main()
