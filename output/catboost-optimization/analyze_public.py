"""Post-freeze diagnostic analysis; never supplies inputs to optimization."""
import collections,json,pickle,re,sys
from pathlib import Path
import numpy as np,pandas as pd
P=Path('output/catboost-optimization')
def norm(t):return ' '.join(t.casefold().split())
def read(p):return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
def main():
 scores=json.loads((P/'final_metrics.json').read_text());pred=pd.read_csv(P/'final_predictions.csv')
 with (P/'cache/final_eval.pkl').open('rb') as f:ev=pickle.load(f)
 rows=ev['rows']['official_208'];excluded=set(ev['clean_exclusions'])
 train=read('output/ngram-comparison/splits/train.jsonl');seen={norm(r['description']) for r in train}
 overlap={i for i,r in enumerate(rows) if norm(r['description']) in seen}
 base=pred[(pred.model=='C_syntax_baseline__fixed')&(pred.dataset=='official_208')]
 diagnostics={'test208_source':'mcId absent, assumed101 throughout, not inferred from labels','clean_policy':json.loads((P/'public_test_exclusions.json').read_text()),'train_overlap_rows':sorted(overlap),'train_overlap_count':len(overlap),'baseline_subsets':{}}
 for name,indices in [('conflicting_rows',excluded),('nonconflicting_rows',set(range(208))-excluded),('train_overlap',overlap),('no_train_overlap',set(range(208))-overlap)]:
  d=base[base.row_idx.isin(indices)];y=d.gold.to_numpy();yp=d.prediction.to_numpy();tp=int(((y==1)&(yp==1)).sum());fp=int(((y==0)&(yp==1)).sum());fn=int(((y==1)&(yp==0)).sum())
  diagnostics['baseline_subsets'][name]={'rows':len(indices),'tp':tp,'fp':fp,'fn':fn,'f1':2*tp/max(1,2*tp+fp+fn)}
 f=ev['frames']['official_208'];fp=f[(f.pred_C_05==1)&(f.y_true==0)]
 diagnostics['baseline_false_positive_features']={c:{str(k):int(v) for k,v in fp[c].value_counts(dropna=False).items()} for c in ['cat_id','best_chunk_polarity','dominant_subject','service_link_status','syn_affirmative_offer','syn_global_turnkey_only','syn_global_partial_refusal','syn_has_neg_on_action']}
 diagnostics['baseline_fp_sgd_probability']={'median':float(fp.p_sgd.median()),'p90':float(fp.p_sgd.quantile(.9))}
 diagnostics['domain_differences']={}
 for dataset in ['pub_validation','official_208']:
  r=scores['C_syntax_baseline__fixed']['metrics'][dataset];diagnostics['domain_differences'][dataset]={k:r[k] for k in ['rows','micro_f1','micro_precision','micro_recall','fp','fn','empty_gold','fp_on_empty_gold','fpr']}
 (P/'public_test_analysis.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
 changed=json.loads((P/'changed_examples.json').read_text());selected={}
 for model in scores:
  if model=='C_syntax_baseline__fixed':continue
  modelchanges=[r for r in changed if r['dataset']=='official_208' and r['model']==model]
  for direction in [1,-1]:
   ranked=sorted([r for r in modelchanges if (r['errors_before']-r['errors_after'])*direction>0],key=lambda r:(-abs(r['errors_before']-r['errors_after']),r['row_idx']))
   for r in ranked[:4]:selected[(r['row_idx'],r['model'])]=r
 review=list(selected.values());(P/'semantic_review_candidates.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
 for r in review:print('REVIEW',r['model'],r['row_idx'],'gold',r['gold'],'baseline',r['baseline'],'candidate',r['candidate'],'TEXT',r['description'][:1400].replace('\n',' '),flush=True)
 print('DIAGNOSTICS',json.dumps(diagnostics,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
