"""Independent metrics recount and paired uncertainty; never train/select a model."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
P=Path('output/human-notes-public-evaluation')
def counts(y,p):
 return np.array([((y==1)&(p==1)).sum(1),((y==0)&(p==1)).sum(1),((y==1)&(p==0)).sum(1)]).T
def main():
 a=json.loads((P/'parsed_annotations.json').read_text());f=json.loads((P/'annotation_manifest.json').read_text());m=json.loads((P/'metrics.json').read_text());pure=json.loads((P/'tfidf_only_metrics.json').read_text());pr=json.loads((P/'tfidf_only_predictions.json').read_text())
 assert hashlib.sha256(Path(f['project_copy']).read_bytes()).hexdigest()==f['source_sha256']
 assert len(a)==139 and sum(r['annotation_status']=='human_resolved' for r in a)==37
 assert sorted(i for r in a for i in r['official_rows'])==list(range(208));assert [r['item_number'] for r in a if r['annotation_status']=='human_ambiguous']==[19]
 rows=[r for r in a if r['annotation_status']=='human_resolved'];ids=[r['item_number'] for r in rows];y=np.array([[int(c in r['human_gold']) for c in range(102,107)] for r in rows])
 frame=pd.read_csv(P/'predictions.csv');allpred={}
 for name,data in m.items():
  sub=frame[(frame.model==name)&frame.item_number.isin(ids)];p=sub.pivot(index='item_number',columns='cat_id',values='prediction').loc[ids,list(range(102,107))].to_numpy();r=data['human_resolved37'];tp,fp,fn=counts(y,p).sum(0)
  assert (int(tp),int(fp),int(fn))==(r['tp'],r['fp'],r['fn']);assert abs(2*tp/(2*tp+fp+fn)-r['micro_f1'])<1e-12;allpred[name]=p
 p=np.array([[int(c in pr[i-1]['prediction']) for c in range(102,107)] for i in ids]);tp,fp,fn=counts(y,p).sum(0);assert (int(tp),int(fp),int(fn))==(67,9,2);assert abs(2*tp/(2*tp+fp+fn)-pure['human_resolved37']['micro_f1'])<1e-12
 out={};rng=np.random.default_rng(42);ctf=counts(y,p)
 for name,cp in allpred.items():
  cc=counts(y,cp);delta=[]
  for _ in range(5000):
   ix=rng.integers(0,len(y),size=len(y));t1,f1,n1=ctf[ix].sum(0);t2,f2,n2=cc[ix].sum(0);delta.append(2*t1/max(1,2*t1+f1+n1)-2*t2/max(1,2*t2+f2+n2))
  out[name]={'delta_micro_f1_tfidf_minus_model':pure['human_resolved37']['micro_f1']-m[name]['human_resolved37']['micro_f1'],'ci95_low':float(np.percentile(delta,2.5)),'ci95_high':float(np.percentile(delta,97.5)),'n_boot':5000,'rows':37,'method':'paired bootstrap by unique ad; small nonrandom already-known public subset'}
 (P/'paired_tfidf_comparisons.json').write_text(json.dumps(out,indent=2)+'\n')
 v={'status':'passed','human_rows':37,'all_inputs':139,'original_rows_covered':208,'model_and_threshold_changes':False,'metrics_independently_recounted':True,'tfidf_compared_on_identical_labels_and_inputs':True,'uncertainty_comparisons':out}
 (P/'comparison_verification.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v,indent=2))
if __name__=='__main__':main()
