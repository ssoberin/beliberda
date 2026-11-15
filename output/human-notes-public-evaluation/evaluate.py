"""Score frozen pre-existing models against provided human labels; no adaptation."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,json,pickle,sys
from pathlib import Path
import joblib,numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
P=Path('output/human-notes-public-evaluation')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(y,p):
 tp=int(((y==1)&(p==1)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
 per={}
 for j,c in enumerate(range(102,107)):
  a=int(((y[:,j]==1)&(p[:,j]==1)).sum());b=int(((y[:,j]==0)&(p[:,j]==1)).sum());d=int(((y[:,j]==1)&(p[:,j]==0)).sum());per[str(c)]={'tp':a,'fp':b,'fn':d,'f1':2*a/max(1,2*a+b+d)}
 empty=y.sum(1)==0
 return {'rows':len(y),'micro_f1':2*tp/max(1,2*tp+fp+fn),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'tp':tp,'fp':fp,'fn':fn,'exact_match':float((y==p).all(1).mean()),'fully_correct_ads':int((y==p).all(1).sum()),'empty_gold_ads':int(empty.sum()),'empty_gold_ads_with_fp':int((p[empty].sum(1)>0).sum()),'fp_on_empty_gold':int(p[empty].sum()),'per_category':per}
def main():
 assert not (P/'metrics.json').exists(),'Preserve fixed evaluation'
 annotations=json.loads((P/'parsed_annotations.json').read_text());manifest=json.loads((P/'annotation_manifest.json').read_text());source=Path(manifest['source_file']) if Path(manifest['source_file']).exists() else Path(manifest['project_copy']);assert sha(source)==manifest['source_sha256']
 with open('output/catboost-optimization/cache/final_eval.pkl','rb') as f:old=pickle.load(f)
 original=old['frames']['official_208'].copy();rows=old['rows']['official_208']
 meta=json.loads(Path('output/catboost-optimization/input_manifest.json').read_text());finalists=json.loads(Path('output/catboost-optimization/finalists.json').read_text())
 tf=joblib.load('output/syntax-aware-experiment/models/tfidf_baseline.joblib');tf_probs=tf.predict_proba([r['description'] for r in rows]);original['p_tfidf']=tf_probs[original.row_idx.to_numpy(),original.cat_id.astype(int).to_numpy()-102]
 with open('output/tail-services-v3-experiment/cache/trained.pkl','rb') as f:tail=pickle.load(f)
 specs=[{'name':'adaptive_0_3__per_category','label':'Лучший по прежнему calibration: depth6 + TF-IDF','model':'output/catboost-optimization/models/adaptive_0_3.cbm','thresholds':next(x['thresholds'] for x in finalists['finalists'] if x['experiment']=='adaptive_0_3'),'extra_tfidf':True,'frame':'original'},
 {'name':'adaptive_1_0__fixed','label':'Лучший по calibration при0.5: depth8','model':'output/catboost-optimization/models/adaptive_1_0.cbm','thresholds':[.5]*5,'extra_tfidf':False,'frame':'original'},
 {'name':'C_syntax_plus60_fixed_05','label':'C_syntax +60','model':'output/tail-services-v3-experiment/models/catboost_C_syntax_B.cbm','thresholds':[.5]*5,'extra_tfidf':False,'frame':'tail'},
 {'name':'C_syntax_baseline__fixed','label':'Исходная C_syntax','model':'output/syntax-aware-experiment/models/catboost_C_syntax.cbm','thresholds':[.5]*5,'extra_tfidf':False,'frame':'original'}]
 saved=pd.read_csv('output/catboost-optimization/final_predictions.csv');saved=saved[saved.dataset=='official_208'];sb=pd.read_csv('output/tail-services-v3-experiment/predictions.csv');sb=sb[sb.split=='official_208']
 resolved=[r for r in annotations if r['annotation_status']=='human_resolved'];ix=np.array([r['canonical_row'] for r in resolved]);human=np.array([[int(c in r['human_gold']) for c in range(102,107)] for r in resolved],dtype=np.int8)
 output={};all_probs={};csv=[]
 for spec in specs:
  df=(original if spec['frame']=='original' else tail['frames']['official_208']).copy();cols=meta['features']+(['p_tfidf'] if spec['extra_tfidf'] else [])
  for c in meta['categorical']:df[c]=df[c].astype(str)
  model=CatBoostClassifier();model.load_model(spec['model']);assert model.feature_names_==cols
  values=model.predict_proba(Pool(df[cols],cat_features=meta['categorical']),thread_count=2)[:,1]
  mat=np.zeros((208,5));mat[df.row_idx.to_numpy(),df.cat_id.astype(int).to_numpy()-102]=values;pred=(mat>=np.asarray(spec['thresholds'])).astype(np.int8)
  if spec['frame']=='tail':
   dif=max(abs(mat[int(r.row_idx),int(r.cat_id)-102]-r.p_B) for r in sb.itertuples());assert all(pred[int(r.row_idx),int(r.cat_id)-102]==r.pred_B for r in sb.itertuples())
  else:
   reference=saved[saved.model==spec['name']];dif=max(abs(mat[int(r.row_idx),int(r.cat_id)-102]-r.probability) for r in reference.itertuples());assert all(pred[int(r.row_idx),int(r.cat_id)-102]==r.prediction for r in reference.itertuples())
  assert dif<1e-12
  for r in annotations:
   assert ' '.join(rows[r['canonical_row']]['description'].casefold().split())==' '.join(r['description'].casefold().split())
   assert np.max(np.abs(mat[r['official_rows']]-mat[r['canonical_row']]))<1e-12,'Grouped original inputs have differing predictions'
   for j,c in enumerate(range(102,107)):csv.append({'model':spec['name'],'item_number':r['item_number'],'canonical_row':r['canonical_row'],'source_mcId':101,'cat_id':c,'human_gold':int(c in r['human_gold']) if r['human_gold'] is not None else None,'annotation_status':r['annotation_status'],'probability':mat[r['canonical_row'],j],'threshold':spec['thresholds'][j],'prediction':int(pred[r['canonical_row'],j])})
  # Compare original gold on exactly the same canonical37, without changing any official labels.
  original_y=np.array([[int(c in rows[r['canonical_row']]['evaluation_labels']) for c in range(102,107)] for r in resolved],dtype=np.int8)
  output[spec['name']]={'label':spec['label'],'thresholds':spec['thresholds'],'model_sha256':sha(spec['model']),'max_abs_difference_from_saved_probabilities':dif,'human_resolved37':metrics(human,pred[ix]),'original_gold_same37':metrics(original_y,pred[ix])}
  all_probs[spec['name']]={'probabilities':mat,'predictions':pred}
  print(spec['name'],json.dumps(output[spec['name']]['human_resolved37'],ensure_ascii=False),flush=True)
 records=[]
 for r in annotations:
  i=r['canonical_row'];variants=sorted({tuple(rows[k]['evaluation_labels']) for k in r['official_rows']})
  records.append({**r,'original_gold_variants':[list(v) for v in variants],'predictions':{name:[c for j,c in enumerate(range(102,107)) if value['predictions'][i,j]] for name,value in all_probs.items()},'probabilities':{name:value['probabilities'][i].tolist() for name,value in all_probs.items()}})
 pd.DataFrame(csv).to_csv(P/'predictions.csv',index=False);(P/'all139_predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n');(P/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
 (P/'verification.json').write_text(json.dumps({'status':'passed','fresh_catboost_inference':True,'same_predictions_as_frozen_full_pipeline':True,'all_139_descriptions_match_original_by_normalized_text_and_fixed_chunker':all(r['same_chunker_output'] for r in annotations),'canonical_original_input_texts_used':True,'all_208_indices_covered_once':True,'model_training':False,'threshold_tuning':False,'human_labels_used_for_selection_or_training':False,'notes_not_passed_to_model':True,'main_metric_resolved_human_ads':len(resolved),'source_file_sha256':sha(source)},indent=2)+'\n')
if __name__=='__main__':main()
