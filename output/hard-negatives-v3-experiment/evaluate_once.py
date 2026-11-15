"""One final A/B evaluation. No threshold/model/data selection by test scores."""
import json,pickle,re,sys
from pathlib import Path
import numpy as np,pandas as pd
from catboost import CatBoostClassifier,Pool
sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts import qwen3_category_aware_experiment as cae
from avito_compare.scripts import run_syntax_experiment as se
from avito_compare.scripts.syntax_features import SYNTAX_CAT_FEATURES
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics
P=Path('output/hard-negatives-v3-experiment')
def main():
 if (P/'metrics.json').exists():raise RuntimeError('Final evaluation already exists; no iterative scoring')
 validation=json.loads((P/'training_validation.json').read_text());assert validation['status']=='passed' and validation['full_head_parameters_match_baseline']
 with (P/'cache/trained.pkl').open('rb') as f:s=pickle.load(f)
 model=CatBoostClassifier();model.load_model(str(P/'models/catboost_C_syntax_B.cbm'))
 saved=pd.read_csv('output/syntax-aware-experiment/predictions.csv');official=pd.read_csv('output/syntax-aware-experiment/public_tests_predictions.csv')
 results={};predrows=[];examplelines=[]
 for name in ['pub_validation','v3_test','official_208']:
  rows=s['rows'][name];frame=s['frames'][name];y,src=cae._rows_xy_for_ids(rows,cae.SCOPE_102_106,use_eval=True)
  a=(official if name=='official_208' else saved);a=a[a['split']==name].sort_values(['row_idx','cat_id'])
  pa=np.zeros((len(rows),5));ya=np.zeros_like(pa,dtype=np.int8)
  for row in a.itertuples():
   j=cae.SCOPE_102_106.index(row.cat_id);pa[row.row_idx,j]=row.p_C;ya[row.row_idx,j]=row.pred_C_05
   assert y[row.row_idx,j]==row.y_true
  pb=se.probs_to_eval_matrix(frame,model.predict_proba(Pool(frame[s['cols']],cat_features=SYNTAX_CAT_FEATURES))[:,1],len(rows),src,cae.SCOPE_102_106)
  ma,_=compute_detailed_metrics(y,pa,np.full(5,.5),src,cae.SCOPE_102_106)
  mb,yb=compute_detailed_metrics(y,pb,np.full(5,.5),src,cae.SCOPE_102_106)
  for key,pred,m in [('A',ya,ma),('B',yb,mb)]:
   m.update(m['confusion'])
   empty=y.sum(1)==0;nonempty=~empty
   m['fp_on_empty_gold']=int(pred[empty].sum());m['empty_gold_ads_with_fp']=int((pred[empty].sum(1)>0).sum())
   m['fn_on_nonempty_gold']=int(((y[nonempty]==1)&(pred[nonempty]==0)).sum())
  transitions=se.error_transition_counts(y,ya,yb,src,cae.SCOPE_102_106)
  newly_empty=(ya.sum(1)>0)&(yb.sum(1)==0);transitions['new_empty_ads']=int(newly_empty.sum());transitions['new_erroneous_empty_on_nonempty_gold']=int((newly_empty&(y.sum(1)>0)).sum())
  results[name]={'A':ma,'B':mb,'transitions':transitions,'paired_bootstrap_f1_delta':se.bootstrap_f1_delta(y,ya,yb,src,cae.SCOPE_102_106,n_boot=2000)}
  complex_mask=np.array([bool(re.search(r'под\s+ключ|капитальн\w*\s+ремонт|комплексн\w*\s+ремонт',r['description'],re.I)) for r in rows])&(y.sum(1)>=2)
  if name=='pub_validation' and complex_mask.any():
   results[name]['complex_repair_multi_service_slice']={'rows':int(complex_mask.sum()),'A':compute_detailed_metrics(y[complex_mask],pa[complex_mask],np.full(5,.5),src[complex_mask],cae.SCOPE_102_106)[0],'B':compute_detailed_metrics(y[complex_mask],pb[complex_mask],np.full(5,.5),src[complex_mask],cae.SCOPE_102_106)[0],'note':'Descriptive report slice: repair wording plus >=2 gold services; no training/postprocessing rule.'}
  for i,row in enumerate(rows):
   ea=int((ya[i]!=y[i]).sum());eb=int((yb[i]!=y[i]).sum())
   for j,cid in enumerate(cae.SCOPE_102_106):
    if src[i]==cid:continue
    predrows.append(dict(split=name,row_idx=i,source_mcId=int(src[i]),cat_id=cid,gold=int(y[i,j]),p_A=float(pa[i,j]),pred_A=int(ya[i,j]),p_B=float(pb[i,j]),pred_B=int(yb[i,j])))
   if ea!=eb:examplelines.append({'split':name,'row_idx':i,'description':row['description'],'gold':[c for j,c in enumerate(cae.SCOPE_102_106) if y[i,j]],'A':[c for j,c in enumerate(cae.SCOPE_102_106) if ya[i,j]],'B':[c for j,c in enumerate(cae.SCOPE_102_106) if yb[i,j]],'errors_A':ea,'errors_B':eb})
  print(name,'A',ma['micro_f1'],'B',mb['micro_f1'],'FP',ma['fp'],mb['fp'],'FN',ma['fn'],mb['fn'],flush=True)
 pd.DataFrame(predrows).to_csv(P/'predictions.csv',index=False)
 (P/'metrics.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')
 (P/'error_examples.json').write_text(json.dumps(examplelines,ensure_ascii=False,indent=2)+'\n')
 lines=['# Добавление 200 hard negatives к V3: A/B','',
 'A — сохранённый C_syntax, B — тот же pipeline на исходном train6319+200. Encoder frozen, fixed chunker, Top3 retrieval, те же признаки Natasha/Slovnet и CatBoost, grouped OOF полностью пересчитан, порог0.5. Calibration1328толькоearlystopping. public_synthetic_300неоценивался и не участвовал в обучении.','',
 '| Набор | F1A | F1B | PA→B | RA→B | TPA→B | FPA→B | FNA→B | ExactA→B |','|---|---:|---:|---|---|---|---|---|---|']
 for n,r in results.items():
  a=r['A'];b=r['B'];lines.append(f"| {n} | {a['micro_f1']:.6f} | {b['micro_f1']:.6f} | {a['micro_precision']:.4f}→{b['micro_precision']:.4f} | {a['micro_recall']:.4f}→{b['micro_recall']:.4f} | {a['tp']}→{b['tp']} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} | {a['exact_match']:.4f}→{b['exact_match']:.4f} |")
 lines+=['','## По категориям','', '| Набор | Категория | F1A→B | FPA→B | FNA→B |','|---|---|---|---|---|']
 for n,r in results.items():
  for cid,a in r['A']['per_category'].items():
   b=r['B']['per_category'][cid];lines.append(f"| {n} | {cid} | {a['f1']:.4f}→{b['f1']:.4f} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} |")
 lines+=['','## Ошибки и интервалы','']
 for n,r in results.items():lines += [f"- {n}: {json.dumps(r['transitions'],ensure_ascii=False)}; paired bootstrap: {json.dumps(r['paired_bootstrap_f1_delta'])}; FP на пустомgold {r['A']['fp_on_empty_gold']}→{r['B']['fp_on_empty_gold']}."]
 lines+=['','## Примеры исправлений и регрессий','']
 for typ,condition in [('Исправления',lambda e:e['errors_B']<e['errors_A']),('Регрессии',lambda e:e['errors_B']>e['errors_A'])]:
  lines.append('### '+typ)
  selected=[e for e in examplelines if e['split']!='official_208' and condition(e)][:5]
  for e in selected:lines += [f"- {e['split']}#{e['row_idx']}: gold={e['gold']},A={e['A']},B={e['B']}. {e['description'][:650].replace(chr(10),' ')}"]
 lines+=['','## Ограничения','',
 'Official208неимеетsource_mcId (принято101), содержит противоречивые дубликаты и пересечения с train. Использован исключительно диагностически. Bootstrapпообъявлениям может недооценивать зависимость шаблонов; интервалы не доказывают перенос на неизвестный поток.','',
 'Новые объявления основаны на200разныхавторскихбрифах и общих вспомогательных коммерческих условиях. Повторяемость этих условий явно отражена вdataset_audit.md: это ограничение стилистического разнообразия, а не независимая человеческая разметка. Уверенность в пустомgold основана на границах реального предложения, не на ошибках baseline. Проверки многократно использовались ранее; эта серия оценена один раз после фиксации train и config. Основная модель не заменена.']
 (P/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
