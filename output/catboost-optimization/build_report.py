"""Build report from completed frozen evaluation; no model selection/training."""
import json
from pathlib import Path
import pandas as pd
P=Path('output/catboost-optimization')
def load(name):return json.loads((P/name).read_text())
def main():
 final=load('final_metrics.json');experiments=load('experiments.json');audit=load('public_test_analysis.json');shap=load('public_test_shap.json');frozen=load('finalists.json')
 base=final['C_syntax_baseline__fixed']['metrics']
 lines=['# Оптимизация CatBoost поверх исходной C_syntax','',
 f"Проведено {len(experiments)}обученийCatBoost на исходныхpublic3398+V3clean2921; новыеhardnegativesнеиспользовались. EncoderQwen3-Embedding-0.6B, fixedchunker, Top3 и полные SGDголовы baseline не менялись. Исходный grouped nested OOF восстановлен отдельно; TF-IDFOOF получен с fit векторизатора только внутри каждого train-фолда. Colabнеиспользовался.",'',
 '## Протокол отбора','',
 'В поиске участвуют только train и calibration1328(public728+V3validation600). Earlystopping — по calibrationLogloss с прежним patience50. Основной score отбора:0.65×publiccalF1+0.35×V3calF1; заранее заданы четыре слота: сбалансированный, лучшийpubliccalсfloorV3, низкийFPприfloorRecall, лучшийTF-IDF-вариант. Пороги0.5/общий/покатегорийные сравнивались только наcalibration. Финалисты и SHA-256весов зафиксированы до финального скоринга. После просмотра publictest никаких новых конфигураций не запускали.','',
 '## Итоговая таблица','',
 '| Модель | pub_validation | public_test_208 | public_test_clean | v3_test |','|---|---:|---:|---:|---:|']
 for name,r in final.items():
  m=r['metrics'];lines.append(f"| {name} | {m['pub_validation']['micro_f1']:.6f} | {m['official_208']['micro_f1']:.6f} | {m['public_test_clean']['micro_f1']:.6f} | {m['v3_test']['micro_f1']:.6f} |")
 lines+=['','Clean=152строки: исключены все56строк13групп с противоречивымgold при одинаковомнормализованномdescription. Это **не** прежние114уникальныхстрокбезпересечений. Здесь сохраняются непротиворечивые повторы и возможные пересечения сtrain. Все модели используют один списокpublic_test_exclusions.json; goldнеисправлялся.','',
 '## FP/FN, Precision/Recall и Exact Match','', '| Модель | Набор | ΔF1,п.п. | P | R | TP | FP(Δ) | FN(Δ) | Exact |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in final.items():
  for dataset in ['pub_validation','official_208','public_test_clean','v3_test']:
   m=r['metrics'][dataset];b=base[dataset]
   lines.append(f"| {name} | {dataset} | {100*(m['micro_f1']-b['micro_f1']):+.3f} | {m['micro_precision']:.4f} | {m['micro_recall']:.4f} | {m['tp']} | {m['fp']}({m['fp']-b['fp']:+d}) | {m['fn']}({m['fn']-b['fn']:+d}) | {m['exact_match']:.4f} |")
 lines+=['','## Параметры финалистов и стоимость','']
 for name,r in final.items():
  sel=r['selection'];lines += [f"- **{name}**: {json.dumps(sel['spec'],ensure_ascii=False)}; пороги={sel['thresholds']}; trees={r['tree_count']}; модель={r['model_bytes']/1024**2:.3f}MiB."]
 lines+=['','ВариантыTF-IDFдобавляют существующую разреженную векторизацию и пятьлинейныхголов к инференсу; encoder/chunker/retrieval прежние. Замерыprediction_seconds относятся только к пакетномуCatBoost и не являются end-to-end latency. Веса/признаки лучших вариантов сохранены отдельно, baselineне заменялся.','',
 '## Все эксперименты: отбор по calibration','', '| № | Конфигурация | Изменения | Cal public F1(0.5) | Cal V3 F1(0.5) | Лучший режим | Лучший score | Fit,с |','|---|---|---|---:|---:|---|---:|---:|']
 for i,r in enumerate(experiments,1):
  best=max(r['calibration'],key=lambda k:r['calibration'][k]['metrics']['score']);m=r['calibration']['fixed']['metrics']
  changes={k:v for k,v in r['spec'].items() if k!='name'}
  lines.append(f"| {i} | {r['spec']['name']} | {json.dumps(changes,ensure_ascii=False)} | {m['public']['f1']:.6f} | {m['v3']['f1']:.6f} | {best} | {r['calibration'][best]['metrics']['score']:.6f} | {r['fit_seconds']:.2f} |")
 lines+=['','Полные параметры, включая дефолты и колонки, —experiments.json. Все три режима порогов сFP/FN—calibration_metrics.csv. Категориальныйcat_idвbaselineужебылкорректным; numeric/onehot—альтернативы, а не исправление отсутствовавшей настройки. Bayesianbaggingисследован как заменаbaselineMVS, а не только изменениеtemperatureвMVS.','',
 '## Почему publictest хуже validation','']
 for name,m in audit['domain_differences'].items():lines.append(f"- {name}: P={m['micro_precision']:.4f},R={m['micro_recall']:.4f},FP={m['fp']},FN={m['fn']},gold=[]в{m['empty_gold']}строках,FPнапустомgold={m['fp_on_empty_gold']},FPR={m['fpr']:.4f}.")
 lines+=['',f"Вofficial208source_mcIdнет;101—допущение, не значение parquet. В{audit['train_overlap_count']}строках есть точное совпадение сactualpublictrain. Противоречивые группы и интервалы качества по срезам: "+json.dumps(audit['baseline_subsets'],ensure_ascii=False)+'.','',
 'СнижениеPrecisionнельзяцеликомприписатьмодели: частьFP—явно предлагаемые услуги, отсутствующие вgold. Удаление конфликтных групп не исправляет остальные спорные метки. Отрицательный эталон при перечислении своих услуг не превращается в правильный только из-за улучшенияF1.','',
 '## Какие признаки повышают ошибочные score','', '| Признак | Mean SHAP на FP | Mean abs SHAP на FP | Mean SHAP на TP |','|---|---:|---:|---:|']
 for r in shap[:12]:lines.append(f"| {r['feature']} | {r['mean_shap_on_fp']:+.4f} | {r['mean_absolute_shap_on_fp']:.4f} | {r['mean_shap_on_tp']:+.4f} |")
 lines+=['','SHAPвыраженвlogitиописываетатрибуциюbaseline, не причинность и не валидностьgold. Одинаковый признак может правильно поддерживатьTPиформально считаться ошибочным из-за спорной разметки. Распределения синтаксических статусов наFP:public_test_analysis.json.','',
 '## Изменения по категориям','', '| Модель | Набор | Категория | F1 | FP(Δ) | FN(Δ) |','|---|---|---|---:|---:|---:|']
 for name,r in final.items():
  if name=='C_syntax_baseline__fixed':continue
  for dataset in ['pub_validation','official_208','v3_test']:
   for cid,m in r['metrics'][dataset]['per_category'].items():
    b=base[dataset]['per_category'][cid];lines.append(f"| {name} | {dataset} | {cid} | {m['f1']:.4f} | {m['fp']}({m['fp']-b['fp']:+d}) | {m['fn']}({m['fn']-b['fn']:+d}) |")
 lines+=['','## Интервалы и регрессии','']
 for name,r in final.items():
  if name=='C_syntax_baseline__fixed':continue
  for dataset in ['pub_validation','official_208','v3_test']:
   m=r['metrics'][dataset];lines.append(f"- {name}/{dataset}: {json.dumps(m['transitions'],ensure_ascii=False)}; pairedCI={json.dumps(m['bootstrap_delta'])}; groupedCI={json.dumps(m.get('grouped_bootstrap_delta'))}.")
 lines+=['','## Ответ на критерий трёх наборов','']
 winners=[]
 for name,r in final.items():
  if name=='C_syntax_baseline__fixed':continue
  m=r['metrics']
  if m['pub_validation']['micro_f1']>=base['pub_validation']['micro_f1'] and m['official_208']['micro_f1']>base['official_208']['micro_f1'] and m['v3_test']['micro_f1']>=base['v3_test']['micro_f1']:winners.append(name)
 lines += ['Формально улучшилиpublictestбезсниженияpublicvalidationиV3: '+(', '.join(winners) if winners else '**ни один из заранее выбранных финалистов**')+'.', '',
 'Семантическая оценка конкретных изменений вынесена вpublic_test_semantic_review.md; она важнее формального исправления сомнительногоgold=[]. Нельзя объявлять модель лучше только за подавление явно предлагаемых услуг. Все рекомендации ограничены этой серией и известными проверками; независимого финального теста нет. Поиск40+вариантов наоднойcalibrationтакже может приводить к оптимизму. Baselineнепродвигался и не заменялся.']
 (P/'report.md').write_text('\n'.join(lines)+'\n')
 print('Report written; formal joint winners:',winners)
if __name__=='__main__':main()
