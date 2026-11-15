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
 f"Проведено {len(experiments)} обучений CatBoost на исходных public 3398 + V3 clean 2921; новые hard negatives не использовались. Qwen3-Embedding-0.6B, fixed chunker, Top-3 и полные SGD-головы baseline не менялись. Исходный grouped nested OOF восстановлен отдельно; TF-IDF OOF получен с обучением векторизатора только внутри каждого train-фолда. Все вычисления выполнены локально на CPU; Colab не использовался.",'',
 '## Протокол отбора','',
 'В поиске участвуют только train и calibration: 1328 объявлений (public 728 + V3 validation 600). Early stopping — по calibration Logloss с прежним patience=50. Основной score отбора: 0.65 × public calibration F1 + 0.35 × V3 calibration F1. До финального скоринга зафиксированы четыре различных финалиста: сбалансированный, лучший public calibration с ограничением падения V3, вариант с низким FP и ограничением Recall, лучший вариант с фиксированным порогом 0.5. Пороги 0.5, общий и покатегорийные сравнивались только на calibration. Финалисты и SHA-256 весов зафиксированы в коммите fff2d4b до оценки на проверочных наборах. После просмотра public test новые конфигурации не запускались.','',
 '## Итоговая таблица','',
 '| Модель | pub_validation | public_test_208 | public_test_clean | v3_test |','|---|---:|---:|---:|---:|']
 for name,r in final.items():
  m=r['metrics'];lines.append(f"| {name} | {m['pub_validation']['micro_f1']:.6f} | {m['official_208']['micro_f1']:.6f} | {m['public_test_clean']['micro_f1']:.6f} | {m['v3_test']['micro_f1']:.6f} |")
 lines+=['','Clean — 152 строки: исключены все 56 строк из 13 групп с противоречивым gold при одинаковом нормализованном description. Это **не** прежние 114 уникальных строк без пересечений. Здесь сохраняются непротиворечивые повторы и возможные пересечения с train. Все модели используют один список public_test_exclusions.json; gold не исправлялся.','',
 '## FP/FN, Precision/Recall и Exact Match','', '| Модель | Набор | ΔF1,п.п. | P | R | TP | FP(Δ) | FN(Δ) | Exact |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
 for name,r in final.items():
  for dataset in ['pub_validation','official_208','public_test_clean','v3_test']:
   m=r['metrics'][dataset];b=base[dataset]
   lines.append(f"| {name} | {dataset} | {100*(m['micro_f1']-b['micro_f1']):+.3f} | {m['micro_precision']:.4f} | {m['micro_recall']:.4f} | {m['tp']} | {m['fp']}({m['fp']-b['fp']:+d}) | {m['fn']}({m['fn']-b['fn']:+d}) | {m['exact_match']:.4f} |")
 lines+=['','## Параметры финалистов и стоимость','']
 for name,r in final.items():
  sel=r['selection'];lines += [f"- **{name}**: {json.dumps(sel['spec'],ensure_ascii=False)}; пороги={sel['thresholds']}; trees={r['tree_count']}; модель={r['model_bytes']/1024**2:.3f}MiB."]
 benchmark=load('tfidf_inference_benchmark.json')
 lines+=['',f"Суммарное время 45 fit CatBoost: {sum(r['fit_seconds'] for r in experiments):.2f} с (без подготовки OOF). Восстановление исходного nested SGD OOF: {load('oof_diagnostics.json')['seconds']:.2f} с. Дополнительный сохранённый TF-IDF занимает {benchmark['model_bytes']/1024**2:.3f} MiB. На {benchmark['texts']} calibration-текстах медиана трёх прогретых пакетных запусков TF-IDF + 5 SGD-голов — {benchmark['median_batch_seconds']:.3f} с, или {benchmark['median_ms_per_ad_amortized']:.3f} мс на объявление в среднем по пакету. Это измерение добавочной CPU-работы, без Qwen и Natasha; оно не является задержкой одиночного запроса."]
 lines+=['','Варианты с TF-IDF добавляют существующую разреженную векторизацию и пять линейных голов к инференсу; encoder, chunker и retrieval прежние. Использованы словесные триграммы (1,3) и char_wb (3,5), с прежними лимитами 18k/28k. Замеры prediction_seconds относятся только к пакетному CatBoost и не являются полной задержкой инференса. Полная задержка дополненного пайплайна не измерялась. Веса лучших вариантов сохранены отдельно; baseline не заменялся.','',
 '## Все эксперименты: отбор по calibration','', '| № | Конфигурация | Изменения | Cal public F1(0.5) | Cal V3 F1(0.5) | Лучший режим | Лучший score | Fit,с |','|---|---|---|---:|---:|---|---:|---:|']
 for i,r in enumerate(experiments,1):
  best=max(r['calibration'],key=lambda k:r['calibration'][k]['metrics']['score']);m=r['calibration']['fixed']['metrics']
  changes={k:v for k,v in r['spec'].items() if k!='name'}
  lines.append(f"| {i} | {r['spec']['name']} | {json.dumps(changes,ensure_ascii=False)} | {m['public']['f1']:.6f} | {m['v3']['f1']:.6f} | {best} | {r['calibration'][best]['metrics']['score']:.6f} | {r['fit_seconds']:.2f} |")
 lines+=['','Полные параметры, включая дефолты и колонки, — experiments.json. Все три режима порогов с FP/FN — calibration_metrics.csv. Категориальный cat_id в baseline уже был корректным; numeric и one-hot — альтернативные представления. Bayesian bagging исследован с заменой исходного MVS. Проверены 34 исходных конфигурации и 12 адаптивных комбинаций; одна комбинация совпала с уже выполненной, поэтому всего 45 обучений, включая воспроизведение baseline.','',
 '## Почему publictest хуже validation','']
 for name,m in audit['domain_differences'].items():lines.append(f"- {name}: P={m['micro_precision']:.4f}, R={m['micro_recall']:.4f}, FP={m['fp']}, FN={m['fn']}; gold=[] в {m['empty_gold']} строках, FP на пустом gold={m['fp_on_empty_gold']}, FPR={m['fpr']:.4f}.")
 lines+=['',f"В official_208 отсутствует source_mcId; 101 — допущение, а не значение parquet. В {audit['train_overlap_count']} строках есть точное совпадение с фактическим public train. Метрики baseline по срезам: "+json.dumps(audit['baseline_subsets'],ensure_ascii=False)+'.','',
 'Precision снижается по двум различимым причинам. Есть реальные ошибки переноса: выравнивание деревянного дома принимается за штукатурку, обсуждение отопления — за сантехнику. Одновременно часть формальных FP — явно предлагаемые услуги, отсутствующие в gold. Пустых gold на public test 51.0%, на validation 27.8%; FPR также выше (14.9% против 5.9%), поэтому одной долей отрицательных строк разрыв не объясняется. Удаление конфликтных групп не исправляет остальные спорные метки. По этим данным нельзя точно разложить весь разрыв на ошибку модели и ошибку разметки.','',
 '## Какие признаки повышают ошибочные score','', '| Признак | Mean SHAP на FP | Mean abs SHAP на FP | Mean SHAP на TP |','|---|---:|---:|---:|']
 for r in shap[:12]:lines.append(f"| {r['feature']} | {r['mean_shap_on_fp']:+.4f} | {r['mean_absolute_shap_on_fp']:.4f} | {r['mean_shap_on_tp']:+.4f} |")
 lines+=['','SHAP выражен в logit и описывает атрибуцию baseline, без доказательства причинности или валидности gold. На FP больше всего положительно влияют p_sgd и флаги упоминания услуги во всём объявлении и Top-3. В 81 из 111 FP нет признака явного положительного предложения; у 74 связь с услугой orphan. Это согласуется с зависимостью от лексического упоминания, но не доказывает причину каждой ошибки. syn_global_turnkey_only равен 0 во всех 111 FP: нельзя объяснить эти ошибки только этим флагом. Подробные распределения: public_test_analysis.json.','',
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
 lines += ['По измеренному F1 улучшили public test без снижения public validation и V3: '+(', '.join(winners) if winners else '**ни один из заранее выбранных финалистов**')+'.', '',
 '**Основной кандидат по балансу calibration — adaptive_0_3__per_category:** глубина 6, learning_rate=0.035, l2_leaf_reg=15, дополнительная OOF-вероятность TF-IDF. Пороги категорий 102–106: [0.600, 0.525, 0.375, 0.375, 0.250]. На public test выигрыш +0.802 п.п.; на clean +1.153 п.п. Но на V3 FN выросли с 80 до 88, а 10 объявлений с непустым gold впервые получили пустой ответ. Поэтому рост F1 не означает улучшение всех типов ошибок.', '',
 '**Более простой кандидат — adaptive_1_0__fixed:** глубина 8, l2_leaf_reg=15, остальные параметры baseline, прежние признаки и порог 0.5. V3 F1 вырос с 0.915916 до 0.926829; FN сократились с 80 до 64. Выигрыш на public test всего +0.119 п.п. и семантически смешанный: одно формальное исправление подавляет явно предложенные малярные работы (строка 30). Это нельзя считать доказанным улучшением public test.', '',
 '**Статистическая убедительность:** у основного кандидата групповой парный bootstrap CI изменения public test F1 составляет [-0.576; +2.638] п.п. и включает ноль. Шесть исправленных строк о деревянных домах — копии одного текста; их нельзя трактовать как шесть независимых подтверждений. На V3 CI основного кандидата также включает ноль. У глубины 8 прирост V3 устойчивее: CI [+0.514; +1.696] п.п., но на обоих публичных наборах интервал включает ноль. Это интервалы по данным наборам, без поправки на историческое многократное использование проверок.', '',
 'Максимальный сырой public test F1=0.830189 у adaptive_0_2__per_category, но V3 снизился до 0.913767 и FN выросли до 89. Выбирать его только ради public test не рекомендуется. Вариант l2_20__global не прошёл итоговую проверку на публичных наборах.', '',
 '**Рекомендация:** сохранить baseline основной моделью, а основной TF-IDF-финалист и глубину 8 — отдельными кандидатами. Измеренный совместный рост F1 получен, но статистически и семантически убедительное улучшение public test без компромиссов не доказано. Для дальнейшего выбора нужен независимый контроль; повторная настройка по этим тестам не проводилась.', '',
 'Смысловая проверка конкретных исправлений и регрессий: [public_test_semantic_review.md](public_test_semantic_review.md). На срезе 377 публичных объявлений о комплексном ремонте с несколькими gold-услугами F1 baseline=0.977054, основного кандидата=0.981145, глубины 8=0.978398. Этот срез задан текстовым фильтром и непустым gold; он не заменяет ручной проверки всех перечислений услуг.', '',
 'Все рекомендации ограничены этой серией; независимого финального теста нет. Отбор 45 моделей и порогов на одной calibration также создаёт риск переоптимизации. Baseline не заменялся; датасеты и тестовые gold не изменялись.']
 (P/'report.md').write_text('\n'.join(lines)+'\n')
 print('Report written; formal joint winners:',winners)
if __name__=='__main__':main()
