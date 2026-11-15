"""Render already-frozen scores; never refit or select additional candidates."""
import hashlib,json
from pathlib import Path
P=Path('output/tail-services-v3-experiment')
def main():
 m=json.loads((P/'metrics.json').read_text());v=json.loads((P/'training_validation.json').read_text());enc=json.loads((P/'encoding.json').read_text());audit=json.loads((P/'dataset_audit.json').read_text());examples=json.loads((P/'error_examples.json').read_text());first_encoding=json.loads((P/'quality_correction/encoding.json').read_text())
 names=['pub_validation','official_208','public_test_clean','v3_test','tail_holdout20']
 lines=['# C_syntax: добавление 60 объявлений с услугами в конце','',
'A — исходная C_syntax без hard-negative200. B — та же цепочка Qwen → category-aware Top3 → SGD → Natasha/Slovnet → CatBoost на public3398 + V3clean2921 + 60. Только новые данные; архитектура, признаки, порог0.5, параметры полных SGD-голов и CatBoost сохранены. Исходная рабочая модель не заменена.','',
'## Данные и протокол','',
'Новые60: 20 предложений в характеристиках/последнем абзаце, 20 разнообразных названий и формулировок, 10 реальных предложений с SEO, 10 контекстных упоминаний без целевой услуги. Каждая категория102–106 имеет10 положительных примеров. Всего50 непустых и10 пустых gold, source101. Это отдельно подключаемый файл V3, исходные записи не редактировались.','',
'Отдельный новый test20: 16 непустых и4 пустых, включая одно предложение двух услуг. Его метки написаны до инференса, в обучение/калибровку они не входят. Нет пар «дословно один вход ± один тег»: здесь оценивается добавление данных и перенос на отдельные объявления, а не причинный эффект каждого SEO-слова.','',
'Все80 нормализованных описаний уникальны, новых точных пересечений с прежними train/calibration/validation/test нет. Максимальное сходство новых train/test замороженным TF-IDF =0.332014. Длина train280–1436 символов, median324; test234–1166, median274. Основная часть набора короткая; это ограничение. Разметка экспертная авторская, без независимого аннотатора. Подробности — dataset_audit.md и отдельные rationale для всех80.','',
'Начальные входы зафиксированы0703f0e; протокол единственной финальной оценки — e41168f. До любой тестовой оценки при повторной сверке справочника один неоднозначный пример о декоративной штукатурке заменён явной окраской. Первый OOF-проход остановлен; результаты не использовались. Исправленные входы и параметры зафиксированы2ddc6c3, затем обучение B запущено заново. test20 и остальные gold не менялись. Новые данные не пересматривались по test-F1. Полный grouped nested OOF пересчитан на всём train6379; объединять старые OOF с новыми нельзя. Внутренний подбор alpha — прежняя train-only процедура. Полные SGD-головы клонируют прежние параметры. CatBoost early stopping — на неизменных calibration728+V3validation600.','',
f"Получено {v['group_diagnostics']['n_components']} групп train; max size {v['group_diagnostics']['max_group_size']}. Завершённое обучение B с OOF и подготовкой признаков: {v['training_seconds']:.1f} с; CatBoost best iteration {v['best_iteration']}. Новые embeddings: сначала {first_encoding['chunk_count']} чанка за {first_encoding['seconds']:.1f} с; после исправления ещё {enc['chunk_count']} чанк за {enc['seconds']:.1f} с. Прежние неизменённые новые чанки переиспользованы. Время прерванного первого OOF в завершённое обучение не включено.",'',
'Colab CLI упал на записи служебного журнала в read-only каталог, в том числе после escalation. VM не создавалась. Использован существующий CPU-путь того же frozen Qwen из локального snapshot: float32 inference, нормированные векторы сохранены float16, max_seq_length1024. Старые кэши не пересчитывались. Это численная разница относительно прежнего CUDA float16, а не замена энкодера; она ограничивает буквальную идентичность вычислительного окружения.','',
'## Основные результаты','',
'| Набор | N | Micro-F1 A | Micro-F1 B | Δ, п.п. | Precision A→B | Recall A→B | FP A→B | FN A→B | Exact A→B |','|---|---:|---:|---:|---:|---|---|---|---|---|']
 for n in names:
  r=m[n];a=r['A'];b=r['B'];lines.append(f"| {n} | {r['rows']} | {a['micro_f1']:.6f} | {b['micro_f1']:.6f} | {100*(b['micro_f1']-a['micro_f1']):+.3f} | {a['precision']:.4f}→{b['precision']:.4f} | {a['recall']:.4f}→{b['recall']:.4f} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} | {a['exact_match']:.4f}→{b['exact_match']:.4f} |")
 lines+=['','Публичный test208 имеет отсутствующее поле source_mcId (101 — прежняя документированная конвенция), противоречивые gold и существующие train-пересечения. Clean152 исключает одинаковые56 строк из13 конфликтных групп для A/B; gold не меняется. Улучшение формального F1 на этом тесте само по себе не доказывает улучшение понимания услуг. Public validation и V3test также уже анализировались раньше. public_synthetic_300 не использован.','',
'## FP/FN и F1 по категориям','',
'| Набор | Категория | F1 A→B | FP A→B | FN A→B |','|---|---:|---|---|---|']
 for n in names:
  for cid,a in m[n]['A']['per_category'].items():
   b=m[n]['B']['per_category'][cid];lines.append(f"| {n} | {cid} | {a['f1']:.4f}→{b['f1']:.4f} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} |")
 lines+=['','## Исправления, регрессии и пустые ответы','',
'| Набор | Исправленные пары | Новые ошибочные пары | Полностью исправленные объявления | Новые ошибочные объявления | Новые ошибочные [] | FP на gold=[] A→B |','|---|---:|---:|---:|---:|---:|---|']
 for n in names:
  r=m[n];t=r['transitions'];lines.append(f"| {n} | {t['base_wrong_cand_right_fixed']} | {t['base_right_cand_wrong_broken']} | {t['ads_fully_fixed']} | {t['ads_newly_wrong']} | {t['new_erroneous_empty_on_nonempty_gold']} | {r['A']['fp_on_empty_gold']}→{r['B']['fp_on_empty_gold']} |")
 lines+=['','## Статистическая неопределённость','',
'Парный bootstrap4000: группировка по нормализованному полному описанию, все точные дубликаты переотбираются совместно. Это учитывает точные повторы, но не все зависимости между семантически близкими шаблонами. Интервалы для нового test20 особенно нестабильны.','',
'| Набор | Δ Micro-F1, п.п. | 95% CI, п.п. | Групп |','|---|---:|---|---:|']
 for n in names:
  r=m[n];c=r['bootstrap_delta_micro_f1'];lines.append(f"| {n} | {100*r['delta']['micro_f1']:+.3f} | [{100*c['ci95_low']:+.3f}; {100*c['ci95_high']:+.3f}] | {c['groups']} |")
 if 'pub_validation_complex_repair' in m:
  r=m['pub_validation_complex_repair'];lines+=['','## Комплексный ремонт с несколькими услугами','',f"Описательный срез public validation: {r['rows']} объявлений с >=2 gold и формулировкой о комплексном/капитальном ремонте или ремонте под ключ. F1 {r['A']['micro_f1']:.6f}→{r['B']['micro_f1']:.6f}; FP {r['A']['fp']}→{r['B']['fp']}, FN {r['A']['fn']}→{r['B']['fn']}. Это эвристика для отчётного среза, не фильтр входа и не новое правило классификации."]
 lines+=['','## Полные входы: исправления и регрессии','',
'Ниже случаи выбираются только для объяснения уже измеренного результата. Все изменения, включая одинаковое число ошибок при другом составе, сохранены в error_examples.json. Оценка по gold не заменяет смысловую проверку; отдельные официальные метки спорны.','']
 chosen=[]
 for n in ['pub_validation','v3_test','official_208','tail_holdout20']:
  for typ,sign in [('Исправление',-1),('Регрессия',1)]:
   arr=[e for e in examples if e['split']==n and (e['errors_B']-e['errors_A'])*sign>0]
   for e in arr[:2]:chosen.append((typ,e))
 for typ,e in chosen:
  lines +=[f"### {typ}: {e['split']} #{e['row_idx']}",'',f"Source={e['source_mcId']}; gold={e['gold']}; A={e['A']}; B={e['B']}.",'','```text',e['description'],'```','']
 lines+=['## Новый test20: полный реестр ответов','',
'| ID | Gold | A | B | Ошибок A→B |','|---|---|---|---|---|']
 for e in examples:
  if e['split']=='tail_holdout20':lines.append(f"| {e['item_id']} | {e['gold']} | {e['A']} | {e['B']} | {e['errors_A']}→{e['errors_B']} |")
 lines+=['','## Вывод','',
'Окончательная смысловая интерпретация и рекомендация сохранены ниже после просмотра исправлений/регрессий. Ни одна последующая интерпретация не ведёт к повторному обучению или перенастройке этого эксперимента. Новые60 — менее1% исходного train; они не гарантируют устойчивость ко всем хвостам и редким формулировкам. Даже выигрыш на test20 может отражать перенос на стиль автора, поэтому общий перенос оцениваем вместе с прежними публичными и V3 наборами.','',
'Один B-кандидат не позволяет отделить эффект содержания новых объявлений от изменений фолдов и чувствительности оптимизации. Проверка обучающих примеров не является доказательством обобщения. Автоматической замены baseline нет. Файлы metrics.json, predictions.csv, config.json, training_validation.json, verification.json и dataset_audit.md содержат измерения и проверки; models/ — отдельные веса B локально.','']
 (P/'report.md').write_text('\n'.join(lines))
 if 'tail_holdout20_focus' in m:
  extra=['','## Новый test20 по заранее заданным конструкциям','','| Конструкция | N | Exact A→B | FP A→B | FN A→B |','|---|---:|---|---|---|']
  for key,r in m['tail_holdout20_focus'].items():
   a=r['A'];b=r['B'];extra.append(f"| {key} | {r['rows']} | {a['exact_match']:.3f}→{b['exact_match']:.3f} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} |")
  with (P/'report.md').open('a') as f:f.write('\n'.join(extra)+'\n')
 # Lightweight standard scientific plot; data are the stored frozen metrics.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 fig,axes=plt.subplots(1,2,figsize=(11,4.2));x=np.arange(len(names));w=.36
 axes[0].bar(x-w/2,[m[n]['A']['micro_f1'] for n in names],w,label='A baseline');axes[0].bar(x+w/2,[m[n]['B']['micro_f1'] for n in names],w,label='B +60')
 axes[0].set_ylim(0,1.03);axes[0].set_ylabel('Micro-F1');axes[0].legend()
 axes[1].bar(x-w/2,[m[n]['delta']['fp'] for n in names],w,label='FP delta');axes[1].bar(x+w/2,[m[n]['delta']['fn'] for n in names],w,label='FN delta');axes[1].axhline(0,color='black',linewidth=.7);axes[1].legend();axes[1].set_ylabel('B - A errors')
 for ax in axes:ax.set_xticks(x);ax.set_xticklabels(['pub val','official208','clean152','V3 test','tail20'],rotation=20,ha='right');ax.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(P/'comparison.png',dpi=160);plt.close(fig)
 # Portable notebook reads artifacts; it never retrains or reevaluates the model.
 nb={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':[{'cell_type':'markdown','id':'intro','metadata':{},'source':['# C_syntax +60 tail-service examples\n','Frozen single-pass A/B. See report.md for caveats and full inputs.']},{'cell_type':'code','id':'metrics','metadata':{},'execution_count':None,'outputs':[],'source':['import json\n','from pathlib import Path\n','import pandas as pd\n','root = Path.cwd()\n',"p = root if (root / 'metrics.json').exists() else root / 'output/tail-services-v3-experiment'\n","m = json.loads((p / 'metrics.json').read_text())\n",'pd.DataFrame([{\'split\': n, \'F1_A\': r[\'A\'][\'micro_f1\'], \'F1_B\': r[\'B\'][\'micro_f1\'], \'FP_A\': r[\'A\'][\'fp\'], \'FP_B\': r[\'B\'][\'fp\'], \'FN_A\': r[\'A\'][\'fn\'], \'FN_B\': r[\'B\'][\'fn\']} for n, r in m.items() if \'A\' in r])']},{'cell_type':'code','id':'chart','metadata':{},'execution_count':None,'outputs':[],'source':['from IPython.display import Image, display\n',"display(Image(filename=str(p / 'comparison.png')))" ]},{'cell_type':'code','id':'cases','metadata':{},'execution_count':None,'outputs':[],'source':["cases = json.loads((p / 'error_examples.json').read_text())\n","pd.DataFrame(cases)[['split','row_idx','source_mcId','gold','A','B','errors_A','errors_B','description']]" ]}]}
 (P/'results.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
 print('Report, plot and notebook saved; no training/evaluation performed.')
if __name__=='__main__':main()
