"""Summarize the single fixed TF-IDF refit and manually reviewed example differences."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path('output/tfidf-public-v3-recheck')
CAT={102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',105:'Электрика',106:'Поклейка обоев и малярные работы'}

def main():
    result=json.loads((P/'metrics.json').read_text());config=json.loads((P/'config.json').read_text())
    baseline=json.loads(Path('output/catboost-optimization/final_metrics.json').read_text())['C_syntax_baseline__fixed']['metrics']
    candidates=json.loads((P/'example_candidates.json').read_text())
    selected=[('official_208',159,'Лишняя штукатурка','Предлагаются чистовые работы: покраска и обои. Штукатурка в тексте не предложена. TF-IDF добавил 102, C_syntax оставила 106.'),
              ('official_208',59,'Пропуск плиточных работ','Облицовка крыльца плиткой прямо предложена своей бригадой. TF-IDF вернул пустой ответ, C_syntax определила 103.'),
              ('v3_test',96,'Отрицание одной услуги при предложении другой','Подключение раковины 104 предложено, укладка керамогранита 103 прямо отвергнута. TF-IDF включает обе категории; C_syntax учитывает отказ.'),
              ('v3_test',97,'Чужой подрядчик','Предложена сборка мебели. Шпаклевку делает подрядчик заказчика, автор ее прямо не предлагает. TF-IDF добавляет 106, C_syntax возвращает пустой ответ.'),
              ('v3_test',103,'Пропуск услуги при условии доступа','Подключение раковины 104 прямо предложено при условии доступа. Штукатурная категория 102 совпадает с исходной и исключается из дополнительных. TF-IDF пропускает 104; C_syntax находит её.')]
    examples=[]
    for dataset,row,title,reason in selected:
        r=next(r for r in candidates if r['dataset']==dataset and r['row_idx']==row)
        assert r['c_syntax_prediction']==r['gold'] and r['tfidf_prediction']!=r['gold']
        examples.append({**r,'title_for_report':title,'expert_assessment':reason,
                         'input':{'description':r['description'],'mcId':r['mcId']}})
    (P/'examples.json').write_text(json.dumps(examples,ensure_ascii=False,indent=2)+'\n')
    pred=pd.read_csv(P/'predictions.csv')
    old=pd.read_csv('output/catboost-optimization/final_predictions.csv');old=old[old.model=='C_syntax_baseline__fixed']
    transitions={}
    for dataset in ['official_208','v3_test']:
        q=pred[pred.dataset==dataset];o=old[old.dataset==dataset]
        gold=q.pivot(index='row_idx',columns='cat_id',values='gold').reindex(columns=list(CAT)).to_numpy()
        tf=q.pivot(index='row_idx',columns='cat_id',values='prediction').reindex(columns=list(CAT)).to_numpy()
        cb=o.pivot(index='row_idx',columns='cat_id',values='prediction').reindex(index=range(len(gold)),columns=list(CAT)).fillna(0).to_numpy()
        tf_right=(tf==gold).all(1);cb_right=(cb==gold).all(1)
        transitions[dataset]=dict(tfidf_wrong_c_syntax_exact_right=int((~tf_right&cb_right).sum()),
                                  tfidf_exact_right_c_syntax_wrong=int((tf_right&~cb_right).sum()))
    (P/'transitions.json').write_text(json.dumps(transitions,indent=2)+'\n')
    lines=['# TF-IDF + SGD на public train + V3 train','',
           'Один новый запуск обучения существующего standalone pipeline, без изменения архитектуры и гиперпараметров и без подбора по тестам. Рабочие модели не заменялись.','',
           '## Обучение и параметры','',
           '- Public train: 3398 объявлений из `output/ngram-comparison/splits/train.jsonl`.',
           '- V3 train: 2921 объявление из `datasets/own_v3_clean/train.jsonl`; без дополнительных 200 hard negatives и без старого отдельного дополнения из 100 примеров.',
           '- Всего: **6319**. Порядок public, затем V3, как в исходном сравнении C_syntax.',
           '- Вход: description, без title. Обучающие метки — предлагаемые категории 102–106; при оценке исходный mcId исключается из ответа прежней функцией pipeline.',
           '- Словесный TF-IDF: **(1,3)**, max_features=18000, min_df=2, sublinear_tf=True.',
           '- Символьный TF-IDF: **char_wb (3,5)**, max_features=28000, min_df=3, sublinear_tf=True.',
           '- OneVsRestClassifier(SGDClassifier): loss=log_loss, alpha=0.0001, max_iter=1000, tol=0.001, random_state=42.',
           '- Все пороги **0.5**; calibration не использовалась. Векторизаторы обучены только на train.',
           f"- Время нового fit на CPU: **{result['training_seconds']:.2f} с**. Фактические признаки: {result['features']['word']} словесных + {result['features']['char']} символьных. Модель: {result['model_bytes']/1024**2:.3f} MiB.",'',
           'Фабрика pipeline: `avito_compare/scripts/qwen3_v3_experiment.py::build_tfidf_pipeline`. Это компактный baseline, который использовался и в сравнении синтаксической архитектуры. Свежие вероятности на всех трех наборах **точно совпали** с ранее сохраненной TF-IDF-моделью на том же train (максимальное отличие 0.0).','',
           '## Результаты','',
           '| Набор | Строк | TF-IDF Micro-F1 | Precision | Recall | FP | FN | Exact Match | C_syntax Micro-F1 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name in ['official_208','v3_test','pub_validation']:
        m=result['metrics'][name];b=baseline[name]
        lines.append(f"| {name} | {m['rows']} | {m['micro_f1']:.6f} | {m['micro_precision']:.6f} | {m['micro_recall']:.6f} | {m['fp']} | {m['fn']} | {m['exact_match']:.6f} | {b['micro_f1']:.6f} |")
    clean=result['metrics']['official_no_exact_train_overlap_188']
    lines+=['',f"Public test — именно 208 строк parquet, не validation на 729 строках. В нём уже есть 20 точных совпадений с фактическим public train и противоречивые gold. Поэтому он не независим: на остальных 188 строках TF-IDF Micro-F1={clean['micro_f1']:.6f}. Исходная категория 101 для parquet принята по документированной конвенции README; поля в самом файле нет. V3 test содержит 600 строк, точных совпадений описаний и пересечений group_id с V3 train не найдено.",'',
            '## Сравнение с исходной C_syntax','',
            '| Набор | FP TF-IDF → C_syntax | FN TF-IDF → C_syntax | ΔMicro-F1 C_syntax, п.п. |',
            '|---|---:|---:|---:|']
    for name in ['official_208','v3_test','pub_validation']:
        a=result['metrics'][name];b=baseline[name]
        lines.append(f"| {name} | {a['fp']} → {b['fp']} | {a['fn']} → {b['fn']} | {100*(b['micro_f1']-a['micro_f1']):+.3f} |")
    lines+=['','На полном public test TF-IDF формально немного лучше C_syntax. На V3 C_syntax лучше на 22.33 п.п.: особенно заметно уменьшение пропусков 408→80. Выбранные ниже положительные примеры не означают превосходства C_syntax на каждом наборе или на всех объявлениях.','',
            f"По полному совпадению набора категорий: на public test C_syntax исправляет {transitions['official_208']['tfidf_wrong_c_syntax_exact_right']} ошибочных TF-IDF-ответов, но в {transitions['official_208']['tfidf_exact_right_c_syntax_wrong']} случаях происходит обратное. На V3 соответствующие числа — {transitions['v3_test']['tfidf_wrong_c_syntax_exact_right']} и {transitions['v3_test']['tfidf_exact_right_c_syntax_wrong']}. Это построчные счетчики с повторами, не независимые ситуации.",'',
            '## Примеры: TF-IDF ошибся, C_syntax верна','',
            'Полные тексты приведены без сокращений. Индексы с нуля; gold не исправлялся. Указаны сохраненные ответы исходной C_syntax при 0.5, а не оптимизированные финалисты. Публичные примеры проверены по смыслу: штукатурка не предложена в строке 159, а облицовка плиткой явно предложена в строке 59.','','| ID | Категория |','|---|---|']
    for c,n in CAT.items():lines.append(f'| {c} | {n} |')
    for idx,r in enumerate(examples,1):
        lines+=['',f"### {idx}. {r['dataset']}, строка {r['row_idx']}: {r['title_for_report']}",'',
                f"**Входной mcId:** `{r['mcId']}`. **Полный description:**",'',
                '```text',r['description'],'```','',
                f"**Gold:** `{r['gold']}`. **TF-IDF:** `{r['tfidf_prediction']}`. **C_syntax:** `{r['c_syntax_prediction']}`.",'',
                r['expert_assessment'],'']
    lines+=['## Воспроизведение и артефакты','',
            '`run.py` использует прежнюю фабрику pipeline; новый fit выполнен один раз. `config.json` содержит параметры, версии и SHA-256 входов. `metrics.json`, `predictions.csv`, `transitions.json`, `examples.json` сохраняют результаты. Новые веса — `models/tfidf_sgd.joblib`, исходные веса не перезаписывались.','',
            'Из корня проекта: `python output/tfidf-public-v3-recheck/run.py`, затем `python output/tfidf-public-v3-recheck/build_report.py`. Повторный fit при уже существующем metrics.json запрещен; завершенную серию нельзя перезаписывать ради подбора по тестам. Здесь не использовались Qwen/Colab для обучения, новые признаки или измененные gold.']
    (P/'report.md').write_text('\n'.join(lines).rstrip()+'\n')
    print('Wrote report with 5 complete examples and metrics comparison. Transitions:',transitions)

if __name__=='__main__':main()
