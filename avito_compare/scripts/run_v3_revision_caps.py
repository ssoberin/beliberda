"""46k/60k/100k on fixed revision3 data; modest threshold choice on calibration only."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
import duckdb,joblib,numpy as np
from .run_v3_revision_study import score
from .run_v3_fraction_study import read,adapt
from .evaluate_v2_experiment import IDS,metrics
from .feature_experiments import rss_mib
from .prepare_dataset import normalize
P=Path('output/v3-revision3-caps');OLD=Path('output/v3-revision3-study');DATA=OLD/'after/splits'
BASE=Path('/tmp/innopolis-dataset-v3-review/datasets/own_v3')
def probability(model,rows):
    y=np.asarray([[int(c in r['evaluation_labels']) for c in IDS] for r in rows],dtype=np.int8)
    prob=model['pipeline'].predict_proba([r['description'][:6000] for r in rows])
    for i,r in enumerate(rows):
        if r['mcId'] in IDS:
            j=IDS.index(r['mcId']);prob[i,j]=-1
    return y,prob
def main():
    if P.exists():raise ValueError('Fresh output required')
    P.mkdir(parents=True)
    reference=json.loads((OLD/'summary.json').read_text())['runs']['after']['fit']
    publicval=[r for r in read(DATA/'validation.jsonl') if r['dataset']=='public_validation']
    calibration=read(DATA/'calibration.jsonl')+[adapt(r,'v3_calibration') for r in read(BASE/'validation.jsonl')]
    sets={'public_validation_729':publicval,'v3_test_600':[adapt(r,'v3_test') for r in read(BASE/'test.jsonl')],'v3_challenge_144':[adapt(r,'v3_challenge') for r in read(BASE/'challenge.jsonl')],'new_control_20':[adapt(r,'authored_test') for r in read(Path('datasets/context_pairs_100/test_20.jsonl'))]}
    names={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,'Электрика':105,'Поклейка обоев и малярные работы':106}
    raw=duckdb.connect().execute('select description,categories from read_parquet(?)',['datasets/public/public-test-dataset.parquet']).fetchall()
    sets['official_public_test_208']=[{'description':d,'mcId':101,'evaluation_labels':sorted(names[c] for c in cats)} for d,cats in raw]
    calkeys={normalize(r['description']) for r in calibration}
    assert not {normalize(r['description']) for r in read(DATA/'train.jsonl')}&calkeys
    for n,rows in sets.items():
        if n!='official_public_test_208':assert not calkeys&{normalize(r['description']) for r in rows}
    results={}
    for name,word,char in [('46k',18000,28000),('60k',24000,36000),('100k',40000,60000)]:
        folder=P/name;folder.mkdir()
        if name=='46k':modeldir=OLD/'after/model';measured=reference
        else:
            modeldir=folder/'model'
            cmd=[sys.executable,'-m','avito_compare.scripts.measure_training','--data-dir',str(DATA),'--output-dir',str(modeldir),'--word-ngram-max','3','--char-analyzer','char_wb','--word-max-features',str(word),'--char-max-features',str(char),'--exclude-source-category']
            env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
            print('START',name,'word(1,3) char_wb(3,5), fixed data and thresholds',flush=True)
            with (folder/'fit.log').open('w') as log:
                proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
                while proc.poll() is None:
                    if rss_mib(proc.pid)>2048 or time.monotonic()-start>180:
                        proc.kill();proc.wait();raise RuntimeError('Resource limit')
                    time.sleep(.1)
            assert proc.returncode==0
            measured=json.loads((modeldir/'measurement.json').read_text())
            for k in ['estimator_parameters','thresholds','versions']:assert measured[k]==reference[k],k
            expected=copy.deepcopy(reference['vectorizer_parameters'])
            expected['word']['max_features']=word;expected['char']['max_features']=char
            assert measured['vectorizer_parameters']==expected
        model=joblib.load(modeldir/'tfidf_baseline.joblib');assert np.array_equal(model['thresholds'],np.full(5,.5))
        yc,pc=probability(model,calibration)
        calibration_scores={str(t):metrics(yc,(pc>=t).astype(np.int8)) for t in [.5,.475,.45]}
        # A higher threshold wins exact ties. Only calibration labels participate in selection.
        chosen=max([.5,.475,.45],key=lambda t:(calibration_scores[str(t)]['micro_f1'],t))
        evaluations={};all_threshold_diagnostics={}
        for n,rows in sets.items():
            gold,prob=probability(model,rows)
            options={str(t):metrics(gold,(prob>=t).astype(np.int8)) for t in [.5,.475,.45]}
            evaluations[n]={'fixed_0.5':options['0.5'],'calibration_selected':options[str(chosen)]}
            all_threshold_diagnostics[n]=options
        results[name]={'fit':measured,'threshold_chosen_on_calibration':chosen,'calibration_scores':calibration_scores,'evaluation':evaluations,'exploratory_fixed_threshold_scores_not_for_selection':all_threshold_diagnostics}
        (P/'summary.json').write_text(json.dumps({'protocol':{'train_rows':6498,'seed':42,'word_ngrams':[1,3],'char_analyzer':'char_wb','char_ngrams':[3,5],'calibration':{'public_rows':728,'v3_validation_rows':600,'total':1328,'note':'V3 validation is now calibration, not an independent held-out metric.'},'threshold_candidates':[.5,.475,.45],'threshold_selection':'Max Micro-F1 on calibration1328 only, ties prefer higher. No test selection.','official_warning':'Organizer208 has conflicts and train/calibration overlaps. Preserved raw; not an independent final holdout.'},'runs':results},ensure_ascii=False,indent=2)+'\n')
        print('DONE',name,'threshold',chosen,{n:round(e['fixed_0.5']['micro_f1'],6) for n,e in evaluations.items()},flush=True)
    lines=['# Revision3: размерность и небольшое снижение порога','',
    'Одинаковый train6498, word(1,3),char_wb(3,5),sublinear_tf=True,SGD без изменений,seed42. Меняются только max_features. 46k взята из A/B без повторного fit. Все модели сохраняют пороги0.5; альтернативный порог применяется только для диагностической оценки и выбирается по calibration1328=public728+v3validation600.','',
    '| Лимит | Проверка | F1 при0.5 | Порог по calibration | F1 с ним | P с ним | R с ним | FP0.5→cal | FN0.5→cal |','|---|---|---:|---:|---:|---:|---:|---|---|']
    for name,r in results.items():
        for n,e in r['evaluation'].items():
            b=e['fixed_0.5'];a=e['calibration_selected'];lines.append(f"| {name} | {n} | {b['micro_f1']:.6f} | {r['threshold_chosen_on_calibration']} | {a['micro_f1']:.6f} | {a['micro_precision']:.6f} | {a['micro_recall']:.6f} | {b['fp']}→{a['fp']} | {b['fn']}→{a['fn']} |")
    lines+=['','| Лимит | Word/char | Fit,с | Пик RAM,MiB | Модель,MiB |','|---|---|---:|---:|---:|']
    for name,r in results.items():
        f=r['fit'];lines.append(f"| {name} | {f['dimensions']['word']}/{f['dimensions']['char']} | {f['fit_wall_seconds']:.3f} | {f['fit_peak_process_rss_mib']:.1f} | {f['artifact_bytes']/1024**2:.3f} |")
    lines+=['','Разницы относительно46k и полные Macro-F1/P/R/FP/FN сохранены в JSON. Порог не подбирался по тестам. V3validation использована как calibration, больше не считается независимой проверкой этой серии. Публичные тесты известны, имеют конфликты и пересечения, control20 мал (10сюжетов). Метки авторские; production-веса не заменялись. Время одиночное.']
    (P/'report.md').write_text('\n'.join(lines)+'\n')
    deltas={name:{n:{k:r['evaluation'][n]['fixed_0.5'][k]-results['46k']['evaluation'][n]['fixed_0.5'][k] for k in ['micro_f1','macro_f1','micro_precision','micro_recall','fp','fn']} for n in sets} for name,r in results.items()}
    (P/'deltas_vs_46k.json').write_text(json.dumps(deltas,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
