"""A/B study of 250 train replacements; fixed data order, parameters and thresholds."""
import collections,hashlib,json,os,re,subprocess,sys,time
from pathlib import Path
import duckdb,joblib,numpy as np
from .run_v3_fraction_study import adapt,read,shingles
from .prepare_dataset import normalize
from .evaluate_v2_experiment import IDS,metrics
from .feature_experiments import rss_mib
ROOT=Path('output/v3-revision3-study')
REV=Path('datasets/own_v3_revision3')
BASE=Path('/tmp/innopolis-dataset-v3-review/datasets/own_v3')
OLD=Path('output/v3-fraction-study/p100')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def score(model,rows):
    gold=np.asarray([[int(c in r['evaluation_labels']) for c in IDS] for r in rows],dtype=np.int8)
    prob=model['pipeline'].predict_proba([r['description'][:6000] for r in rows])
    pred=(prob>=model['thresholds']).astype(np.int8)
    for i,r in enumerate(rows):
        if r['mcId'] in IDS:pred[i,IDS.index(r['mcId'])]=0
    return metrics(gold,pred)
def main():
    if ROOT.exists():raise ValueError('Fresh output required')
    ROOT.mkdir(parents=True)
    manifest=json.loads((REV/'replacement_manifest.json').read_text())
    newrows=read(REV/'replacements_250.jsonl');new={r['replaces_item_id']:adapt(r,'v3_train') for r in newrows}
    oldtrain=read(OLD/'splits/train.jsonl');after=[new.get(r['item_id'],r) if r['dataset']=='v3_train' else r for r in oldtrain]
    assert len(oldtrain)==len(after)==6498
    assert sum(a!=b for a,b in zip(oldtrain,after))==250
    assert collections.Counter((tuple(r['labels']),r['mcId']) for r in oldtrain)==collections.Counter((tuple(r['labels']),r['mcId']) for r in after)
    public={s:read(OLD/f'splits/{s}.jsonl') for s in ['calibration','validation']}
    pubval=[r for r in public['validation'] if r['dataset']=='public_validation']
    sets={'public_validation_729':pubval,**{f'v3_{s}':[adapt(r,f'v3_{s}') for r in read(BASE/f'{s}.jsonl')] for s in ['validation','test','challenge']},'new_control_20':[adapt(r,'authored_test') for r in read(Path('datasets/context_pairs_100/test_20.jsonl'))]}
    held={normalize(r['description']) for rs in sets.values() for r in rs}|{normalize(r['description']) for r in public['calibration']}
    assert not {normalize(r['description']) for r in after}&held
    assert len({normalize(r['description']) for r in after})==6498
    # Exact and near-copy checks against all prior corpora, including replaced originals and public test.
    names={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,'Электрика':105,'Поклейка обоев и малярные работы':106}
    parquet=Path('datasets/public/public-test-dataset.parquet')
    raw=duckdb.connect().execute('select description,categories from read_parquet(?)',[str(parquet)]).fetchall()
    official=[{'description':d,'mcId':101,'evaluation_labels':sorted(names[c] for c in cats)} for d,cats in raw]
    sets['official_public_test_208']=official
    allpublic=read(Path('output/v3-fraction-public-splits/train.jsonl'))+read(Path('output/v3-fraction-public-splits/calibration.jsonl'))+read(Path('output/v3-fraction-public-splits/validation.jsonl'))
    prior=oldtrain+allpublic+[r for rs in sets.values() for r in rs]
    for path in Path('datasets/hard_dataset/v2').glob('*.jsonl'):prior+=read(path)
    allkeys={normalize(r['description']) for r in prior}
    assert not {normalize(r['description']) for r in newrows}&allkeys
    signatures=[shingles(r['description']) for r in newrows];inv=collections.defaultdict(set)
    for i,ss in enumerate(signatures):
        for token in ss:inv[token].add(i)
    near=[]
    for row in prior:
        ss=shingles(row['description']);candidates=set().union(*(inv[x] for x in ss if x in inv))
        for i in candidates:
            other=signatures[i]
            if min(len(ss),len(other))<.8*max(len(ss),len(other)):continue
            j=len(ss&other)/len(ss|other)
            if j>=.8:near.append([newrows[i]['item_id'],row.get('item_id'),j])
    assert not near,near
    labelsets=collections.defaultdict(set)
    for r in official:labelsets[normalize(r['description'])].add(tuple(r['evaluation_labels']))
    conflict={k for k,v in labelsets.items() if len(v)>1};pubkeys={normalize(r['description']) for r in allpublic}
    sets['official_clean_unique_114']=list({normalize(r['description']):r for r in official if normalize(r['description']) not in conflict|pubkeys}.values())
    assert len(sets['official_clean_unique_114'])==114
    def words(rows):return set(w for r in rows for w in re.findall(r'[а-яёa-z]+',r['description'].casefold()))
    vbefore=words([r for r in oldtrain if r['dataset']=='v3_train']);vafter=words([r for r in after if r['dataset']=='v3_train'])
    quality={'exact_new_prior_overlap':0,'near_5gram_jaccard_ge_0_8':near,'checked_new_rows':250,'checked_prior_rows':len(prior),'original_v3_word_types':len(vbefore),'revised_v3_word_types':len(vafter),'added_word_types':sorted(vafter-vbefore),'removed_word_types':sorted(vbefore-vafter),'warning':'No exact/near copy does not prove semantic independence; 250 rows are 50 groups. Public test informed diagnosis, hence is not an untouched final holdout.'}
    (ROOT/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2)+'\n')
    hashed=[REV/'train.jsonl',REV/'replacements_250.jsonl',parquet,*[BASE/f'{s}.jsonl' for s in ['train','validation','test','challenge']]]
    beforehash={str(p):sha(p) for p in hashed}
    results={};reference=None
    for name,train in [('before',oldtrain),('after',after)]:
        folder=ROOT/name;split=folder/'splits';split.mkdir(parents=True)
        for s,rs in [('train',train),*public.items()]:
            (split/f'{s}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rs))
        command=[sys.executable,'-m','avito_compare.scripts.measure_training','--data-dir',str(split),'--output-dir',str(folder/'model'),'--word-ngram-max','3','--char-analyzer','char_wb','--exclude-source-category']
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        print(f'START {name}:6498 training rows',flush=True)
        with (folder/'fit.log').open('w') as log:
            proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while proc.poll() is None:
                if rss_mib(proc.pid)>2048 or time.monotonic()-start>180:
                    proc.kill();proc.wait();raise RuntimeError('Resource guard')
                time.sleep(.1)
        assert proc.returncode==0
        measured=json.loads((folder/'model/measurement.json').read_text())
        if reference:
            for k in ['vectorizer_parameters','estimator_parameters','thresholds','versions']:assert measured[k]==reference[k],k
        else:reference=measured
        model=joblib.load(folder/'model/tfidf_baseline.joblib');assert np.array_equal(model['thresholds'],np.full(5,.5))
        evaluations={n:score(model,rs) for n,rs in sets.items()}
        # Reproduction check for the previously saved baseline on identical held-outs.
        if name=='before':
            old=json.loads(Path('output/v3-fraction-study/summary.json').read_text())['runs']['100']['evaluation']
            for n,k in [('public_validation_729','public_validation_729'),('v3_test','v3_test_600'),('official_public_test_208','official_public_test_208'),('new_control_20','new_control_20')]:
                for metric in ['micro_f1','fp','fn']:assert evaluations[n][metric]==old[k][metric]
        fit_slices={'all_train':score(model,train),'synthetic_train':score(model,[r for r in train if r['dataset']!='public_train'])}
        results[name]={'fit':measured,'evaluation':evaluations,'training_diagnostics_not_holdout':fit_slices}
        print(name,{n:round(v['micro_f1'],6) for n,v in evaluations.items()},flush=True)
        (ROOT/'summary.json').write_text(json.dumps({'protocol':{'train_rows':6498,'public_train':3398,'v3_train':3000,'previous_authored_train':100,'replacements':250,'thresholds':[.5]*5,'seed':42,'same_order':True,'source_and_scope_label_histograms_preserved':True,'source_sha256':beforehash},'runs':results},ensure_ascii=False,indent=2)+'\n')
    assert beforehash=={str(p):sha(p) for p in hashed}
    lines=['# Замена 250 простых текстов v3: сравнение до/после','',
    'Обе модели обучены заново на 6498 строках: public3398 + v3train3000 + прежние авторские100. В v3 заменены250 строк на50 контрастных сюжетов по5версий. Число строк, порядок, source и распределение обучающих/целевых меток102–106 сохранены. Word(1,3), char_wb(3,5),18k+28k,sublinear_tf=True, прежнийSGD,seed42,пороги0.5. Словарь переобучен только на train. Исходные v3 и проверочные части не изменены.','',
    '| Выборка | F1 до | F1 после | Δ, п.п. | P после | R после | FP до→после | FN до→после |',
    '|---|---:|---:|---:|---:|---:|---|---|']
    deltas={}
    for n,b in results['before']['evaluation'].items():
        a=results['after']['evaluation'][n]
        deltas[n]={k:{'absolute':a[k]-b[k],'relative_percent':100*(a[k]-b[k])/b[k] if b[k] else None} for k in ['micro_f1','macro_f1','micro_precision','micro_recall','fp','fn']}
        lines.append(f"| {n} | {b['micro_f1']:.6f} | {a['micro_f1']:.6f} | {100*(a['micro_f1']-b['micro_f1']):+.3f} | {a['micro_precision']:.6f} | {a['micro_recall']:.6f} | {b['fp']}→{a['fp']} | {b['fn']}→{a['fn']} |")
    lines+=['','## Ресурсы и разрыв train/holdout','', '| Модель | Fit, с | Пик RAM, MiB | Размер, MiB | Word/char | Synthetic train F1 |','|---|---:|---:|---:|---|---:|']
    for name,r in results.items():
        f=r['fit'];lines.append(f"| {name} | {f['fit_wall_seconds']:.3f} | {f['fit_peak_process_rss_mib']:.1f} | {f['artifact_bytes']/1024**2:.3f} | {f['dimensions']['word']}/{f['dimensions']['char']} | {r['training_diagnostics_not_holdout']['synthetic_train']['micro_f1']:.6f} |")
    lines+=['',f'Словоформы v3train по одинаковому правилу подсчёта: {len(vbefore)}→{len(vafter)}; новых {len(vafter-vbefore)}, исчезнувших {len(vbefore-vafter)}. Новых точных/близких копий в проверенных старых корпусах не найдено. Это не доказательство семантической независимости.','',
    'Public729 — validation. Official208 имеет известные конфликты и пересечения с train; clean114 — отдельный малый диагностический срез. Проверочные наборы многократно использовались, а public test повлиял на выбор направлений дополнения: прирост не доказывает качество на неизвестном финальном тесте. Новые250 — авторская разметка и самооценка, не независимый gold. Веса production не заменялись. Время измерено однократно.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');(ROOT/'deltas.json').write_text(json.dumps(deltas,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
