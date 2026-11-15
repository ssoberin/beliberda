"""Candidate verification ablation; saved models, test data and API are read-only."""
import argparse
import asyncio
import collections
import hashlib
import json
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'segmentation'))
from segmentation_benchmark import IDS, NAMES, units  # noqa: E402
from verifier import Verifier  # noqa: E402

ROOT = Path('/srv/Hakaton')
REPO = ROOT / 'innopolis_2026'
OUT = ROOT / 'results/verification_v1'
CACHE = OUT / 'qwen_cache.jsonl'
# Chosen previously on public calibration, before this verifier experiment.
SEGMENT_THRESHOLDS = np.asarray([.84, .82, .78, .90, .68])


def jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def parquet(path):
    import pyarrow.parquet as pq
    return [{'title': row['title'], 'description': row['description'], 'mcId': 101,
             'labels': [NAMES[name] for name in row['categories']]}
            for row in pq.read_table(path).to_pylist()]


def fixed_sample(rows, n):
    return sorted(rows, key=lambda row: hashlib.sha256(row['description'].encode()).hexdigest())[:n]


def gold(rows):
    return np.asarray([[int(mid in (set(row['labels']) if 'labels' in row else
                                    {item['mcId'] for item in row['found_microcategories']})
                          and mid != row.get('mcId', 101)) for mid in IDS] for row in rows])


def prepare(rows, model):
    start = perf_counter()
    descriptions = [row['description'] for row in rows]
    full = model['pipeline'].predict_proba(descriptions)
    full_seconds = perf_counter()-start
    start = perf_counter()
    blocks = [units(text) for text in descriptions]
    flat = [part for block in blocks for part in block]
    probs = np.concatenate([model['pipeline'].predict_proba(flat[i:i+512])
                            for i in range(0, len(flat), 512)])
    seg = np.zeros_like(full)
    best = []
    cursor = 0
    for i, block in enumerate(blocks):
        chunk = probs[cursor:cursor+len(block)]
        cursor += len(block)
        index = np.argmax(chunk, axis=0)
        seg[i] = chunk[index, np.arange(len(IDS))]
        best.append(index.tolist())
    seg_seconds = perf_counter()-start
    baseline = full >= .5
    extra = seg >= SEGMENT_THRESHOLDS
    for i,row in enumerate(rows):
        if row.get('mcId',101) in IDS:
            j=IDS.index(row['mcId']); baseline[i,j]=False; extra[i,j]=False
    union = baseline|extra
    jobs=[]
    locations=[]
    for i,row in enumerate(rows):
        for j,mid in enumerate(IDS):
            if not union[i,j]:
                continue
            block=blocks[i]; pos=best[i][j]
            job=(row['description'],mid,block[pos],block[pos-1] if pos else '',
                 block[pos+1] if pos+1<len(block) else '')
            jobs.append(job)
            locations.append((i,j))
    return baseline,union,jobs,locations,full,seg,{'full':round(full_seconds,3),
                                                    'segments':round(seg_seconds,3)}


def measure(yt,yp):
    return {'n':len(yt),'precision':round(precision_score(yt,yp,average='micro',zero_division=0),4),
            'recall':round(recall_score(yt,yp,average='micro',zero_division=0),4),
            'micro_f1':round(f1_score(yt,yp,average='micro',zero_division=0),4),
            'macro_f1':round(f1_score(yt,yp,average='macro',zero_division=0),4),
            'fp':int(((yp==1)&(yt==0)).sum()),'fn':int(((yp==0)&(yt==1)).sum()),
            'empty':int((yp.sum(axis=1)==0).sum()),
            'exact':round(float(np.all(yp==yt,axis=1).mean()),4)}


MANUAL = [
    ('Трубы и сантехнику не монтируем; это работа другого мастера.',104,False),
    ('Монтируем трубы, устанавливаем смесители и раковину.',104,True),
    ('Электрику выполняет другой мастер, мы только кладём плитку.',105,False),
    ('Электрику выполняет другой мастер, мы только кладём плитку.',103,True),
    ('Выполняем ремонт квартиры под ключ. Смета и материалы по договору.',103,False),
    ('Ставим раковины и смесители. Трубы не монтируем.',104,True),
    ('Плитку не кладём, но монтируем керамогранит.',103,True),
    ('Выполняем плиточные работы? Нет. Устанавливаем раковины.',103,False),
    ('Выполняем плиточные работы? Нет. Устанавливаем раковины.',104,True),
    ('Заказчик хотел монтаж розеток, но мы эту услугу не оказываем.',105,False),
    ('Электромонтажные работы можно обсудить после осмотра.',105,False),
]


async def diagnostic(model, verifier):
    jobs=[]
    for description,mid,_ in MANUAL:
        block=units(description)
        probs=model['pipeline'].predict_proba(block)
        pos=int(np.argmax(probs[:,IDS.index(mid)]))
        jobs.append((description,mid,block[pos],block[pos-1] if pos else '',
                     block[pos+1] if pos+1<len(block) else ''))
    results=await verifier.run(jobs)
    table=[]
    for (description,mid,expected),record in zip(MANUAL,results):
        table.append({'description':description,'mcId':mid,'gold_manual':expected,
                      'decision':record['accepted'],'rejection':record['rejection'],
                      'answer':record['answer'],'seconds':record['seconds']})
    path=OUT/'manual_diagnostic_v2.json'
    path.write_text(json.dumps(table,ensure_ascii=False,indent=2),encoding='utf-8')
    for item in table:
        print('manual',item['mcId'],item['gold_manual'],item['decision'],item['rejection'],
              item['description'],item['answer'],flush=True)
    print('manual correct',sum(item['gold_manual']==item['decision'] for item in table),
          '/',len(table),'network calls',verifier.network_calls, 'saved',path)


async def dataset(name,rows,model,verifier):
    baseline,union,jobs,locations,full,seg,infer_time=prepare(rows,model)
    before=verifier.network_calls
    answers=await verifier.run(jobs)
    rejected=np.zeros_like(union)
    rejections=collections.Counter()
    ad_seconds=[0.0]*len(rows)
    baseline_seconds=[0.0]*len(rows)
    source_calls=[0]*len(rows)
    verification_cases=[]
    y=gold(rows)
    for (i,j),record in zip(locations,answers):
        source_calls[i]+=1
        ad_seconds[i]=max(ad_seconds[i],record['seconds'])
        if baseline[i,j]:
            baseline_seconds[i]=max(baseline_seconds[i],record['seconds'])
        if record['accepted']:
            rejected[i,j]=True
        else:
            rejections[record['rejection']]+=1
        if bool(y[i,j]) != bool(record['accepted']):
            verification_cases.append({'index':i,'mcId':IDS[j],
                'description':rows[i]['description'],'gold':bool(y[i,j]),
                'baseline':bool(baseline[i,j]),'segment_candidate':bool(union[i,j] and not baseline[i,j]),
                'full_score':round(float(full[i,j]),3),'segment_score':round(float(seg[i,j]),3),
                'answer':record['answer'],'rejection':record['rejection']})
    variants={'baseline':baseline,'baseline_qwen':baseline & rejected,
              'baseline_segmentation':union,'baseline_segmentation_qwen':union & rejected}
    metrics={k:measure(y,yp) for k,yp in variants.items()}
    per_ad={k: {'mean_seconds':round(float(np.mean(baseline_seconds if k=='baseline_qwen' else ad_seconds))+
                                       (infer_time['full']+ (infer_time['segments'] if 'segmentation' in k else 0))/len(rows),3),
                'median_seconds':round(float(np.median(baseline_seconds if k=='baseline_qwen' else ad_seconds)),3),
                'p95_seconds':round(float(np.percentile(baseline_seconds if k=='baseline_qwen' else ad_seconds,95)),3)}
            for k in ('baseline_qwen','baseline_segmentation_qwen')}
    return {'metrics':metrics,'inference_total_seconds':infer_time,
            'logical_candidate_calls':len(jobs),
            'baseline_candidate_calls':int(baseline.sum()),
            'union_candidate_calls':int(union.sum()),
            'actual_new_network_calls':verifier.network_calls-before,
            'candidate_count_per_ad_mean':round(float(np.mean(source_calls)),3),
            'qwen_per_ad_latency':per_ad,
            'request_rejections':dict(rejections),
            'verification_mistakes':len(verification_cases)},verification_cases


async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=('diagnostic','benchmark'),required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    model=joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')  # Trusted local artifact.
    assert tuple(model['ids'])==IDS and np.allclose(model['thresholds'],.5)
    verifier=Verifier(CACHE,concurrency=4)
    if args.stage=='diagnostic':
        await diagnostic(model,verifier)
        return
    if not (OUT/'manual_diagnostic_v2.json').exists():
        raise RuntimeError('Run --stage diagnostic and inspect its output first')
    sets={
        'development_calibration_80':fixed_sample(jsonl(ROOT/'data/processed/calibration.jsonl'),80),
        'public_validation_160':fixed_sample(jsonl(ROOT/'data/processed/validation.jsonl'),160),
        'synthetic_300':parquet(REPO/'datasets/public/public-test-synthetic-300.parquet'),
        'public_test_diagnostic_208':parquet(REPO/'datasets/public/public-test-dataset.parquet'),
    }
    output={'prompt_version':'verification-v2','segmentation_thresholds':SEGMENT_THRESHOLDS.tolist(),
            'routing':'all baseline positives and additional calibrated unit-max positives; no test tuning',
            'sets':{}}
    for name,rows in sets.items():
        score,cases=await dataset(name,rows,model,verifier)
        output['sets'][name]=score
        (OUT/f'{name}_cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
        (OUT/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
        print(name,score['metrics'],'calls',score['logical_candidate_calls'],
              'new network',score['actual_new_network_calls'],'rejections',score['request_rejections'],flush=True)
    print('saved',OUT/'metrics.json')


if __name__=='__main__':
    asyncio.run(main())
