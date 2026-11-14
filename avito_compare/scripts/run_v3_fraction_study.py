"""Fixed public train + nested 0/20/60/100% v3/authored training additions."""
import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import joblib
import numpy as np

from .evaluate_v2_experiment import IDS, metrics
from .feature_experiments import rss_mib, verify_reference
from .prepare_dataset import normalize


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def adapt(row,dataset):
    offered=row.get('offered_microcategories',row.get('labels'))
    source=row.get('mcId',101)
    expected={e['mcId'] for e in row['found_microcategories']} & set(IDS) if 'found_microcategories' in row else set(row['labels'])
    labels=sorted(set(offered)&set(IDS))
    assert set(labels)-{source}==expected
    return {'description':row['description'],'mcId':source,'labels':labels,
            'evaluation_labels':sorted(expected),'dataset':dataset,
            'group_id':row.get('group_id',hashlib.sha256(normalize(row['description']).encode()).hexdigest()),
            'item_id':row.get('item_id')}


def shingles(text):
    tokens=__import__('re').findall(r'[а-яёa-z0-9]+',text.casefold())
    return {tuple(tokens[i:i+5]) for i in range(max(1,len(tokens)-4))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v3',type=Path,required=True)
    p.add_argument('--public',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('output/v3-fraction-study'))
    args=p.parse_args()
    if args.output.exists():raise ValueError('Fresh output directory required')
    args.output.mkdir(parents=True);(args.output/'logs').mkdir()
    newdir=Path('datasets/context_pairs_100')
    paths=[args.v3/f'{name}.jsonl' for name in ['train','validation','test','challenge']]+[newdir/'data.jsonl',newdir/'test_20.jsonl']
    hashes={str(path):sha(path) for path in paths}
    public={name:[adapt(r,'public_'+name) for r in read(args.public/f'{name}.jsonl')] for name in ['train','calibration','validation']}
    assert len(public['train'])==3398
    v3={name:[adapt(r,'v3_'+name) for r in read(args.v3/f'{name}.jsonl')] for name in ['train','validation','test','challenge']}
    authored=[adapt(r,'authored_train') for r in read(newdir/'data.jsonl')]
    controls=[adapt(r,'authored_test') for r in read(newdir/'test_20.jsonl')]
    assert len(v3['train'])==3000 and len(authored)==100 and len(controls)==20
    # All novel source descriptions must be outside every pre-existing corpus.
    novel=authored+controls;newkeys={normalize(r['description']) for r in novel}
    assert len(newkeys)==120
    previous=[r for rows in public.values() for r in rows]+[r for rows in v3.values() for r in rows]
    for path in Path('datasets/hard_dataset/v2').glob('*.jsonl'):previous.extend(read(path))
    assert not newkeys & {normalize(r['description']) for r in previous},'New/old exact overlap'
    newsets=[shingles(r['description']) for r in novel];inverted=collections.defaultdict(set)
    for i,values in enumerate(newsets):
        for value in values:inverted[value].add(i)
    near=[]
    for row in previous:
        values=shingles(row['description'])
        candidates=set().union(*(inverted[s] for s in values if s in inverted))
        for i in candidates:
            other=newsets[i]
            if min(len(values),len(other))<.8*max(len(values),len(other)):continue
            similarity=len(values&other)/len(values|other)
            if similarity>=.8:near.append({'new':novel[i]['item_id'],'old':row.get('item_id'),'jaccard':similarity})
    assert not near,'New texts too similar to old corpus'
    (newdir/'quality.json').write_text(json.dumps({'exact_overlap_with_prior_corpora':0,'near_copy_pairs_jaccard_5gram_ge_0_8':len(near),'new_rows_checked':120,'prior_rows_checked':len(previous),'semantic_independence_proven':False},indent=2)+'\n')
    rng=np.random.default_rng(42);order=rng.permutation(3000)
    groups=collections.defaultdict(list)
    for r in authored:groups[r['group_id']].append(r)
    raw_auth={r['item_id']:r for r in read(newdir/'data.jsonl')}
    by_topic=collections.defaultdict(list)
    for group,rows in groups.items():
        assert len(rows)==2
        by_topic[raw_auth[rows[0]['item_id']]['topic']].append(group)
    for names in by_topic.values():rng.shuffle(names)
    labels={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,'Электрика':105,'Поклейка обоев и малярные работы':106}
    parquet=Path('datasets/public/public-test-dataset.parquet')
    raw=duckdb.connect().execute('SELECT description,categories FROM read_parquet(?)',[str(parquet)]).fetchall()
    official=[{'description':text,'mcId':101,'evaluation_labels':sorted(labels[c] for c in cats)} for text,cats in raw]
    labelsets=collections.defaultdict(set)
    for r in official:labelsets[normalize(r['description'])].add(tuple(r['evaluation_labels']))
    conflicts={k for k,v in labelsets.items() if len(v)>1}
    seenkeys={normalize(r['description']) for r in previous}
    clean=list({normalize(r['description']):r for r in official if normalize(r['description']) not in conflicts and normalize(r['description']) not in seenkeys}.values())
    evaluations={'official_public_test_208':official,'official_clean_unique':clean,
                 'public_validation_729':public['validation'],'v3_test_600':v3['test'],
                 'new_control_20':controls,'v3_challenge_144':v3['challenge']}
    validation=public['validation']+v3['test']
    heldout=public['calibration']+validation+v3['validation']+v3['challenge']+controls
    heldkeys={normalize(r['description']) for r in heldout}
    protocol={'public_fit_rows':3398,'public_rows_never_removed':True,'synthetic_pool':3100,
              'synthetic_composition':{'v3_train':3000,'authored_train':100},
              'seed':42,'thresholds':[.5]*5,'word_ngrams':[1,3],'char_analyzer':'char_wb','char_ngrams':[3,5],
              'max_features':{'word':18000,'char':28000},'sublinear_tf':True,
              'sgd':'Original SGD parameters unchanged, random_state42',
              'sampling':'Nested random v3 prefix; topic-proportional whole authored pairs; no test samples in fitting.',
              'source_sha256':hashes,'official_test_warning':'Organizer208 has13 conflicting groups; original public train overlaps20 rows. Preserved per user instruction; clean diagnostic reported separately.',
              'near_copy_quality':json.loads((newdir/'quality.json').read_text())}
    results={};reference=None;old_ids=set()
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    for percent in [0,20,60,100]:
        addition=[v3['train'][int(i)] for i in order[:3000*percent//100]]
        for names in by_topic.values():
            for group in names[:len(names)*percent//100]:addition.extend(groups[group])
        assert len(addition)==3100*percent//100
        current_ids={(r['dataset'],r['item_id']) for r in addition};assert old_ids<=current_ids;old_ids=current_ids
        train=public['train']+addition
        assert not {normalize(r['description']) for r in train}&heldkeys
        assert len({normalize(r['description']) for r in train})==len(train)
        counts={str(mid):sum(mid in r['labels'] for r in addition) for mid in IDS}
        if percent:assert all(0<value<len(addition) for value in counts.values())
        folder=args.output/f'p{percent:03d}';data=folder/'splits';data.mkdir(parents=True)
        for name,rows in [('train',train),('calibration',public['calibration']),('validation',validation)]:
            (data/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        command=[sys.executable,'-m','avito_compare.scripts.measure_training','--data-dir',str(data),'--output-dir',str(folder/'model'),
                 '--word-ngram-max','3','--char-analyzer','char_wb','--exclude-source-category']
        print(f'START {percent}%: public3398 + synthetic{len(addition)}, per-class={counts}',flush=True)
        with (args.output/'logs'/f'p{percent:03d}.log').open('w') as log:
            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);start=time.monotonic()
            while process.poll() is None:
                if rss_mib(process.pid)>2048 or time.monotonic()-start>180:
                    process.kill();process.wait();raise RuntimeError('Resource budget exceeded; stopped')
                time.sleep(.1)
        if process.returncode:raise RuntimeError(f'Training failed: inspect p{percent:03d}.log')
        measured=json.loads((folder/'model/measurement.json').read_text())
        if reference:
            for key in ['vectorizer_parameters','estimator_parameters','thresholds','versions']:
                assert measured[key]==reference[key],key
        else:reference=measured
        model=joblib.load(folder/'model/tfidf_baseline.joblib');scores={}
        for name,rows in evaluations.items():
            gold=np.asarray([[int(c in row['evaluation_labels']) for c in IDS] for row in rows],dtype=np.int8)
            prediction=(model['pipeline'].predict_proba([r['description'][:6000] for r in rows])>=model['thresholds']).astype(np.int8)
            for i,r in enumerate(rows):
                if r['mcId'] in IDS:prediction[i,IDS.index(r['mcId'])]=0
            scores[name]=metrics(gold,prediction)
            if name=='new_control_20':
                grouped=collections.defaultdict(list)
                for i,r in enumerate(rows):grouped[r['group_id']].append(i)
                scores[name]['fully_correct_pairs']=sum(np.array_equal(gold[indices],prediction[indices]) for indices in grouped.values())
                scores[name]['total_pairs']=len(grouped)
        results[str(percent)]={'public_rows':3398,'synthetic_rows':len(addition),'synthetic_positive_counts':counts,
                              'fit':measured,'evaluation':scores,'train_row_ids':list(sorted(current_ids))}
        (args.output/'summary.json').write_text(json.dumps({'protocol':protocol,'runs':results},ensure_ascii=False,indent=2)+'\n')
        print(f"DONE {percent}%: official={scores['official_public_test_208']['micro_f1']:.6f}; v3={scores['v3_test_600']['micro_f1']:.6f}; new20={scores['new_control_20']['micro_f1']:.6f}; fit={measured['fit_wall_seconds']:.2f}s",flush=True)
    assert hashes=={str(path):sha(path) for path in paths}
    lines=['# Public train +0/20/60/100% synthetic train','',
           'Public train3398 unchanged in all runs. Synthetic pool=v3train3000+new authored100. Samples nested, seed42, authored pairs kept together and proportionally sampled by topic. Fixed model word(1,3)+char_wb(3,5),18k+28k,SGD unchanged,thresholds0.5. No test data in fit.','',
           '| Synthetic fraction | Added rows | Public208 F1 | Public validation729 F1 | V3test600 F1 | New20 F1 | New20 exact pairs | Fit s | Model MiB |',
           '|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for fraction,r in results.items():
        e=r['evaluation'];m=r['fit']
        lines.append(f"| {fraction}% | {r['synthetic_rows']} | {e['official_public_test_208']['micro_f1']:.6f} | {e['public_validation_729']['micro_f1']:.6f} | {e['v3_test_600']['micro_f1']:.6f} | {e['new_control_20']['micro_f1']:.6f} | {e['new_control_20']['fully_correct_pairs']}/10 | {m['fit_wall_seconds']:.2f} | {m['artifact_bytes']/1024**2:.2f} |")
    lines+=['','## All measured metrics','', '| Fraction | Set | Micro-F1 | Macro-F1 | P | R | FP | FN | Exact-set |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for fraction,r in results.items():
        for name,m in r['evaluation'].items():
            lines.append(f"| {fraction}% | {name} | {m['micro_f1']:.6f} | {m['macro_f1']:.6f} | {m['micro_precision']:.6f} | {m['micro_recall']:.6f} | {m['fp']} | {m['fn']} | {m['exact_set_accuracy']:.6f} |")
    lines+=['','Official208 includes conflicts and training overlap; clean_unique is a smaller diagnostic subset. Public729 is validation, not final test. Synthetic and new20 are authored, not independent human gold. New20 contains10scenarios, not20independent stories. Ratios were predeclared; no threshold or hyperparameter selection. Timings are single fits. Production model unchanged.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
