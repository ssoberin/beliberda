"""Public gold audit and blind human-review sampling; no model loading or training."""
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

SOURCE=Path('/srv/Hakaton/results/baseline_fp_audit')
OUT=Path('/srv/Hakaton/results/data_quality_audit')
IDS=(102,103,104,105,106)
SOURCE_MC_ID=101  # public/README.md: the entire public test originates from 101.
SEED=20261115

def norm(text):
    return ' '.join(text.casefold().split())

def key(row):
    return (norm(row['description']),SOURCE_MC_ID)

def metrics(rows):
    y=np.asarray([[int(mid in r['gold_mcId']) for mid in IDS] for r in rows],dtype=int).reshape(-1,5)
    pred=np.asarray([[int(mid in r['predicted_mcId']) for mid in IDS] for r in rows],dtype=int).reshape(-1,5)
    return {'n':len(rows),'positive_labels':int(y.sum()),
            'precision':round(precision_score(y,pred,average='micro',zero_division=0),6),
            'recall':round(recall_score(y,pred,average='micro',zero_division=0),6),
            'micro_f1':round(f1_score(y,pred,average='micro',zero_division=0),6),
            'tp':int((y&pred).sum()),'fp':int(((1-y)&pred).sum()),
            'fn':int((y&(1-pred)).sum())}

def theoretical_ceiling(groups):
    """Exact optimum F1 over arbitrary deterministic predictions per (text, source) key.

    A group/category has size n and k positives. Predicting it adds TP=k and
    predicted positives=n. For a candidate F1=f, the decision adds (2*k-f*n)
    to 2*TP-f*(gold positives + predicted positives). Choose it iff positive.
    Binary search f; final decision evaluates actual integral TP and predicted
    positives. Ties do not affect F1. This optimizes *micro* F1 on these rows.
    """
    p=sum(len(set(r['gold_mcId']) & set(IDS)) for g in groups for r in g)
    if p==0:return {'micro_f1':1.0,'tp':0,'fp':0,'fn':0,'predicted_positives':0,'gold_positives':0}
    blocks=[(len(g),sum(mid in r['gold_mcId'] for r in g),mid)
            for g in groups for mid in IDS]
    lo,hi=0.0,1.0
    for _ in range(80):
        f=(lo+hi)/2
        value=sum(max(0,2*k-f*n) for n,k,_ in blocks)-f*p
        if value>=0:lo=f
        else:hi=f
    chosen=[(n,k,mid) for n,k,mid in blocks if 2*k-lo*n>1e-11]
    tp=sum(k for n,k,_ in chosen)
    positives=sum(n for n,k,_ in chosen)
    return {'micro_f1':round(2*tp/(p+positives),8),'tp':tp,
            'fp':positives-tp,'fn':p-tp,
            'predicted_positives':positives,'gold_positives':p,
            'groups':len(groups),'decision_blocks':len(blocks)}

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    rows=json.loads((SOURCE/'public_predictions.json').read_text(encoding='utf-8'))
    errors=json.loads((SOURCE/'public_errors.json').read_text(encoding='utf-8'))
    assert len(rows)==208 and [r['index'] for r in rows]==list(range(208))
    assert len(errors)==126 and collections.Counter(x['kind'] for x in errors)=={'FP':104,'FN':22}
    assert all(set(r['gold_mcId']+r['predicted_mcId'])<=set(IDS) for r in rows)
    assert all(all(float(v)==.5 for v in r['thresholds'].values()) for r in rows)
    OUT.mkdir(parents=True,exist_ok=True)
    public=OUT/'annotation';private=OUT/'private'
    public.mkdir(exist_ok=True);private.mkdir(exist_ok=True)
    public.chmod(0o750)
    private.chmod(0o700)

    grouped=collections.defaultdict(list)
    for row in rows:grouped[key(row)].append(row)
    conflict={k for k,g in grouped.items()
              if len({tuple(sorted(r['gold_mcId'])) for r in g})>1}
    indices_conflict={r['index'] for k in conflict for r in grouped[k]}
    all_conflict=[r for r in rows if r['index'] in indices_conflict]
    other=[r for r in rows if r['index'] not in indices_conflict]
    assert len(conflict)==13 and len(all_conflict)==56
    summary={'source_mcId':101,'source_mcId_provenance':'datasets/public/README.md; not a parquet column',
             'normalization':'casefold + whitespace collapse, description only',
             'rows':len(rows),'unique_keys':len(grouped),
             'conflicting_groups':len(conflict),
             'conflicting_records':len(all_conflict),
             'conflicting_positive_labels':sum(len(r['gold_mcId']) for r in all_conflict),
             'varying_label_assignments':sum(
                sum(len({int(mid in r['gold_mcId']) for r in grouped[k]})>1 for mid in IDS)
                for k in conflict),
             'metrics':{'all':metrics(rows),'nonconflicting':metrics(other),
                        'conflicting':metrics(all_conflict)},
             'deterministic_ceiling':theoretical_ceiling(list(grouped.values())),
             'sampling':{}}
    assert summary['metrics']['all']['fp']==104 and summary['metrics']['all']['fn']==22

    # 56 conflict rows, up to 50 *other* error rows, remainder random clean.
    error_indices={e['index'] for e in errors}-indices_conflict
    error_rows=[r for r in other if r['index'] in error_indices]
    assert len(error_rows)<=50
    rng=random.Random(SEED)
    rng.shuffle(error_rows)
    selected_errors=error_rows[:50]
    selected_error_indices={r['index'] for r in selected_errors}
    clean=[r for r in other if r['index'] not in error_indices]
    rng.shuffle(clean)
    selected_clean=clean[:150-len(all_conflict)-len(selected_errors)]
    selected_indices=indices_conflict|selected_error_indices|{r['index'] for r in selected_clean}
    assert len(selected_indices)==150 and not (selected_error_indices & indices_conflict)
    chosen_keys={key(rows[i]) for i in selected_indices}
    ordered=sorted(chosen_keys,key=lambda k:(hashlib.sha256(
        (str(k[1])+'\0'+k[0]).encode()).hexdigest(),k[0]))
    blind=[];reference=[]
    for k in ordered:
        g=grouped[k]
        ids=[r['index'] for r in g]
        selected=[i for i in ids if i in selected_indices]
        item_id='public-101-'+hashlib.sha256(k[0].encode('utf-8')).hexdigest()[:16]
        blind.append({'item_id':item_id,'description':g[0]['description'],
                      'source_mcId':101,'human_labels':None,'human_comment':'',
                      'original_indices':ids,'selected_indices':selected})
        reference.append({'item_id':item_id,'original_indices':ids,'selected_indices':selected,
                          'original_rows':[{'index':r['index'],'gold_mcId':r['gold_mcId'],
                                            'predicted_mcId':r['predicted_mcId'],
                                            'probabilities':r['probabilities'],
                                            'thresholds':r['thresholds']} for r in g]})
    # A blind annotator only gets annotation/public_blind.jsonl, never private/.
    with (public/'public_blind.jsonl').open('w',encoding='utf-8') as f:
        for r in blind:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    (public/'public_blind.jsonl').chmod(0o640)
    dump(private/'public_reference.json',reference)
    (private/'public_reference.json').chmod(0o600)
    conflict_groups=[{'item_id':'public-101-'+hashlib.sha256(k[0].encode()).hexdigest()[:16],
                      'indices':[r['index'] for r in grouped[k]],
                      'variants':[list(v) for v in sorted({tuple(sorted(r['gold_mcId'])) for r in grouped[k]})],
                      'positive_label_assignments':sum(len(r['gold_mcId']) for r in grouped[k])}
                     for k in sorted(conflict)]
    dump(private/'conflict_groups.json',conflict_groups)
    (private/'conflict_groups.json').chmod(0o600)
    summary['sampling']={'seed':SEED,'selected_source_rows':150,
        'conflict_rows':len(all_conflict),'other_error_rows':len(selected_errors),
        'clean_random_rows':len(selected_clean),'unique_annotation_tasks':len(blind),
        'conflict_tasks':len(conflict),'selected_keys':len(chosen_keys),
        'deduplicated_source_rows':150-len(blind),
        'linked_original_rows_all_in_selected_groups':sum(len(grouped[k]) for k in chosen_keys)}
    dump(OUT/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
