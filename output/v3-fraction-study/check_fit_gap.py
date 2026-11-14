"""Inference-only train/heldout comparison; never fits or adjusts thresholds."""
import json
from pathlib import Path
import joblib
import numpy as np
from avito_compare.scripts.evaluate_v2_experiment import IDS, metrics
root=Path('output/v3-fraction-study')
summary=json.loads((root/'summary.json').read_text())
results={}
for percent in [0,20,60,100]:
    folder=root/f'p{percent:03d}'
    model=joblib.load(folder/'model/tfidf_baseline.joblib')
    assert np.array_equal(model['thresholds'],np.full(5,.5))
    rows=[json.loads(line) for line in (folder/'splits/train.jsonl').read_text().splitlines() if line.strip()]
    subsets={'all_train':rows,'public_train':[r for r in rows if r['dataset']=='public_train'],'synthetic_train':[r for r in rows if r['dataset']!='public_train']}
    scores={}
    for name,part in subsets.items():
        if not part:continue
        gold=np.asarray([[int(c in r['evaluation_labels']) for c in IDS] for r in part],dtype=np.int8)
        pred=(model['pipeline'].predict_proba([r['description'][:6000] for r in part])>=model['thresholds']).astype(np.int8)
        for i,r in enumerate(part):
            if r['mcId'] in IDS:pred[i,IDS.index(r['mcId'])]=0
        scores[name]=metrics(gold,pred)
    results[str(percent)]={'train':scores,'heldout':summary['runs'][str(percent)]['evaluation']}
    print(percent,{k:round(v['micro_f1'],6) for k,v in scores.items()},flush=True)
(root/'overfitting_diagnostic.json').write_text(json.dumps({'method':'Inference only, existing fitted models, source-excluded predictions, fixed thresholds .5; training scores do not measure generalization.','runs':results},ensure_ascii=False,indent=2)+'\n')
