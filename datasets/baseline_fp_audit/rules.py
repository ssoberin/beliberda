"""Development-only lexical guard experiment. Never fit/retrain TF-IDF."""
import json
import re
from pathlib import Path

import joblib
import numpy as np

from audit import ROOT,OUT,IDS,TERMS,load_json,load_public,load_synthetic,raw,scores

# Rule hypotheses are tied to absence of concrete category terms, not the gold.
# Include service names/synonyms; conservative: only suppress a candidate if *none*
# of the terms is present anywhere in the full text, including title where provided.
LEXICAL={
    102:r'штукатур|оштукатур|маяк|выравнивани[ея] стен|выравнива[а-я]+ стен|штукатурн',
    103:r'плитк|керамогранит|кафел|мозаик|затирк|облицовк|межплиточн|фартук',
    104:r'сантех|труб|раковин|смесител|водопровод|канализац|унитаз|душев|ванн|санузел|водоснабж',
    105:r'электр|проводк|розетк|светильник|выключател|щит|освещени|люстр|электромонтаж',
    106:r'обо[ий]|покраск|окраск|маляр|шпакл|шпатл|краск|шпатлев|шпаклев',
}

def apply(rows,pred,ids=IDS):
    result=pred.copy()
    for i,r in enumerate(rows):
        text=r['description']
        for j,mid in enumerate(ids):
            if mid not in (104,105):continue
            if result[i,j] and not re.search(LEXICAL[mid],text,re.I):
                result[i,j]=False
    return result

def compare(name,rows,model,fn):
    p,b,y=raw(model,rows)
    proposed=fn(rows,b)
    changed=np.where(b & ~proposed)
    delta={'removed_fp':int(((b==1)&(proposed==0)&(y==0)).sum()),
           'added_fn':int(((b==1)&(proposed==0)&(y==1)).sum()),
           'decisions_changed':len(changed[0])}
    return {'baseline':scores(y,b),'rule':scores(y,proposed),'delta':delta,
            'changes':[{'index':int(i),'mid':IDS[j],'gold':bool(y[i,j]),
                        'score':round(float(p[i,j]),4),
                        'fragment':rows[i]['description'][:500]} for i,j in zip(*changed)][:50]}

def main(stage):
    model=joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')
    assert tuple(model['ids'])==IDS and np.allclose(model['thresholds'],.5)
    if stage=='development':
        cal=load_json(ROOT/'data/processed/calibration.jsonl')
        # Candidate evaluation only on calibration. No test metrics read here.
        report={'calibration':compare('calibration',cal,model,apply)}
        p,b,y=raw(model,cal)
        report['by_category']={str(mid):{'removed_fp':int(((b[:,j]==1)&(apply(cal,b)[:,j]==0)&(y[:,j]==0)).sum()),
                                         'added_fn':int(((b[:,j]==1)&(apply(cal,b)[:,j]==0)&(y[:,j]==1)).sum())}
                               for j,mid in enumerate(IDS)}
        (OUT/'rule_development.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(json.dumps({k:v for k,v in report['calibration'].items() if k!='changes'},ensure_ascii=False))
        print('by_category',report['by_category'])
        return
    decision=json.loads((OUT/'rule_decision.json').read_text())
    if decision['status']!='frozen':raise RuntimeError('Freeze rule before evaluation')
    sets={'validation':load_json(ROOT/'data/processed/validation.jsonl'),
          'public_test':load_public(),'synthetic_300':load_synthetic(),
          'hard_v2_test':load_json(ROOT/'innopolis_2026/datasets/hard_dataset/v2/test.jsonl')}
    report={'rule':'for 104/105 only: no lexical category anchor anywhere in description',
            'decision':decision,'sets':{}}
    for name,rows in sets.items():
        result=compare(name,rows,model,apply)
        report['sets'][name]=result
        (OUT/'rule_holdout.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(name,json.dumps({k:v for k,v in result.items() if k!='changes'},ensure_ascii=False),flush=True)

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2 or sys.argv[1] not in ('development','holdout'):
        raise SystemExit('usage: rules.py development|holdout')
    main(sys.argv[1])
