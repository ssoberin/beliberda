"""Inference-only public coverage audit, no fitting or dataset edits."""
import collections
import hashlib
import json
import re
from pathlib import Path
import duckdb
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from avito_compare.scripts.prepare_dataset import normalize
from avito_compare.scripts.evaluate_v2_experiment import IDS,metrics
OUT=Path('output/public-test-coverage-audit')
BASE=Path('output/ngram-comparison/splits')
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
train=read(BASE/'train.jsonl');val=read(BASE/'validation.jsonl');cal=read(BASE/'calibration.jsonl')
labels={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,'Электрика':105,'Поклейка обоев и малярные работы':106}
raw=duckdb.connect().execute('select title,description,categories from read_parquet(?)',['datasets/public/public-test-dataset.parquet']).fetchall()
test=[dict(title=t,description=d,labels=sorted(labels[x] for x in cats),mcId=101,row_index=i) for i,(t,d,cats) in enumerate(raw)]
modelpath=Path('output/tfidf-feature-experiments/char_wb/tfidf_baseline.joblib')
m=joblib.load(modelpath);pipe=m['pipeline'];vec=pipe.named_steps['vectorizer'];word=dict(vec.transformer_list)['word']
assert tuple(m['ids'])==IDS and np.array_equal(m['thresholds'],np.full(5,.5))
tokens=word.build_tokenizer();pre=word.build_preprocessor()
def tok(text):return tokens(pre(text[:6000]))
trainwords=collections.Counter(w for r in train for w in set(tok(r['description'])))
trainbigrams={b for r in train for b in zip(tok(r['description']),tok(r['description'])[1:])}
trainkeys={normalize(r['description']) for r in train};allkeys={normalize(r['description']) for r in train+cal+val}
labelsets=collections.defaultdict(set)
for r in test:labelsets[normalize(r['description'])].add(tuple(r['labels']))
conflicts={k for k,v in labelsets.items() if len(v)>1}
X=vec.transform([r['description'][:6000] for r in train])
allresults={};rowresults={}
for name,rows in [('train',train),('validation',val),('test',test)]:
    desc=[r['description'][:6000] for r in rows]
    prob=pipe.predict_proba(desc);pred=(prob>=m['thresholds']).astype(np.int8)
    gold=np.asarray([[int(c in r['labels']) for c in IDS] for r in rows],dtype=np.int8)
    for i,r in enumerate(rows):
        if r.get('mcId',101) in IDS:
            j=IDS.index(r['mcId']);pred[i,j]=0;gold[i,j]=0
    xx=X if name=='train' else vec.transform(desc)
    sims=cosine_similarity(xx,X)
    if name=='train':np.fill_diagonal(sims,-1)
    idx=np.argmax(sims,axis=1);nearest=sims[np.arange(len(rows)),idx]
    details=[]
    for i,r in enumerate(rows):
        t=tok(r['description']);b=list(zip(t,t[1:]));key=normalize(r['description'])
        details.append(dict(index=i,title=r.get('title',''),description=r['description'],gold=[IDS[j] for j in range(5) if gold[i,j]],predicted=[IDS[j] for j in range(5) if pred[i,j]],probabilities=prob[i].tolist(),fp=int(((gold[i]==0)&(pred[i]==1)).sum()),fn=int(((gold[i]==1)&(pred[i]==0)).sum()),length=len(r['description']),unknown_word_fraction=sum(w not in trainwords for w in t)/max(1,len(t)),unknown_bigram_fraction=sum(x not in trainbigrams for x in b)/max(1,len(b)),unseen_words=sorted(set(t)-trainwords.keys()),nearest_similarity=float(nearest[i]),nearest_train_index=int(idx[i]),nearest_train_text=train[idx[i]]['description'],nearest_train_labels=train[idx[i]]['labels'],exact_train_overlap=key in trainkeys,any_public_overlap=key in allkeys,conflicting_test_labels=key in conflicts))
    slices={'all':np.ones(len(rows),dtype=bool)}
    for a,b in [(0,.5),(.5,.8),(.8,.95),(.95,1.00001)]:slices[f'nearest_cosine_{a}_{b}']=(nearest>=a)&(nearest<b)
    slices['has_near_train_ge_0.8']=nearest>=.8
    slices['has_near_train_ge_0.95']=nearest>=.95
    if name=='test':
        slices.update(no_label_conflicts=np.array([not r['conflicting_test_labels'] for r in details]),exact_train_overlap=np.array([r['exact_train_overlap'] for r in details]),no_exact_train_overlap=np.array([not r['exact_train_overlap'] for r in details]))
        seen=set();keep=[]
        for r in details:
            k=normalize(r['description']);yes=not r['conflicting_test_labels'] and not r['any_public_overlap'] and k not in seen
            keep.append(yes);seen.add(k)
        slices['clean_unique_114']=np.array(keep)
    score={k:metrics(gold[mask],pred[mask]) for k,mask in slices.items() if mask.any()}
    summary={f:dict(zip(['p10','p50','p90'],np.percentile([r[f] for r in details],[10,50,90]).tolist())) for f in ['length','unknown_word_fraction','unknown_bigram_fraction','nearest_similarity']}
    combos=collections.Counter(tuple(r['labels']) for r in rows)
    traincombos={tuple(r['labels']) for r in train}
    allresults[name]={'metrics':score,'distributions':summary,'label_counts':{str(c):int(gold[:,j].sum()) for j,c in enumerate(IDS)},'empty_gold':int((gold.sum(1)==0).sum()),'unseen_label_combination_rows':sum(v for k,v in combos.items() if k not in traincombos),'label_combinations':{str(k):v for k,v in combos.items()}}
    rowresults[name]=details
    print(name,json.dumps(allresults[name],ensure_ascii=False),flush=True)
    if name=='test':
        for r in sorted(details,key=lambda r:r['fp']+r['fn'],reverse=True)[:24]:
            print('ERROR',r['index'],r['title'],'gold',r['gold'],'pred',r['predicted'],'sim',round(r['nearest_similarity'],3),'conflict',r['conflicting_test_labels'],'TEXT',r['description'][:900].replace('\n',' '),'NEAREST',r['nearest_train_text'][:250].replace('\n',' '),flush=True)
(OUT/'summary.json').write_text(json.dumps({'model_sha256':hashlib.sha256(modelpath.read_bytes()).hexdigest(),'protocol':'No fitting; fixed saved public-only word(1,3)/char_wb46k SGD, thresholds .5; description only; cosine uses existing TF-IDF; train nearest excludes self.','sets':allresults},ensure_ascii=False,indent=2)+'\n')
(OUT/'examples.json').write_text(json.dumps(rowresults,ensure_ascii=False,indent=2)+'\n')
