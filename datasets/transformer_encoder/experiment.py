"""Isolated five-label encoder experiment. Invoke audit, probe, train, evaluate in order."""
import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

ROOT = Path('/srv/Hakaton')
DATA = ROOT / 'data/processed'
REPO = ROOT / 'innopolis_2026/datasets'
OUT = ROOT / 'results/transformer_tiny2_v1'
MODEL = 'cointegrated/rubert-tiny2'
IDS = (102,103,104,105,106)
LENGTH = 768
BATCH = 2
ACCUM = 8
EPOCHS = 3
SEED = 42

def rows_json(path):
    with Path(path).open(encoding='utf8') as f:
        return [json.loads(s) for s in f if s.strip()]

def rows_parquet(path):
    return pq.read_table(path).to_pylist()

def labels(row):
    if 'labels' in row:
        return set(row['labels'])
    if 'categories' in row:
        mapping={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,
                 'Электрика':105,'Поклейка обоев и малярные работы':106}
        # Existing public mapping is authoritative (inspect unexpected labels, don't silently drop).
        return {mapping[name] for name in row['categories']}
    return {r['mcId'] for r in row['found_microcategories']}

def targets(rows):
    return np.asarray([[int(mid in labels(r) and mid!=r.get('mcId',101)) for mid in IDS]
                       for r in rows],dtype=np.int64)

def encode(tokenizer,text):
    ids=tokenizer.encode(text,add_special_tokens=False,truncation=False)
    room=LENGTH-2
    if len(ids)>room:
        ids=ids[:room//2]+ids[-(room-room//2):]
    return [tokenizer.cls_token_id]+ids+[tokenizer.sep_token_id]

class TextSet(Dataset):
    def __init__(self,rows,tok):
        self.ids=[encode(tok,r['description']) for r in rows]
        self.y=targets(rows).astype(np.float32)
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):return self.ids[i],self.y[i]

def collate(items):
    length=max(len(x[0]) for x in items)
    ids=torch.zeros((len(items),length),dtype=torch.long)
    mask=torch.zeros_like(ids)
    for i,(x,_) in enumerate(items):
        ids[i,:len(x)]=torch.tensor(x)
        mask[i,:len(x)]=1
    return ids,mask,torch.tensor(np.stack([x[1] for x in items]))

class Classifier(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(MODEL) if pretrained else AutoModel.from_config(
            __import__('transformers').AutoConfig.from_pretrained(MODEL))
        self.dropout=nn.Dropout(.1)
        self.head=nn.Linear(self.encoder.config.hidden_size,5)
    def forward(self,ids,mask):
        h=self.encoder(input_ids=ids,attention_mask=mask).last_hidden_state
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        return self.head(self.dropout(pooled))

def audit(tok):
    sets={k:rows_json(DATA/f'{k}.jsonl') for k in ('train','calibration','validation')}
    normalized={k:{' '.join(r['description'].casefold().split()) for r in v} for k,v in sets.items()}
    report={'model':MODEL,'max_length':LENGTH,'long_strategy':'first 383 and last 383 tokens + specials',
            'seed':SEED,'batch':BATCH,'accumulation':ACCUM,'epochs':EPOCHS,
            'learning_rate':2e-5,'splits':{},'overlaps':{}}
    for k,v in sets.items():
        lengths=np.array([len(tok.encode(r['description'],add_special_tokens=True,
                                         truncation=False)) for r in v])
        report['splits'][k]={'rows':len(v),'unique_normalized':len(normalized[k]),
                            'percentiles':{str(n):int(np.percentile(lengths,n)) for n in (50,75,90,95,99,100)},
                            'truncated':{str(n):{'count':int((lengths>n).sum()),
                                                 'fraction':round(float((lengths>n).mean()),4)}
                                         for n in (512,768,1024)}}
    for a,b in (('train','calibration'),('train','validation'),('calibration','validation')):
        report['overlaps'][a+'_'+b]=len(normalized[a]&normalized[b])
    (OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
    assert all(v==0 for v in report['overlaps'].values())

def capacity_check():
    free,total=torch.cuda.mem_get_info()
    print('VRAM free MiB',free//1048576,'total MiB',total//1048576,flush=True)
    if free<4*1024**3:
        raise RuntimeError('Less than 4 GiB free; stop without touching vLLM')

def step(model,batch,optimizer):
    ids,mask,y=(x.cuda(non_blocking=True) for x in batch)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(ids,mask)
        loss=nn.functional.binary_cross_entropy_with_logits(logits.float(),y)
    (loss/ACCUM).backward()
    return float(loss.item())

def probe(tok):
    capacity_check()
    model=Classifier().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
    # Deliberately worst-case 768-token batch, two texts; backward and optimizer step.
    ids=torch.full((BATCH,LENGTH),tok.unk_token_id,dtype=torch.long)
    ids[:,0]=tok.cls_token_id;ids[:,-1]=tok.sep_token_id
    batch=(ids,torch.ones_like(ids),torch.zeros((BATCH,5)))
    try:
        loss=step(model,batch,optimizer)
        optimizer.step()
        torch.cuda.synchronize()
        peak=torch.cuda.max_memory_allocated()/1048576
        print('probe loss',loss,'peak allocated MiB',round(peak),flush=True)
        (OUT/'probe.json').write_text(json.dumps({'loss':loss,'peak_allocated_mib':round(peak),
                                                  'batch':BATCH,'length':LENGTH}))
    except torch.cuda.OutOfMemoryError:
        print('OOM: stop; no vLLM changes',flush=True)
        raise

def train(tok):
    if not (OUT/'probe.json').exists():raise RuntimeError('probe first')
    capacity_check()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    ds=TextSet(rows_json(DATA/'train.jsonl'),tok)
    loader=DataLoader(ds,batch_size=BATCH,shuffle=True,collate_fn=collate,num_workers=0)
    model=Classifier().cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    progress=[]
    start=time.perf_counter()
    opt.zero_grad(set_to_none=True)
    for epoch in range(EPOCHS):
        losses=[]
        for i,batch in enumerate(loader):
            losses.append(step(model,batch,opt))
            if (i+1)%ACCUM==0 or i+1==len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                opt.step();opt.zero_grad(set_to_none=True)
            if (i+1)%250==0:
                print('epoch',epoch+1,'batch',i+1,'/',len(loader),'loss',round(np.mean(losses[-250:]),4),
                      'elapsed_s',round(time.perf_counter()-start),flush=True)
        progress.append({'epoch':epoch+1,'mean_loss':round(float(np.mean(losses)),5),
                         'elapsed_s':round(time.perf_counter()-start,1)})
        print(progress[-1],flush=True)
    torch.save(model.cpu().state_dict(),OUT/'weights.pt')
    (OUT/'train.json').write_text(json.dumps(progress,indent=2))
    print('saved weights outside git',OUT/'weights.pt',flush=True)

@torch.no_grad()
def predict(model,tok,rows,batch_size=8):
    ds=TextSet(rows,tok)
    loader=DataLoader(ds,batch_size=batch_size,collate_fn=collate,num_workers=0)
    probs=[]
    start=time.perf_counter()
    for ids,mask,_ in loader:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(ids.cuda(),mask.cuda())
        probs.append(torch.sigmoid(z.float()).cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(probs),time.perf_counter()-start

def metrics(y,p):
    d={'precision':round(precision_score(y,p,average='micro',zero_division=0),4),
       'recall':round(recall_score(y,p,average='micro',zero_division=0),4),
       'micro_f1':round(f1_score(y,p,average='micro',zero_division=0),4),
       'fp':int(((p==1)&(y==0)).sum()),'fn':int(((p==0)&(y==1)).sum()),
       'f1_categories':{str(mid):round(f1_score(y[:,i],p[:,i],zero_division=0),4)
                        for i,mid in enumerate(IDS)}}
    return d

def baseline(rows,model):
    start=time.perf_counter()
    pred=model['pipeline'].predict_proba([r['description'] for r in rows])>=model['thresholds']
    for i,r in enumerate(rows):
        if r.get('mcId',101) in IDS:pred[i,IDS.index(r['mcId'])]=False
    return pred,time.perf_counter()-start

def evaluate(tok,stage):
    if not (OUT/'weights.pt').exists():raise RuntimeError('train first')
    if stage=='test' and not (OUT/'validation.json').exists():raise RuntimeError('validation first')
    state=torch.load(OUT/'weights.pt',map_location='cpu',weights_only=True)
    model=Classifier(pretrained=False)
    model.load_state_dict(state)
    model.cuda().eval()
    base=joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')
    assert tuple(base['ids'])==IDS and np.allclose(base['thresholds'],.5)
    if stage=='validation':
        cal=rows_json(DATA/'calibration.jsonl')
        scores,_=predict(model,tok,cal)
        truth=targets(cal)
        # Independent threshold per class on calibration; one-pass classwise F1, deterministic tie break.
        grid=np.arange(.05,.951,.025)
        th=np.asarray([grid[np.argmax([f1_score(truth[:,i],scores[:,i]>=v,zero_division=0)
                                    for v in grid])] for i in range(5)])
        (OUT/'thresholds.json').write_text(json.dumps({'ids':IDS,'thresholds':th.tolist(),
                              'method':'per-class calibration F1; grid .05:.025:.95'},indent=2))
        sets={'calibration':cal,'validation':rows_json(DATA/'validation.jsonl')}
    else:
        th=np.array(json.loads((OUT/'thresholds.json').read_text())['thresholds'])
        sets={'public_test':rows_parquet(REPO/'public/public-test-dataset.parquet'),
              'synthetic_300':rows_parquet(REPO/'public/public-test-synthetic-300.parquet'),
              'hard_v2_test':rows_json(REPO/'hard_dataset/v2/test.jsonl'),
              'postoyanka_test':rows_json(REPO/'postoyanka/splits/test.jsonl')}
    report={'stage':stage,'thresholds':th.tolist(),'sets':{}}
    for name,rows in sets.items():
        y=targets(rows)
        scores,seconds=predict(model,tok,rows)
        pred=scores>=th
        for i,r in enumerate(rows):
            if r.get('mcId',101) in IDS:pred[i,IDS.index(r['mcId'])]=False
        b,bsec=baseline(rows,base)
        mistakes=[]
        for i,r in enumerate(rows):
            if not np.array_equal(y[i],pred[i]):
                mistakes.append({'index':i,'description':r['description'],
                    'gold':[IDS[j] for j in range(5) if y[i,j]],
                    'transformer':[IDS[j] for j in range(5) if pred[i,j]],
                    'tfidf':[IDS[j] for j in range(5) if b[i,j]],
                    'scores':[round(float(v),4) for v in scores[i]]})
        (OUT/f'{name}_errors.json').write_text(json.dumps(mistakes[:100],ensure_ascii=False,indent=2))
        report['sets'][name]={'n':len(rows),'transformer':metrics(y,pred),'baseline':metrics(y,b),
                              'latency_ms_per_ad_transformer':round(seconds*1000/len(rows),2),
                              'latency_ms_per_ad_baseline':round(bsec*1000/len(rows),2),
                              'transformer_errors':len(mistakes)}
        (OUT/f'{stage}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        print(name,json.dumps(report['sets'][name],ensure_ascii=False),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('audit','probe','train','validation','test'))
    args=parser.parse_args()
    os.environ['TOKENIZERS_PARALLELISM']='false'
    OUT.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL)
    if args.stage=='audit':audit(tok)
    elif args.stage=='probe':probe(tok)
    elif args.stage=='train':train(tok)
    else:evaluate(tok,args.stage)

if __name__=='__main__':main()
