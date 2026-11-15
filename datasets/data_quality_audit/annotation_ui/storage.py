"""Blind task projection and per-annotator, atomic JSONL answer storage.

Never imports, opens or references private/public_reference.json.
"""
import fcntl
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASE=Path('/srv/Hakaton/results/data_quality_audit')
TASKS=BASE/'annotation'
ANSWERS=BASE/'human_labels'
LABELS=(102,103,104,105,106)
DATASETS={'public':'public_blind.jsonl','train':'train_review.jsonl'}
ANNOTATORS=('annotator_1','annotator_2')

def _jsonl(path):
    with path.open(encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]

def tasks(dataset, task_dir=TASKS):
    if dataset not in DATASETS:raise ValueError('unknown dataset')
    rows=_jsonl(task_dir/DATASETS[dataset])
    projected=[]
    ids=set()
    for row in rows:
        item_id=row['item_id']
        if item_id in ids:raise ValueError('duplicate task ID')
        ids.add(item_id)
        # Explicit allow-list; do not expose current_labels in train source.
        item={'item_id':item_id,'description':row['description'],
              'source_mcId':row.get('source_mcId',101)}
        if item['source_mcId']!=101:raise ValueError('unexpected source mcId')
        if dataset=='public':
            item['original_indices']=row['original_indices']
            item['selected_indices']=row['selected_indices']
        else:
            item['original_index']=row['original_index']
        projected.append(item)
    return projected

def answer_path(dataset,annotator,answer_dir=ANSWERS):
    if dataset not in DATASETS or annotator not in ANNOTATORS:
        raise ValueError('unknown dataset or annotator')
    return answer_dir/f'{dataset}_{annotator}.jsonl'

def _load(path,valid_ids):
    if not path.exists():return {}
    rows=_jsonl(path)
    found={}
    for row in rows:
        item_id=row.get('item_id')
        if item_id in found or item_id not in valid_ids:
            raise ValueError('duplicate or unknown answer item_id')
        validate_labels(row.get('human_labels'))
        if not isinstance(row.get('human_comment'),str):raise ValueError('invalid comment')
        found[item_id]=row
    return found

def load_answers(dataset,annotator,items,answer_dir=ANSWERS):
    return _load(answer_path(dataset,annotator,answer_dir),{x['item_id'] for x in items})

def validate_labels(labels):
    if not isinstance(labels,list) or len(labels)!=len(set(labels)) or any(
            type(mid)!=int or mid not in LABELS for mid in labels):
        raise ValueError('human_labels must be a unique list of IDs 102–106; [] is valid')

def _write_atomic(path,records):
    path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    path.parent.chmod(0o700)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            os.fchmod(stream.fileno(),0o600)
            for row in records:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name):os.unlink(name)

def save_answer(dataset,annotator,item,labels,comment,items,answer_dir=ANSWERS):
    validate_labels(labels)
    if not isinstance(comment,str) or len(comment)>5000:raise ValueError('comment too long')
    if item['item_id'] not in {x['item_id'] for x in items}:
        raise ValueError('unknown item_id')
    path=answer_path(dataset,annotator,answer_dir)
    answer_dir.mkdir(parents=True,exist_ok=True,mode=0o700)
    answer_dir.chmod(0o700)
    lock=answer_dir/(path.name+'.lock')
    with lock.open('a') as lock_file:
        os.fchmod(lock_file.fileno(),0o600)
        fcntl.flock(lock_file,fcntl.LOCK_EX)
        try:
            current=_load(path,{x['item_id'] for x in items})
            current[item['item_id']]={**item,'human_labels':sorted(labels),
                'human_comment':comment,'saved_at':datetime.now(timezone.utc).isoformat(),
                'annotator':annotator,'dataset':dataset}
            _write_atomic(path,[current[x['item_id']] for x in items if x['item_id'] in current])
        finally:
            fcntl.flock(lock_file,fcntl.LOCK_UN)
    return path

def export_jsonl(dataset,annotator,items,answer_dir=ANSWERS):
    answers=load_answers(dataset,annotator,items,answer_dir)
    records=[answers[x['item_id']] for x in items if x['item_id'] in answers]
    return ''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in records).encode('utf-8')

def completeness(items,answers):
    return {'total':len(items),'saved':len(answers),
            'missing_ids':[x['item_id'] for x in items if x['item_id'] not in answers],
            'empty_without_comment':[
                x['item_id'] for x in items if x['item_id'] in answers and
                not answers[x['item_id']]['human_comment'].strip() and
                answers[x['item_id']]['human_labels']==[]]}
