"""Score completed blind annotation against frozen baseline (selection-biased sample)."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score

IDS=(102,103,104,105,106)
ROOT=Path('/srv/Hakaton/results/data_quality_audit')

def read_jsonl(path):
    return [json.loads(s) for s in path.open(encoding='utf-8') if s.strip()]

def calculate(annotations,reference):
    by_id={r['item_id']:r for r in annotations}
    if len(by_id)!=len(annotations):raise ValueError('duplicate item_id')
    ref_ids={r['item_id'] for r in reference}
    if set(by_id)!=ref_ids:raise ValueError('missing or unexpected item_id')
    y=[];pred=[];original=[];details=[]
    for item in reference:
        item_id=item['item_id'];annotation=by_id[item_id]
        labels=annotation.get('human_labels')
        if not isinstance(labels,list) or any(type(mid)!=int or mid not in IDS for mid in labels) or len(labels)!=len(set(labels)):
            raise ValueError(f'{item_id}: human_labels must be a completed unique list of IDs 102–106 (empty [] allowed)')
        if annotation.get('source_mcId')!=101:raise ValueError(f'{item_id}: unexpected source_mcId')
        if sorted(annotation.get('original_indices',[]))!=sorted(item['original_indices']):
            raise ValueError(f'{item_id}: original indices changed')
        for source in item['original_rows']:
            if source['index'] not in item['selected_indices']:continue
            y.append([int(mid in labels) for mid in IDS])
            pred.append([int(mid in source['predicted_mcId']) for mid in IDS])
            original.append([int(mid in source['gold_mcId']) for mid in IDS])
            details.append({'item_id':item_id,'source_index':source['index'],
                            'human_labels':sorted(labels),'original_labels':source['gold_mcId'],
                            'predicted_labels':source['predicted_mcId']})
    y,pred,original=(np.asarray(v,dtype=int) for v in (y,pred,original))
    def score(target):
        return {'n':len(target),'gold_positives':int(target.sum()),
                'precision':round(precision_score(target,pred,average='micro',zero_division=0),6),
                'recall':round(recall_score(target,pred,average='micro',zero_division=0),6),
                'micro_f1':round(f1_score(target,pred,average='micro',zero_division=0),6),
                'fp':int(((pred==1)&(target==0)).sum()),
                'fn':int(((pred==0)&(target==1)).sum())}
    return {'warning':'Error-enriched, duplicated-source-row sample; NOT an independent test or population estimate.',
            'selected_source_rows':len(y),'unique_human_tasks':len(annotations),
            'baseline_vs_human':score(y),'baseline_vs_original_gold_same_rows':score(original),
            'changed_label_assignments':int((y!=original).sum()),'details':details}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('completed_jsonl',type=Path,help='Copy of public_blind.jsonl with human_labels filled in')
    parser.add_argument('--output',type=Path,default=ROOT/'private/human_score.json')
    args=parser.parse_args()
    ref=json.loads((ROOT/'private/public_reference.json').read_text(encoding='utf-8'))
    result=calculate(read_jsonl(args.completed_jsonl),ref)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    args.output.chmod(0o600)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
