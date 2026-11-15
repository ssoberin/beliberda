"""Text-only screening and a fixed random sample; never loads model outputs."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

P=Path('output/v3-realism-vs-models')
PATTERN_SOURCE=Path('output/v3-realism-screen/screen.py')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    P.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('historical_realism_screen',PATTERN_SOURCE)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    catalog={};summary={};keys=['repeated_team','repeated_faq','repeated_offer_correction','repeated_price']
    for split in ['train','validation','test','quarantine']:
        path=Path('datasets/own_v3_clean')/('quarantine/excluded_train_79.jsonl' if split=='quarantine' else split+'.jsonl')
        digest=sha(path);raw=[json.loads(s) for s in path.read_text().splitlines() if s.strip()];rs=[]
        for i,r in enumerate(raw):
            a=mod.assess(r);mx=max(a['text_counts'][k] for k in keys)
            a.update(row_idx=i,max_complex_repeats=mx,extreme_flag=bool(mx>=5 or a['text_counts']['price_explanation']))
            rs.append(a)
        assert sha(path)==digest
        catalog[split]=rs
        summary[split]=dict(rows=len(raw),sha256=digest,tiers=dict(collections.Counter(r['tier'] for r in rs)),
                            max_complex_repeats=dict(collections.Counter(r['max_complex_repeats'] for r in rs)),
                            extreme_or_meta=sum(r['extreme_flag'] for r in rs),
                            at_least4_or_meta=sum(bool(r['max_complex_repeats']>=4 or r['text_counts']['price_explanation']) for r in rs),
                            meta_explanation=sum(bool(r['text_counts']['price_explanation']) for r in rs))
    write('screen_catalog.json',catalog);write('screen_summary.json',summary)
    indices=sorted(np.random.default_rng(42).choice(600,60,replace=False).tolist())
    sample=[dict(review_id=j+1,row_idx=i,item_id=catalog['test'][i]['item_id'],description=catalog['test'][i]['description']) for j,i in enumerate(indices)]
    previous=P/'blind_sample_inputs.json'
    if previous.exists():assert json.loads(previous.read_text())==sample
    write('blind_sample_inputs.json',sample)
    write('screen_manifest.json',dict(pattern_source_sha256=sha(PATTERN_SOURCE),no_model_predictions_loaded=True,
                                     no_training=True,no_dataset_edits=True,sample_seed=42,sample_size=60))
    print('Prepared historical screen for current splits and the fixed 60-row sample.')

if __name__=='__main__':main()
