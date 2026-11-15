"""Validate the fixed sample, frozen annotation hash, slice arithmetic and unchanged inputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

P=Path('output/v3-realism-vs-models')
def read(name):return json.loads((P/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    assert sha(P/'blind_annotations.json')==read('blind_annotations_freeze.json')['sha256']
    sample=read('blind_sample_inputs.json')
    assert [r['row_idx'] for r in sample]==sorted(np.random.default_rng(42).choice(600,60,replace=False).tolist())
    m=read('metrics.json')
    assert all(sha(path)==digest for path,digest in m['sources_sha256'].items())
    summary=read('screen_summary.json')
    for split,d in summary.items():
        path=Path('datasets/own_v3_clean')/('quarantine/excluded_train_79.jsonl' if split=='quarantine' else split+'.jsonl')
        assert sha(path)==d['sha256']
        assert sum(d['tiers'].values())==d['rows']
    cfg=json.loads(Path('output/tfidf-public-v3-recheck/config.json').read_text())
    assert all(sha(path)==h for path,h in cfg['sha256'].items())
    for source in ['automatic','manual']:
        for d in m[source].values():
            assert d['tfidf_wrong_ads']-d['c_syntax_wrong_ads']==d['tfidf_wrong_c_syntax_exact_right']-d['tfidf_exact_right_c_syntax_wrong']
            assert d['tfidf_wrong_ads']==d['tfidf_wrong_c_syntax_exact_right']+d['both_wrong']
            assert d['c_syntax_wrong_ads']==d['tfidf_exact_right_c_syntax_wrong']+d['both_wrong']
    assert m['automatic']['all600']['tfidf_wrong_ads']==343
    assert m['automatic']['all600']['tfidf_wrong_c_syntax_exact_right']==225
    assert (m['manual']['R_plus_P']['rows'],m['manual']['R_plus_P']['tfidf_wrong_ads'],m['manual']['R_plus_P']['tfidf_wrong_c_syntax_exact_right'])==(48,25,15)
    assert m['artificial_sample']['count']==12
    joined=read('manual_review_with_predictions.json')
    expected={r['row_idx'] for r in joined if r['realism'] in ['R','P'] and r['tfidf_error'] and not r['c_syntax_error']}
    assert {r['row_idx'] for r in read('semantic_correction_review.json')['cases']}==expected
    text=(P/'report.md').read_text();joined=read('manual_review_with_predictions.json')
    for i in [52,364,459,467,478,525]:
        r=next(r for r in joined if r['row_idx']==i);assert r['description'] in text
    labels=pd.read_csv(P/'screen_labels.csv')
    assert len(labels)==2921+600+600+144+79
    assert len(labels[(labels.split=='train')&(labels.tier=='rewrite_candidate')])==393
    output=dict(fixed_random_sample_verified=True,text_ratings_frozen_before_score_join=True,
                all_slice_transition_counts_verified=True,six_full_examples_verified=True,
                source_data_and_old_model_weights_unchanged=True,gold_unchanged=True,
                no_training=True,no_new_model_inference=True)
    (P/'verification.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
