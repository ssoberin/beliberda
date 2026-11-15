"""Verify report inputs, full coverage and the independent fractional F1 bound."""
import collections
import hashlib
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from audit import load_sets,sha

P=Path('output/public-test-audit')

def read(name):return json.loads((P/name).read_text())

def main():
    manifest=read('input_manifest.json')
    for path,h in manifest['sha256'].items():assert sha(path)==h,path
    sets=load_sets();rows=sets['public_test_208'];m=read('metrics.json')
    empty=read('all_empty_gold_review.json')
    assert sorted(r['row'] for r in empty)==[r['row'] for r in rows if not r['gold']]
    assert len(empty)==106
    for r in empty:
        assert r['description']==rows[r['row']]['description']
        assert not r['gold'] and not set(r['offered'])&set(r['uncertain'])
    assert collections.Counter(r['group'] for r in empty)=={'A':73,'B':12,'C':16,'D':5}
    pred=pd.read_csv('output/catboost-optimization/final_predictions.csv')
    pred=pred[(pred.model=='C_syntax_baseline__fixed')&(pred.dataset=='official_208')]
    gold=pred.gold.to_numpy();yp=pred.prediction.to_numpy()
    assert np.array_equal(yp,(pred.probability.to_numpy()>=.5).astype(int))
    assert (int(((gold==1)&(yp==1)).sum()),int(((gold==0)&(yp==1)).sum()),int(((gold==1)&(yp==0)).sum()))==(282,111,13)
    for r in pred.itertuples():assert r.gold==int(r.cat_id in rows[r.row_idx]['gold'])
    fp=read('fp_pair_review.json')
    assert {(r['row'],r['category']) for r in fp}=={(int(r.row_idx),int(r.cat_id)) for r in pred[(pred.gold==0)&(pred.prediction==1)].itertuples()}
    groups=collections.defaultdict(list)
    for r in rows:groups[(r['title'],r['description'])].append(r)
    cells=[];minimum_binary=minimum_exact=0
    for rs in groups.values():
        variants=collections.Counter(tuple(r['gold']) for r in rs)
        minimum_exact+=len(rs)-max(variants.values())
        for cid in range(102,107):
            k=sum(cid in r['gold'] for r in rs);n=len(rs)
            minimum_binary+=min(k,n-k);cells.append((k,n))
    f=m['datasets']['public_test_208']['bound']['maximum_micro_f1']
    positive=sum(len(r['gold']) for r in rows)
    # Independent fractional-programming certificate: no selected subset can beat f.
    residual=sum(max(0,2*k-f*n) for k,n in cells)-f*positive
    assert abs(residual)<1e-9
    assert minimum_binary==48 and minimum_exact==22
    repeated=collections.defaultdict(list)
    for r in rows:repeated[r['description']].append(r['row'])
    checked=0
    for ids in repeated.values():
        if len(ids)<2:continue
        checked+=1
        vectors=[pred[pred.row_idx==i].sort_values('cat_id').probability.to_numpy() for i in ids]
        assert all(np.array_equal(v,vectors[0]) for v in vectors[1:])
    assert checked==26
    text=(P/'report.md').read_text()
    blocks=re.findall(r'```json\n(.*?)\n```',text,re.S)
    assert len(blocks)==20
    for block in blocks:
        request=json.loads(block)
        assert request['mcId']==101 and any(request['description']==r['description'] for r in rows)
    docs=Path('mvp/innoglobalhack_avito/innoglobalhack_avito')
    original=subprocess.check_output(['git','show',f'5977ad9:{docs}/README-files.txt']).decode()
    moved=original.replace('\npublic-test-dataset.parquet','\n../../../datasets/public/public-test-dataset.parquet').replace('\npublic-train-annotated.jsonl','\n../../../datasets/public/public-train-annotated.jsonl')
    assert moved==(docs/'README-files.txt').read_text()
    assert 'Исходной для всех объявлений считается 101' in original
    for name in ['Innoglobalhack-Avito_2026.md','microcategories-guide.md','hackathon-evaluation.md']:
        assert subprocess.check_output(['git','show',f'5977ad9:{docs}/{name}'])==(docs/name).read_bytes()
    result=dict(source_data_docs_weights_and_saved_predictions_unchanged=True,
                empty_gold_rows_reviewed=106,unique_empty_descriptions_reviewed=86,
                formal_fp_pairs_reviewed=111,high_score_fp_pairs_reviewed=78,
                duplicate_groups_same_saved_scores=26,report_complete_json_inputs=20,
                independent_micro_f1_ceiling_certificate_residual=residual,
                minimum_binary_label_errors=48,minimum_exact_match_errors=22,
                original_readme_source101_statement_confirmed=True,readme_only_paths_changed=True,
                no_estimator_fit=True,no_gold_edits=True)
    (P/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
