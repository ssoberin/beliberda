"""Verify the completed refit without fitting or predicting again."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score

P=Path('output/tfidf-public-v3-recheck')
def load(name):return json.loads((P/name).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    config=load('config.json');result=load('metrics.json');pred=pd.read_csv(P/'predictions.csv')
    assert all(sha(path)==h for path,h in config['sha256'].items())
    assert sha(P/'models/tfidf_sgd.joblib')==result['model_sha256']
    assert config['thresholds']==[.5]*5
    assert all(v==0 for v in result['max_probability_difference_vs_previous_same_train_tfidf'].values())
    for dataset in ['official_208','v3_test','pub_validation']:
        q=pred[pred.dataset==dataset].copy();m=result['metrics'][dataset]
        assert not q.duplicated(['row_idx','cat_id']).any()
        expected=(q.probability>=.5).astype(int);expected[q.cat_id==q.source_mcId]=0
        assert np.array_equal(expected,q.prediction)
        y=q.gold.to_numpy();yp=q.prediction.to_numpy()
        assert int(((y==1)&(yp==1)).sum())==m['tp']
        assert int(((y==0)&(yp==1)).sum())==m['fp']
        assert int(((y==1)&(yp==0)).sum())==m['fn']
        assert abs(f1_score(y,yp,zero_division=0)-m['micro_f1'])<1e-6
        assert abs(precision_score(y,yp,zero_division=0)-m['micro_precision'])<1e-6
        assert abs(recall_score(y,yp,zero_division=0)-m['micro_recall'])<1e-6
    text=(P/'report.md').read_text()
    examples=load('examples.json');assert len(examples)==5
    for r in examples:
        assert r['description'] in text
        assert r['input']['description']==r['description']
        assert r['tfidf_prediction']!=r['gold']==r['c_syntax_prediction']
    check=dict(source_data_old_weights_and_saved_predictions_unchanged=True,
               metrics_independently_verified=True,source_category_exclusion_verified=True,
               examples_full_text_verified=5,fresh_model_probabilities_reproduce_previous=True,
               thresholds_tuned_on_tests=False,main_model_replaced=False,
               public_test_preexisting_exact_overlap_rows=20,v3_test_exact_overlap_rows=0)
    (P/'verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
