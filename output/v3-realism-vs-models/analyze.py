"""Join fixed text-realism assessments with previously saved predictions; no fit/inference."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

P=Path('output/v3-realism-vs-models')
IDS=list(range(102,107))
def read(name):return json.loads((P/name).read_text())
def write(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def wilson(k,n):
    if not n:return None
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [c-h,c+h]

def main():
    assert sha(P/'blind_annotations.json')==read('blind_annotations_freeze.json')['sha256']
    catalog=read('screen_catalog.json');rows=catalog['test']
    tf_path=Path('output/tfidf-public-v3-recheck/predictions.csv')
    cb_path=Path('output/catboost-optimization/final_predictions.csv')
    tf=pd.read_csv(tf_path);tf=tf[tf.dataset=='v3_test']
    cb=pd.read_csv(cb_path);cb=cb[(cb.dataset=='v3_test')&(cb.model=='C_syntax_baseline__fixed')]
    def matrix(df,column):return df.pivot(index='row_idx',columns='cat_id',values=column).reindex(index=range(600),columns=IDS).fillna(0).to_numpy().astype(int)
    gold=matrix(tf,'gold');a=matrix(tf,'prediction');b=matrix(cb,'prediction')
    assert np.array_equal(gold,matrix(cb,'gold'))
    assert np.array_equal(gold,np.array([[int(c in r['found']) for c in IDS] for r in rows]))
    def metric(pred,selection):
        yt=gold[selection];yp=pred[selection]
        tp=int(((yt==1)&(yp==1)).sum());fp=int(((yt==0)&(yp==1)).sum());fn=int(((yt==1)&(yp==0)).sum())
        return dict(tp=tp,fp=fp,fn=fn,micro_f1=2*tp/max(1,2*tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn))
    def summarize(indices):
        indices=np.array(indices,dtype=int);n=len(indices)
        ae=(a[indices]!=gold[indices]).any(1);be=(b[indices]!=gold[indices]).any(1)
        aw=int(ae.sum());bw=int(be.sum());fixed=int((ae&~be).sum());broken=int((~ae&be).sum())
        return dict(rows=n,tfidf_wrong_ads=aw,c_syntax_wrong_ads=bw,tfidf_wrong_pct=100*aw/max(1,n),c_syntax_wrong_pct=100*bw/max(1,n),
                    tfidf_wrong_c_syntax_exact_right=fixed,correction_pct_of_all=100*fixed/max(1,n),correction_pct_of_tfidf_errors=100*fixed/max(1,aw),
                    tfidf_exact_right_c_syntax_wrong=broken,both_wrong=int((ae&be).sum()),
                    tfidf=metric(a,indices),c_syntax=metric(b,indices),
                    tfidf_error_rate_wilson95=wilson(aw,n),corrected_fraction_wilson95=wilson(fixed,aw))
    automatic={'all600':list(range(600)),
               'no_high_priority_flag':[r['row_idx'] for r in rows if r['tier']=='no_high_priority_flag'],
               'rewrite_candidate':[r['row_idx'] for r in rows if r['tier']=='rewrite_candidate'],
               'stale_tag_only':[r['row_idx'] for r in rows if r['tier']=='review_stale_tag'],
               'extreme_or_meta':[r['row_idx'] for r in rows if r['extreme_flag']],
               'exclude_extreme':[r['row_idx'] for r in rows if not r['extreme_flag']]}
    sample={r['review_id']:r for r in read('blind_sample_inputs.json')};ann=read('blind_annotations.json')
    assert {r['review_id'] for r in ann}==set(sample)
    manual={k:[sample[r['review_id']]['row_idx'] for r in ann if r['realism'] in chars] for k,chars in [('R','R'),('P','P'),('A','A'),('R_plus_P','RP'),('all60','RPA')]}
    joined=[]
    for r in ann:
        i=sample[r['review_id']]['row_idx'];record=rows[i]
        joined.append({**sample[r['review_id']],**r,'historical_tier':record['tier'],
                       'source_mcId':record['source'],'gold':[c for j,c in enumerate(IDS) if gold[i,j]],
                       'tfidf_prediction':[c for j,c in enumerate(IDS) if a[i,j]],'c_syntax_prediction':[c for j,c in enumerate(IDS) if b[i,j]],
                       'tfidf_error':bool((a[i]!=gold[i]).any()),'c_syntax_error':bool((b[i]!=gold[i]).any())})
    write('manual_review_with_predictions.json',joined)
    result=dict(automatic={k:summarize(v) for k,v in automatic.items()},manual={k:summarize(v) for k,v in manual.items()},
                artificial_sample=dict(count=len(manual['A']),total=60,percent=100*len(manual['A'])/60,wilson95=wilson(len(manual['A']),60)),
                annotations_frozen_before_prediction_join=True,model_scope=IDS,
                sources_sha256={str(p):sha(p) for p in [tf_path,cb_path,Path('datasets/own_v3_clean/test.jsonl'),P/'blind_annotations.json']})
    write('metrics.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
