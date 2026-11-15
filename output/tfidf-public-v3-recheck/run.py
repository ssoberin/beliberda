"""Refit the existing standalone TF-IDF baseline; frozen thresholds, separate artifacts."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
    os.environ[key]='1'
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import sys
import time
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts.qwen3_v3_experiment import build_tfidf_pipeline
from avito_compare.scripts.qwen3_category_aware_experiment import load_public_rows,load_v3_rows_full,_rows_xy_for_ids,SCOPE_102_106
from avito_compare.scripts.qwen3_embedding_experiment import load_official_parquet
from avito_compare.scripts.eval_fixed_cat_aware import compute_detailed_metrics

P=Path('output/tfidf-public-v3-recheck')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(text):return ' '.join(text.casefold().split())
def write(name,value):(P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def main():
    assert not (P/'metrics.json').exists(),'Completed run exists; no repeated fitting after test inspection'
    paths={
        'pub_train':'output/ngram-comparison/splits/train.jsonl',
        'v3_train':'datasets/own_v3_clean/train.jsonl',
        'pub_validation':'output/ngram-comparison/splits/validation.jsonl',
        'v3_test':'datasets/own_v3_clean/test.jsonl',
        'official_208':'datasets/public/public-test-dataset.parquet',
        'existing_tfidf':'output/syntax-aware-experiment/models/tfidf_baseline.joblib',
        'existing_c_syntax':'output/syntax-aware-experiment/models/catboost_C_syntax.cbm',
        'saved_predictions':'output/catboost-optimization/final_predictions.csv',
    }
    hashes={path:sha(path) for path in paths.values()}
    public=load_public_rows(Path(paths['pub_train']));v3=load_v3_rows_full(Path(paths['v3_train']))
    train=public+v3;assert len(public)==3398 and len(v3)==2921
    y_train,_=_rows_xy_for_ids(train,SCOPE_102_106,use_eval=False)
    model=build_tfidf_pipeline();params=model.get_params(deep=True)
    selected={k:v for k,v in params.items() if k.startswith(('vectorizer__word__','vectorizer__char__','classifier__estimator__')) and isinstance(v,(str,int,float,bool,type(None),tuple))}
    config=dict(train_counts={'public_train':len(public),'v3_clean_train':len(v3),'total':len(train)},
                data_paths=paths,sha256=hashes,parameters=selected,thresholds=[.5]*5,ids=list(SCOPE_102_106),
                training_target='offered categories 102-106; evaluate additional categories, exclude input mcId',
                seed=42,no_calibration_used=True,no_hard_negatives_200=True,
                packages={k:importlib.metadata.version(k) for k in ['scikit-learn','numpy','scipy','joblib','pandas']})
    write('config.json',config)
    print('FROZEN CONFIG: public3398 + V3clean2921, word(1,3)18k + char_wb(3,5)28k, alpha1e-4, thresholds0.5',flush=True)
    start=time.perf_counter();model.fit([r['description'] for r in train],y_train);fit_seconds=time.perf_counter()-start
    model_path=P/'models/tfidf_sgd.joblib';joblib.dump(model,model_path)
    _,_,official=load_official_parquet(Path(paths['official_208']))
    datasets={'official_208':official,'v3_test':load_v3_rows_full(Path(paths['v3_test'])),'pub_validation':load_public_rows(Path(paths['pub_validation']))}
    old_model=joblib.load(paths['existing_tfidf'])
    c_saved=pd.read_csv(paths['saved_predictions']);c_saved=c_saved[c_saved.model=='C_syntax_baseline__fixed']
    metrics={};records=[];candidates=[];overlaps={};diffs={};all_rows={}
    train_keys={norm(r['description']) for r in train}
    for name,rows in datasets.items():
        texts=[r['description'] for r in rows];gold,sources=_rows_xy_for_ids(rows,SCOPE_102_106,use_eval=True)
        start=time.perf_counter();prob=model.predict_proba(texts);seconds=time.perf_counter()-start
        m,yp=compute_detailed_metrics(gold,prob,np.full(5,.5),sources,SCOPE_102_106)
        m.update(m['confusion']);m['prediction_seconds']=seconds;metrics[name]=m
        diffs[name]=float(np.abs(old_model.predict_proba(texts)-prob).max())
        overlap=[i for i,r in enumerate(rows) if norm(r['description']) in train_keys]
        overlaps[name]={'exact_train_overlap_rows':overlap,'count':len(overlap)}
        if name=='v3_test':
            assert not overlap
            assert not ({r['group_id'] for r in v3}&{r['group_id'] for r in rows})
        if name=='official_208':
            assert len(overlap)==20
            mask=np.array([i not in overlap for i in range(len(rows))])
            clean,_=compute_detailed_metrics(gold[mask],prob[mask],np.full(5,.5),sources[mask],SCOPE_102_106)
            metrics['official_no_exact_train_overlap_188']=clean
        for i,r in enumerate(rows):
            base=c_saved[(c_saved.dataset==name)&(c_saved.row_idx==i)]
            c_pred=sorted(base[base.prediction==1].cat_id.astype(int).tolist())
            gold_ids=[c for j,c in enumerate(SCOPE_102_106) if gold[i,j]]
            tf_pred=[c for j,c in enumerate(SCOPE_102_106) if yp[i,j]]
            c_gold=sorted(base[base.gold==1].cat_id.astype(int).tolist());assert c_gold==gold_ids
            for j,c in enumerate(SCOPE_102_106):records.append(dict(dataset=name,row_idx=i,source_mcId=int(sources[i]),cat_id=c,gold=int(gold[i,j]),probability=float(prob[i,j]),prediction=int(yp[i,j])))
            if name!='pub_validation' and tf_pred!=gold_ids and c_pred==gold_ids:
                candidates.append(dict(dataset=name,row_idx=i,description=r['description'],mcId=int(sources[i]),gold=gold_ids,tfidf_prediction=tf_pred,c_syntax_prediction=c_pred,
                                       tfidf_errors=len(set(tf_pred)^set(gold_ids)),tfidf_scores={str(c):float(prob[i,j]) for j,c in enumerate(SCOPE_102_106)}))
        all_rows[name]=rows
        print(name,{k:m[k] for k in ['rows','micro_f1','micro_precision','micro_recall','fp','fn','exact_match']},flush=True)
    assert all(sha(path)==h for path,h in hashes.items())
    vectorizers=dict(model.named_steps['vectorizer'].transformer_list)
    result=dict(metrics=metrics,training_seconds=fit_seconds,model_bytes=model_path.stat().st_size,
                features={k:len(v.vocabulary_) for k,v in vectorizers.items()},
                max_probability_difference_vs_previous_same_train_tfidf=diffs,
                overlaps=overlaps,max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                model_sha256=sha(model_path),original_sources_and_baselines_unchanged=True)
    write('metrics.json',result);write('example_candidates.json',candidates);write('evaluation_rows.json',all_rows)
    pd.DataFrame(records).to_csv(P/'predictions.csv',index=False)
    print('Fit seconds',fit_seconds,'model bytes',model_path.stat().st_size,'previous TF-IDF probability differences',diffs,flush=True)
    print('Candidate counts',pd.Series([r['dataset'] for r in candidates]).value_counts().to_dict(),flush=True)

if __name__=='__main__':main()
