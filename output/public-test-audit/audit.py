"""Read-only audit of original public data and saved C_syntax outputs.

No estimator fit, training, threshold selection, label modification, or new predictions.
Near-duplicate similarity uses a pre-existing frozen TF-IDF vectorizer.
"""
import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name] = '1'
import collections
import hashlib
import itertools
import json
from pathlib import Path
import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

P = Path('output/public-test-audit')
CAT = {102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',105:'Электрика',106:'Поклейка обоев и малярные работы'}
REV = {v:k for k,v in CAT.items()}
PATHS = {
    'public_raw_10000':Path('datasets/public/public-train-annotated.jsonl'),
    'public_train_3398':Path('output/ngram-comparison/splits/train.jsonl'),
    'public_calibration_728':Path('output/ngram-comparison/splits/calibration.jsonl'),
    'public_validation_729':Path('output/ngram-comparison/splits/validation.jsonl'),
    'public_test_208':Path('datasets/public/public-test-dataset.parquet'),
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def norm(text):
    return ' '.join(text.casefold().split())

def read(name):
    return json.loads((P/name).read_text())

def write(name, value):
    (P/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def load_sets():
    sets={}
    for name,path in PATHS.items():
        if name=='public_test_208':
            raw=duckdb.connect().execute('select title,description,categories from read_parquet(?)',[str(path)]).fetchall()
            rows=[dict(row=i,title=t,description=d,gold=sorted(REV[c] for c in cats)) for i,(t,d,cats) in enumerate(raw)]
        else:
            raw=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            rows=[dict(row=i,title=r.get('title'),description=r['description'],
                       item_id=r.get('item_id'),
                       gold=sorted(set(x['mcId'] for x in r['found_microcategories'])) if 'found_microcategories' in r else r['labels'])
                  for i,r in enumerate(raw)]
        sets[name]=rows
    return sets

def conflict_audit(rows):
    groups=collections.defaultdict(list)
    for r in rows:groups[norm(r['description'])].append(r)
    conflicts=[];binary_min=exact_min=0;cells=[]
    for key,items in groups.items():
        variants=collections.Counter(tuple(r['gold']) for r in items)
        exact_min+=len(items)-max(variants.values())
        for cid in CAT:
            k=sum(cid in r['gold'] for r in items);binary_min+=min(k,len(items)-k);cells.append((k,len(items)))
        if len(variants)<2:continue
        # Unknown source can remove at most one category from a common offered set.
        # Existence of such a common set is a necessary consistency test, not source recovery.
        can_source=False
        for bits in itertools.product([False,True],repeat=5):
            offered={cid for cid,flag in zip(CAT,bits) if flag}
            possible={tuple(sorted(offered-{source})) for source in [101,*CAT]}
            if set(variants)<=possible:can_source=True;break
        sizes={len(v) for v in variants}
        if 0 in sizes and max(sizes)>1:kind='empty_vs_multiple_categories'
        elif 0 in sizes:kind='empty_vs_one_category'
        else:kind='nonempty_subset_difference'
        conflicts.append(dict(rows=[r['row'] for r in items],kind=kind,
                              variants=[dict(gold=list(v),rows=[r['row'] for r in items if tuple(r['gold'])==v]) for v in variants],
                              hypothetical_variable_source_can_explain=can_source,
                              title_description_identical=len({(r['title'],r['description']) for r in items})==1,
                              description=items[0]['description']))
    positive=sum(len(r['gold']) for r in rows);tp=predicted=0;best=(0.,0,0)
    for k,n in sorted(cells,key=lambda cell:cell[0]/cell[1],reverse=True):
        tp+=k;predicted+=n;f1=2*tp/max(1,positive+predicted)
        if f1>best[0]:best=(f1,tp,predicted)
    bound=dict(minimum_binary_errors=binary_min,minimum_exact_errors=exact_min,
               maximum_micro_f1=best[0],oracle_tp=best[1],oracle_fp=best[2]-best[1],oracle_fn=positive-best[1])
    return groups,conflicts,bound

def summary(rows):
    groups,conflicts,bound=conflict_audit(rows)
    lens=np.array([len(r['description']) for r in rows])
    ids=[r.get('item_id') for r in rows if r.get('item_id') is not None]
    return dict(rows=len(rows),unique_descriptions=len(groups),
                excess_duplicates=len(rows)-len(groups),rows_in_duplicate_groups=sum(len(g) for g in groups.values() if len(g)>1),
                conflict_groups=len(conflicts),conflict_rows=sum(len(g['rows']) for g in conflicts),
                empty_gold=sum(not r['gold'] for r in rows),
                cardinality=dict(sorted(collections.Counter(len(r['gold']) for r in rows).items())),
                category_counts={str(c):sum(c in r['gold'] for r in rows) for c in CAT},
                length_chars={**{f'p{p}':float(np.percentile(lens,p)) for p in [10,25,50,75,90,95,99]},'min':int(lens.min()),'max':int(lens.max())},
                descriptions_under20=int((lens<20).sum()),item_id_present=len(ids),unique_item_ids=len(set(ids)),
                item_id_direct_overlap_check_possible=bool(ids),bound=bound)

def main():
    P.mkdir(exist_ok=True,parents=True)
    docs=Path('mvp/innoglobalhack_avito/innoglobalhack_avito')
    protected={str(p):sha(p) for p in [*PATHS.values(),
        *[docs/n for n in ['README-files.txt','Innoglobalhack-Avito_2026.md','microcategories-guide.md','hackathon-evaluation.md']],
        Path('output/syntax-aware-experiment/models/catboost_C_syntax.cbm'),
        Path('output/syntax-aware-experiment/models/tfidf_baseline.joblib'),
        Path('output/catboost-optimization/final_predictions.csv')]}
    sets=load_sets();rows=sets['public_test_208']
    raw_train=[json.loads(line) for line in PATHS['public_raw_10000'].read_text().splitlines() if line.strip()]
    raw_test=duckdb.connect().execute('select title,description,categories from read_parquet(?)',[str(PATHS['public_test_208'])]).fetchall()
    schema_integrity=dict(
        raw_train_unknown_target_ids=sum(x.get('mcId') not in CAT for r in raw_train for x in r['found_microcategories']),
        raw_train_id_label_mismatches=sum(x.get('label')!=CAT.get(x.get('mcId')) for r in raw_train for x in r['found_microcategories']),
        raw_train_duplicate_target_entries=sum(len(r['found_microcategories'])!=len({x.get('mcId') for x in r['found_microcategories']}) for r in raw_train),
        raw_train_null_or_blank_descriptions=sum(not isinstance(r.get('description'),str) or not r['description'].strip() for r in raw_train),
        raw_train_missing_item_id=sum(r.get('item_id') is None for r in raw_train),
        test_null_fields=sum(t is None or d is None or cats is None for t,d,cats in raw_test),
        test_blank_descriptions=sum(not d.strip() for _,d,_ in raw_test),
        test_duplicate_target_entries=sum(len(cats)!=len(set(cats)) for _,_,cats in raw_test),
        test_unknown_target_names=sum(c not in REV for _,_,cats in raw_test for c in cats))
    write('schema_integrity.json',schema_integrity)
    summaries={name:summary(rs) for name,rs in sets.items()}
    conflicts={name:conflict_audit(rs)[1] for name,rs in sets.items()}
    reviews={};empty_table=[]
    for inputs,annotations,empty in [('empty_review_inputs.json','empty_annotations.json',True),('nonempty_fp_review_inputs.json','nonempty_annotations.json',False)]:
        jobs={r['review_id']:r for r in read(inputs)};ann=read(annotations)
        assert {r['review_id'] for r in ann}==set(jobs)
        for a in ann:
            j=jobs[a['review_id']]
            assert not set(a['offered'])&set(a['uncertain'])
            for i in j['rows']:
                assert rows[i]['description']==j['description']
                reviews[i]={**a,'review_source':'empty' if empty else 'nonempty_fp','review_id':a['review_id']}
                if empty and not rows[i]['gold']:
                    empty_table.append({**rows[i],**a,'description_sha256':hashlib.sha256(rows[i]['description'].encode()).hexdigest()})
    assert len(empty_table)==106 and len({r['row'] for r in empty_table})==106
    write('all_empty_gold_review.json',sorted(empty_table,key=lambda r:r['row']))
    empty_stats={k:dict(rows=sum(r['group']==k for r in empty_table),unique_descriptions=len({r['description'] for r in empty_table if r['group']==k})) for k in 'ABCD'}
    saved_predictions=pd.read_csv('output/catboost-optimization/final_predictions.csv')
    pred=saved_predictions[(saved_predictions.model=='C_syntax_baseline__fixed')&(saved_predictions.dataset=='official_208')]
    assert len(pred)==1040 and not pred.duplicated(['row_idx','cat_id']).any()
    decisions={i:pred[pred.row_idx==i].sort_values('cat_id') for i in range(208)}
    fp_records=[]
    for r in pred[(pred.gold==0)&(pred.prediction==1)].itertuples():
        a=reviews[r.row_idx]
        verdict='supported_service_missing_in_gold' if r.cat_id in a['offered'] else ('ambiguous' if r.cat_id in a['uncertain'] else 'model_error')
        fp_records.append(dict(row=int(r.row_idx),category=int(r.cat_id),score=float(r.probability),verdict=verdict,
                               gold=rows[r.row_idx]['gold'],prediction=decisions[r.row_idx].query('prediction == 1').cat_id.astype(int).tolist(),
                               rationale=a['rationale'],description=rows[r.row_idx]['description']))
    assert len(fp_records)==111 and sum(r['score']>.9 for r in fp_records)==78
    write('fp_pair_review.json',fp_records)
    def counts(rs):return dict(collections.Counter(r['verdict'] for r in rs))
    fp_stats={'all':counts(fp_records),'over_0_9':counts([r for r in fp_records if r['score']>.9]),
              'by_category':{str(c):counts([r for r in fp_records if r['category']==c]) for c in CAT},
              'unique_text_category_pairs':counts(list({(r['description'],r['category']):r for r in fp_records}.values()))}
    confirmed_missing=[];ambiguous_missing=[]
    for i,a in reviews.items():
        missing=sorted(set(a['offered'])-set(rows[i]['gold']))
        uncertain=sorted(set(a['uncertain'])-set(rows[i]['gold']))
        if missing:confirmed_missing.append(dict(row=i,missing=missing))
        if uncertain:ambiguous_missing.append(dict(row=i,uncertain=uncertain))
    write('reviewed_label_candidates.json',dict(confirmed_expert_missing=confirmed_missing,ambiguous_missing=ambiguous_missing))
    # Frozen vectorizer only: no fitting on any audit set.
    vectorizer=joblib.load('output/syntax-aware-experiment/models/tfidf_baseline.joblib').named_steps['vectorizer']
    near={};details={}
    for left,right in [('public_test_208','public_raw_10000'),('public_test_208','public_train_3398'),
                       ('public_test_208','public_validation_729'),('public_validation_729','public_train_3398'),
                       ('public_calibration_728','public_train_3398')]:
        leftrows,rightrows=sets[left],sets[right]
        rkeys={norm(r['description']) for r in rightrows}
        unique={}
        for r in rightrows:unique.setdefault(norm(r['description']),r)
        ref=list(unique.values())
        A=vectorizer.transform([r['description'] for r in leftrows]);B=vectorizer.transform([r['description'] for r in ref])
        sim=cosine_similarity(A,B);ix=sim.argmax(1);best=sim[np.arange(len(leftrows)),ix]
        key=left+'__'+right
        records=[dict(left_row=r['row'],right_representative_row=ref[int(j)]['row'],right_item_id=ref[int(j)].get('item_id'),
                      cosine=float(v),exact=norm(r['description']) in rkeys,left_gold=r['gold'],right_gold=ref[int(j)]['gold'],
                      left_description=r['description'],right_description=ref[int(j)]['description']) for r,j,v in zip(leftrows,ix,best)]
        near[key]={'exact_rows':sum(r['exact'] for r in records),'exact_unique_descriptions':len({norm(r['description']) for r in leftrows if norm(r['description']) in rkeys}),
                   'nonexact_near_rows':{str(t):sum(not r['exact'] and r['cosine']>=t for r in records) for t in [.8,.85,.9,.95,.98]},
                   'cosine_percentiles':{str(p):float(np.percentile(best,p)) for p in [10,50,90,95]},
                   'similarity_scope':'existing 18k word(1,3)+28k char_wb(3,5) TF-IDF, fitted previously on original public+V3 train; no fit in audit; cosine is a candidate signal, not semantic identity'}
        details[key]=[{k:v for k,v in r.items() if not k.endswith('_description')} for r in records]
    write('near_duplicate_candidates.json',details)
    slices={}
    for dataset,pair in [('pub_validation','public_validation_729__public_train_3398'),('official_208','public_test_208__public_train_3398')]:
        q=saved_predictions[(saved_predictions.model=='C_syntax_baseline__fixed')&(saved_predictions.dataset==dataset)]
        out={}
        for label,selected in [('near_ge_0.95',[r['left_row'] for r in details[pair] if r['cosine']>=.95]),
                               ('less_than_0.95',[r['left_row'] for r in details[pair] if r['cosine']<.95])]:
            d=q[q.row_idx.isin(selected)]
            tp=int(((d.gold==1)&(d.prediction==1)).sum());fp=int(((d.gold==0)&(d.prediction==1)).sum());fn=int(((d.gold==1)&(d.prediction==0)).sum())
            out[label]=dict(rows=len(selected),tp=tp,fp=fp,fn=fn,f1=2*tp/max(1,2*tp+fp+fn))
        slices[dataset]=out
    write('saved_baseline_similarity_slices.json',slices)
    joined=collections.defaultdict(list)
    for name in ['public_raw_10000','public_test_208']:
        for r in sets[name]:joined[norm(r['description'])].append(dict(dataset=name,row=r['row'],item_id=r.get('item_id'),gold=r['gold']))
    cross=[dict(description=key,records=rs) for key,rs in joined.items()
           if len({r['dataset'] for r in rs})>1 and len({tuple(r['gold']) for r in rs})>1]
    write('cross_dataset_exact_conflicts.json',cross)
    # Hypothesis-specific text markers, used only to describe this audit, never in model inference.
    markers={}
    for name,rs in sets.items():
        markers[name]=dict(common_template_phrase=sum('Строительная бригада выполнит ' in r['description'] and 'Организация работ:' in r['description'] for r in rs),
                           explicit_service_headers=sum(any(s in r['description'].casefold() for s in ['наши услуги','что мы делаем','что выполняем','в работу входит']) for r in rs))
    write('metrics.json',dict(datasets=summaries,conflict_types=dict(collections.Counter(g['kind'] for g in conflicts['public_test_208'])),
                              variable_source_compatible_groups=sum(g['hypothetical_variable_source_can_explain'] for g in conflicts['public_test_208']),
                              variable_source_incompatible_groups=sum(not g['hypothetical_variable_source_can_explain'] for g in conflicts['public_test_208']),
                              empty_review=empty_stats,empty_error_assessments=dict(collections.Counter(r['gold_error_assessment'] for r in empty_table)),
                              fp_review=fp_stats,confirmed_missing_rows_in_reviewed_subset=len(confirmed_missing),
                              confirmed_missing_label_pairs_in_reviewed_subset=sum(len(r['missing']) for r in confirmed_missing),
                              semantic_reviewed_test_rows=len(reviews),near_duplicates=near,text_markers=markers))
    write('all_conflicting_groups.json',conflicts)
    assert all(sha(path)==h for path,h in protected.items())
    write('input_manifest.json',dict(sha256=protected,no_model_fit=True,no_new_predictions=True,gold_changed=False,
                                   vectorizer_fit_in_audit=False,baseline_threshold=.5))
    print(json.dumps(dict(empty_review=empty_stats,fp=fp_stats,near=near,summary={n:{k:r[k] for k in ['rows','unique_descriptions','conflict_groups','conflict_rows','empty_gold','length_chars','item_id_present']} for n,r in summaries.items()}),ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
