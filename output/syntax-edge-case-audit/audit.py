"""Record feature probes and fixed-test coverage; no training or label edits."""
import hashlib
import json
from pathlib import Path
import sys
import collections
import duckdb
import pandas as pd
from natasha import Doc,Segmenter

sys.path.insert(0,str(Path.cwd()))
from avito_compare.scripts.syntax_features import SyntaxAnalyzer,extract_pair_features,SYNTAX_ONLY_FEATURES

P=Path('output/syntax-edge-case-audit')
ART=Path('/home/user/hackaton/innopolis_2026')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(n,value):(P/n).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]

def main():
    paths=[Path('datasets/own_v3_clean')/f'{s}.jsonl' for s in ['train','validation','test']]
    paths += [Path('datasets/public/public-test-dataset.parquet'),Path('datasets/context_pairs_100/test_20.jsonl'),Path('avito_compare/scripts/syntax_features.py'),ART/'output/catboost-optimization/final_predictions.csv']
    hashes={str(p):sha(p) for p in paths}
    probes=[
        ('base','Мы не укладываем плитку.',{'syn_affirmative_offer':0,'syn_has_neg_on_action':1}),
        ('positive','Мы укладываем плитку.',{'syn_affirmative_offer':1,'syn_has_neg_on_action':0}),
        ('price','Мы укладываем плитку, но не бесплатно.',{'syn_affirmative_offer':1}),
        ('partial_explicit','Мы не укладываем напольную плитку, но укладываем плитку на стены.',{'syn_affirmative_offer':1}),
        ('partial_ellipsis','Плитку на пол не укладываем, только на стены.',{'syn_affirmative_offer':1}),
        ('resumed','Раньше мы не укладывали плитку. Теперь укладываем плитку.',{'syn_affirmative_offer':1,'temporal_state':'past_neg_to_now_pos'}),
        ('stopped','Раньше мы укладывали плитку. Теперь плитку не укладываем.',{'syn_affirmative_offer':0,'temporal_state':'past_pos_to_now_neg'}),
        ('not_only','Мы не только укладываем плитку, но и затираем швы.',{'syn_has_not_only':1,'syn_has_neg_on_action':0}),
        ('other_service','Мы укладываем плитку, но не устанавливаем раковины.',{'syn_has_neg_on_action':0}),
        ('other_service_next_sentence','Мы укладываем плитку. Раковины не устанавливаем.',{'syn_has_neg_on_action':0}),
        ('third_party','Плитку укладывает подрядчик заказчика.',{'syn_third_party_actor':1,'syn_affirmative_offer':0}),
        ('own_team','Я сам плитку не укладываю, но наша бригада укладывает плитку по нашему договору.',{'syn_affirmative_offer':1}),
        ('only_floor_no','Плитку на пол не укладываем.',{'syn_affirmative_offer':0}),
    ]
    analyzer=SyntaxAnalyzer();feature_results=[]
    for name,text,expected in probes:
        f=extract_pair_features(analyzer,[text],[0],[0.0],103,len(text))
        failures={k:dict(expected=v,observed=f[k]) for k,v in expected.items() if f[k]!=v}
        feature_results.append(dict(id=name,description=text,target_category=103,
                                    syntax_features={k:f[k] for k in SYNTAX_ONLY_FEATURES},
                                    expected_selected_features=expected,failed_expectations=failures,
                                    scope='feature extractor only; synthetic similarity0.0; no classifier call here'))
    write('feature_probes.json',feature_results)
    same_category={}
    for split in ['train','validation','test']:
        data=rows(Path('datasets/own_v3_clean')/f'{split}.jsonl')
        found=[]
        for i,r in enumerate(data):
            positive={e['mcId'] for e in r['evidence'] if e['status']=='offered'}
            negative={e['mcId'] for e in r['evidence'] if e['status']=='not_offered'}
            if positive&negative:found.append(dict(row_idx=i,categories=sorted(positive&negative)))
        same_category[split]=dict(rows=len(data),same_category_both_evidence_statuses=len(found),matches=found)
    data=rows('datasets/own_v3_clean/test.jsonl')
    saved=pd.read_csv(ART/'output/catboost-optimization/final_predictions.csv')
    saved=saved[(saved.model=='C_syntax_baseline__fixed')&(saved.dataset=='v3_test')]
    segmenter=Segmenter();mixed=[];same_sentence=[]
    for i,r in enumerate(data):
        positive={e['mcId'] for e in r['evidence'] if e['status']=='offered' and e['mcId']!=r['mcId'] and 102<=e['mcId']<=106}
        negative={e['mcId'] for e in r['evidence'] if e['status']=='not_offered' and e['mcId']!=r['mcId'] and 102<=e['mcId']<=106}
        if not positive or not negative:continue
        doc=Doc(r['description']);doc.segment(segmenter)
        for sent in doc.sents:
            pos={e['mcId'] for e in r['evidence'] if e['status']=='offered' and e['mcId'] in positive and e['start']<sent.stop and e['end']>sent.start}
            neg={e['mcId'] for e in r['evidence'] if e['status']=='not_offered' and e['mcId'] in negative and e['start']<sent.stop and e['end']>sent.start}
            if pos and neg:same_sentence.append(dict(row_idx=i,sentence=sent.text,positive=sorted(pos),negative=sorted(neg)))
        p=saved[saved.row_idx==i];prediction=sorted(p[p.prediction==1].cat_id.astype(int).tolist())
        gold=sorted(e['mcId'] for e in r['found_microcategories'] if 102<=e['mcId']<=106)
        mixed.append(dict(row_idx=i,mcId=r['mcId'],description=r['description'],gold=gold,prediction=prediction,
                          positive=sorted(positive),negative=sorted(negative),exact=prediction==gold,
                          missing_positive=sorted(positive-set(prediction)),predicted_negative=sorted(negative&set(prediction)),
                          scores={str(int(x.cat_id)):float(x.probability) for x in p.itertuples()}))
    write('mixed_category_test_rows.json',mixed);write('same_sentence_candidates.json',same_sentence)
    public=[d for (d,) in duckdb.connect().execute('select description from read_parquet(?)',['datasets/public/public-test-dataset.parquet']).fetchall()]
    exact_phrase='Плитку на пол не укладываем, только на стены.'
    public_exact=[i for i,t in enumerate(public) if exact_phrase.casefold() in t.casefold()]
    test_exact=[i for i,r in enumerate(data) if exact_phrase.casefold() in r['description'].casefold()]
    control=[dict(row_idx=i,item_id=r['item_id'],description=r['description'],mcId=r['mcId'],gold=[e['mcId'] for e in r['found_microcategories']])
             for i,r in enumerate(rows('datasets/context_pairs_100/test_20.jsonl')) if r.get('topic')=='partial_refusal']
    write('context20_partial_refusal_cases.json',control)
    summary=dict(main_test_definition='V3 clean test600 and public parquet208, unchanged',same_category_status_coverage=same_category,
                 exact_partial_phrase_matches=dict(public_test=public_exact,v3_test=test_exact),
                 public_partial_floor_wall_analogue='No analogue found by targeted text search and inspection; not an exhaustive semantic proof for all possible paraphrases',
                 mixed_different_target_categories=dict(rows=len(mixed),exact_correct=sum(r['exact'] for r in mixed),
                    exact_wrong=sum(not r['exact'] for r in mixed),missing_positive_ads=sum(bool(r['missing_positive']) for r in mixed),
                    predicted_negative_ads=sum(bool(r['predicted_negative']) for r in mixed),
                    positive_negative_evidence_in_same_segmented_sentence=len(same_sentence),
                    note='Evidence selects a broad slice; separate sentences can still share a chunk. Sentence separation does not guarantee isolation of grammatical features.'),
                 feature_probes=dict(rows=len(feature_results),failed_ids=[r['id'] for r in feature_results if r['failed_expectations']]),
                 separate_context20_partial_rows=len(control),sha256=hashes,no_training=True,no_gold_changes=True)
    assert all(sha(p)==h for p,h in hashes.items())
    write('coverage.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
