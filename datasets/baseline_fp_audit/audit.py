"""Read-only baseline audit; heuristic reason tags are hypotheses, not adjudication."""
import collections
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT=Path('/srv/Hakaton')
REPO=ROOT/'innopolis_2026/datasets'
OUT=ROOT/'results/baseline_fp_audit'
IDS=(102,103,104,105,106)
NAMES={'Штукатурные работы':102,'Плиточные работы':103,'Сантехника':104,
       'Электрика':105,'Поклейка обоев и малярные работы':106}
TERMS={102:r'штукатур|оштукатур|маяк|выравнивани[ея] стен',
       103:r'плитк|керамогранит|кафел|мозаик',
       104:r'сантех|труб|раковин|смесител|водопровод|канализац|унитаз',
       105:r'электр|проводк|розетк|светильник|выключател|щит',
       106:r'обо[ий]|покраск|окраск|маляр|шпакл|шпатл'}
NEG=re.compile(r'\b(?:не|нет|без|отказ|исключа|не\s+бер[её]мся|не\s+делаем)\b',re.I)
OTHER=re.compile(r'друг(?:ой|ие|им)\s+(?:мастер|специалист|бригада)|сторонн(?:ие|ий)|приглаш[а-я]+\s+(?:мастер|специалист)|подрядчик',re.I)
INDIRECT=re.compile(r'например|допустим|если\s+нужн|можно\s+обсудить|по\s+желанию|планиру|материал|смет|проект|консультац',re.I)
OFFER=re.compile(r'выполня|делаем|устанавлива|монтиру|укладыва|оказыва|предлага|занимаемся|клад[её]м|проводим|производим|работаем',re.I)

def norm(s):return ' '.join(s.casefold().split())
def load_public():
    return [{'title':r['title'],'description':r['description'],'labels':sorted(NAMES[x] for x in r['categories'])}
            for r in pq.read_table(REPO/'public/public-test-dataset.parquet').to_pylist()]
def load_json(path):
    return [json.loads(s) for s in path.open(encoding='utf8') if s.strip()]
def load_synthetic():
    return [{'title':r['title'],'description':r['description'],'labels':sorted(NAMES[x] for x in r['categories'])}
            for r in pq.read_table(REPO/'public/public-test-synthetic-300.parquet').to_pylist()]
def gold(r):
    if 'labels' in r:return set(r['labels'])
    return {x['mcId'] for x in r['found_microcategories']}

def raw(model,rows):
    p=model['pipeline'].predict_proba([r['description'] for r in rows])
    pred=p>=model['thresholds']
    y=np.asarray([[int(mid in gold(r) and mid!=r.get('mcId',101)) for mid in IDS] for r in rows])
    for i,r in enumerate(rows):
        if r.get('mcId',101) in IDS:pred[i,IDS.index(r['mcId'])]=False
    return p,pred,y

def scores(y,p):
    return {'n':len(y),'precision':round(precision_score(y,p,average='micro',zero_division=0),4),
            'recall':round(recall_score(y,p,average='micro',zero_division=0),4),
            'micro_f1':round(f1_score(y,p,average='micro',zero_division=0),4),
            'fp':int(((p==1)&(y==0)).sum()),'fn':int(((p==0)&(y==1)).sum())}

def conflicts(rows):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[norm(r['description'])].append(i)
    conflict={}
    details=[]
    for indices in groups.values():
        variants={tuple(sorted(gold(rows[i]))) for i in indices}
        if len(variants)>1:
            labels={mid:{i:int(mid in gold(rows[i])) for i in indices} for mid in IDS}
            changing=[mid for mid,v in labels.items() if len(set(v.values()))>1]
            for i in indices:conflict[i]=changing
            details.append({'indices':indices,'label_variants':[list(v) for v in sorted(variants)],
                            'changing_categories':changing,
                            'description':rows[indices[0]]['description']})
    return conflict,details,groups

def evidence(text,mid):
    terms=list(re.finditer(TERMS[mid],text,re.I))
    if not terms:return []
    fragments=[]
    for match in terms[:8]:
        # Nearest sentence/list item, capped so long texts remain inspectable.
        start=max(0,match.start()-85);end=min(len(text),match.end()+115)
        fragments.append(text[start:end])
    return fragments

def reason(row,mid,kind,conflict,groups):
    text=row['description'];fragments=evidence(text,mid)
    local=' '.join(fragments).lower()
    tags=[]
    if conflict:tags.append('possible_label_conflict')
    if re.search(r'\bне\b|\bнет\b|\bбез\b',local,re.I):tags.append('negation_near_mention')
    if OTHER.search(local):tags.append('third_party_near_mention')
    if INDIRECT.search(local):tags.append('indirect_or_hypothetical_near_mention')
    if len(text)>2500:tags.append('long_text')
    if len(text)<160:tags.append('short_or_incomplete_description')
    if not fragments:tags.append('no_explicit_category_keyword')
    if kind=='FP' and not OFFER.search(local):tags.append('ambiguous_offer')
    if kind=='FN' and OFFER.search(local):tags.append('missed_explicit_offer_candidate')
    if not tags:tags.append('other_requires_review')
    # Primary tag is for grouping only: presence of words is not semantic proof.
    primary=tags[0]
    return primary,tags,fragments

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    model=joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')
    assert tuple(model['ids'])==IDS and np.allclose(model['thresholds'],.5)
    rows=load_public();p,pred,y=raw(model,rows)
    conflict,details,groups=conflicts(rows)
    full=[]; errors=[]
    for i,r in enumerate(rows):
        item={'index':i,'title':r['title'],'description':r['description'],
              'gold_mcId':[mid for j,mid in enumerate(IDS) if y[i,j]],
              'predicted_mcId':[mid for j,mid in enumerate(IDS) if pred[i,j]],
              'probabilities':{str(mid):round(float(p[i,j]),6) for j,mid in enumerate(IDS)},
              'thresholds':{str(mid):float(model['thresholds'][j]) for j,mid in enumerate(IDS)},
              'conflict_group_indices':groups[norm(r['description'])] if i in conflict else []}
        full.append(item)
        for j,mid in enumerate(IDS):
            if pred[i,j]==y[i,j]:continue
            kind='FP' if pred[i,j] else 'FN'
            primary,tags,fragments=reason(r,mid,kind,mid in conflict.get(i,[]),groups)
            errors.append({'index':i,'category':mid,'kind':kind,'probability':round(float(p[i,j]),6),
                           'threshold':float(model['thresholds'][j]),'suggested_primary_reason':primary,
                           'hypothesis_tags':tags,'source_fragments':fragments,
                           'description':r['description'],
                           'conflict_group_indices':item['conflict_group_indices']})
    def dump(name,obj):(OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
    dump('public_predictions.json',full);dump('public_errors.json',errors)
    dump('conflicts.json',details)
    report={'public_baseline':scores(y,pred),'conflict_groups':len(details),
            'conflict_records':len(conflict),'repeated_groups':sum(len(v)>1 for v in groups.values()),
            'unique_normalized_texts':len(groups),
            'errors_in_conflicting_category':collections.Counter(x['kind'] for x in errors
                if x['suggested_primary_reason']=='possible_label_conflict'),
            'reason_counts':{kind:dict(collections.Counter(x['suggested_primary_reason'] for x in errors
                                                          if x['kind']==kind)) for kind in ('FP','FN')},
            'probability_distributions':{}}
    for kind,mask in (('TP',(pred==1)&(y==1)),('FP',(pred==1)&(y==0)),('FN',(pred==0)&(y==1))):
        values=p[mask]
        report['probability_distributions'][kind]={'n':len(values),
              'p0_p25_p50_p75_p90_p95_p100':[round(float(v),4) for v in np.percentile(values,[0,25,50,75,90,95,100])],
              'gte_0_8':int((values>=.8).sum()),'gte_0_9':int((values>=.9).sum())}
    report['high_confidence_fp']={'gte_0_8':int(sum(x['kind']=='FP' and x['probability']>=.8 for x in errors)),
                                  'in_conflicting_category':int(sum(x['kind']=='FP' and x['probability']>=.8 and
                                      x['suggested_primary_reason']=='possible_label_conflict' for x in errors))}
    report['errors_in_conflicting_category']=dict(report['errors_in_conflicting_category'])
    dump('diagnostic_summary.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
