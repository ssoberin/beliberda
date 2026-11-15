"""Find review candidates in existing train; lexical cues are not label verdicts."""
import collections
import hashlib
import json
import random
import re
from pathlib import Path

TRAIN=Path('/srv/Hakaton/data/processed/train.jsonl')
OUT=Path('/srv/Hakaton/results/data_quality_audit')
SEED=20261115
TERMS={102:r'штукатур|оштукатур|маяк',
       103:r'плитк|керамогранит|кафел|мозаик|затирк',
       104:r'сантех|труб|раковин|смесител|водопровод|унитаз',
       105:r'электр|проводк|розетк|светильник|выключател|щит',
       106:r'обо[ий]|покраск|окраск|маляр|шпакл|шпатл'}
NEG=re.compile(r'\bне\s+(?:монтир\w*|устанавл\w*|выполн\w*|дела\w*|клад\w*|кле\w*|занима\w*|провод\w*|оказыва\w*|бер\w*|работа\w*)|\b(?:не\s+предлага\w*|отказ\w*|не\s+будем)\b',re.I)
THIRD=re.compile(r'друг(?:ой|ие|им)\s+(?:мастер\w*|специалист\w*|бригад\w*|исполнител\w*)|сторонн(?:ий|ие|им)\s+(?:мастер\w*|специалист\w*|подрядчик\w*)|заказчик\s+сам|приглаш[а-я]+\s+(?:мастер\w*|специалист\w*)',re.I)
CONDITIONAL=re.compile(r'если\s+потребу\w*|можно\s+обсудить|например|по\s+желанию|при\s+необходимости',re.I)
OFFER=re.compile(r'выполн\w*|монтир\w*|устанавл\w*|укладыва\w*|клад\w*|делаем|предлага\w*|оказыва\w*|занимаемся|проводим|производим',re.I)

def fragments(text):
    return [s.strip() for s in re.split(r'(?<=[.!?;])\s+|[\r\n]+',text) if s.strip()]

def cues(text):
    parts=fragments(text)
    neg_categories=set();offered_categories=set();third_categories=set()
    tags=set();excerpts=[]
    for part in parts:
        mids={mid for mid,pat in TERMS.items() if re.search(pat,part,re.I)}
        if not mids:continue
        if NEG.search(part):
            tags.add('negation_near_service');neg_categories|=mids;excerpts.append(part[:450])
        if THIRD.search(part):
            tags.add('third_party_near_service');third_categories|=mids;excerpts.append(part[:450])
        if OFFER.search(part) and not NEG.search(part) and not THIRD.search(part):
            offered_categories|=mids
        if CONDITIONAL.search(part):tags.add('conditional_or_example_near_service');excerpts.append(part[:450])
    overlap=neg_categories&offered_categories
    if overlap:tags.add('positive_and_negative_same_category')
    # Wide paragraphs may bring unrelated statements together; candidate signal only.
    return sorted(tags),sorted(overlap),list(dict.fromkeys(excerpts))[:4]

def main():
    rows=[json.loads(line) for line in TRAIN.open(encoding='utf-8') if line.strip()]
    candidates=[]
    for i,row in enumerate(rows):
        tags,overlap,quotes=cues(row['description'])
        if tags:
            candidates.append({'index':i,'tags':tags,'overlap_categories':overlap,
                               'source_fragments':quotes,'row':row})
    counts=collections.Counter(tag for r in candidates for tag in r['tags'])
    rng=random.Random(SEED)
    rng.shuffle(candidates)
    selected=[];picked=set()
    quotas={'positive_and_negative_same_category':40,'third_party_near_service':35,
            'negation_near_service':30,'conditional_or_example_near_service':25}
    for tag,limit in quotas.items():
        for item in candidates:
            if limit<=0:break
            if tag not in item['tags'] or item['index'] in picked:continue
            selected.append(item);picked.add(item['index']);limit-=1
    for item in candidates:
        if len(selected)>=130:break
        if item['index'] not in picked:
            selected.append(item);picked.add(item['index'])
    selected.sort(key=lambda x:x['index'])
    assert 100<=len(selected)<=150
    public=OUT/'annotation';private=OUT/'private'
    public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,exist_ok=True)
    public.chmod(0o750)
    private.chmod(0o700)
    refs=[]
    with (public/'train_review.jsonl').open('w',encoding='utf-8') as stream:
        for item in selected:
            row=item['row'];i=item['index']
            item_id=f'train-{i:05d}-'+hashlib.sha256(row['description'].encode()).hexdigest()[:10]
            # Train review deliberately shows current labels for confirmation/correction.
            task={'item_id':item_id,'description':row['description'],
                  'current_labels':row['labels'],'review_status':'',
                  'corrected_labels':None,'human_comment':'','original_index':i}
            stream.write(json.dumps(task,ensure_ascii=False)+'\n')
            refs.append({'item_id':item_id,'original_index':i,'current_labels':row['labels'],
                         'suspicion_tags':item['tags'],
                         'possible_contradiction_categories':item['overlap_categories'],
                         'source_fragments':item['source_fragments']})
    (public/'train_review.jsonl').chmod(0o640)
    (private/'train_review_candidates.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
    (private/'train_review_candidates.json').chmod(0o600)
    result={'train_rows':len(rows),'candidate_rows':len(candidates),
            'candidate_tag_counts':dict(counts),'selected':len(selected),
            'selected_tag_counts':dict(collections.Counter(t for x in selected for t in x['tags'])),
            'seed':SEED,'reviewer_sees_current_train_labels':True,
            'warning':'Lexical triggers are candidate retrieval, not proof of a wrong label.'}
    (OUT/'train_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
