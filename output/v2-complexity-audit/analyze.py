"""Read-only text and frozen-model complexity audit; no fit or threshold search."""
import collections
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT if (ROOT/'datasets/hard_dataset/generate.py').exists() else Path('/tmp/innopolis-test-progress-20261114')
OUT = Path(__file__).resolve().parent
DATA = ROOT/'datasets/hard_dataset/v2'
PUBLIC = REPO/'datasets/public/public-train-annotated.jsonl'
if not PUBLIC.exists():
    PUBLIC = ROOT/'mvp/innoglobalhack_avito/innoglobalhack_avito/public-train-annotated.jsonl'
MODEL = ROOT/'output/mixed-charwb-comparison/runs/char_wb/tfidf_baseline.joblib'
IDS = (102,103,104,105,106)
GROUP_NAMES = {'realistic':'Реалистичные', 'plausible_hard':'Сложные правдоподобные', 'artificial':'Искусственные'}

MENTIONS = {
    102:r'штукатур|штукотур|оштукатур|маяк',
    103:r'плитк|плиточ|кафел|керамогр[ао]нит|моза[ий]к',
    104:r'сантех|см[еи]сител|раков|ракав|ун[ие]таз|водоснаб|канализац|труб|радиатор|сифон',
    105:r'электр|електр|розет|разет|выключ|электрощ|електрощ|провод(?:к|а|ы)|провад',
    106:r'обо[ийв]|або[ийв]|покрас|пакрас|красим|крашу|маляр|шпа[кт]л|шлиф|грунт',
}
PRED = re.compile(r'\b(?:выполн\w*|выпалн\w*|предлага\w*|предлога\w*|оказыва\w*|принима\w*|принимаит|дела\w*|бер[её]м|укладыва\w*|клеим|клею|красим|крашу|меняем|меняю)\b',re.I)
TOKENS = re.compile(r'[а-яёa-z]+',re.I)
FLAGS = {
    'negation_particle':r'\b(?:не|ни|нет)\b',
    'not_only':r'\bне\s+только\b',
    'double_negation_candidate':r'не\s+отказыва|не\s+исключа|не\s+могу\s+сказать.{0,30}\bне\b|неверно\s*,?\s*что.{0,80}\bне\b|не\s+соответств\w*.{0,30}(?:действительност|правд)',
    'formal_meta':r'опроверж|соответств\w*\s+(?:факт|действительност)|сообщение.{0,25}будто\s+неверно|предложение\s+выполн',
    'foreign_work':r'друг(?:ой|ая|ую|ие|их|ого)\s+(?:мастер|бригад|специалист|подрядчик)|заказчик\s+(?:делает|заказывает|сам)|сантехник\s+заказчика|клеит\s+заказчик',
    'quote_or_report':r'[«»]|клиент\s+(?:написал|спрашива)|заказчик:|говорят|сообщени|слух',
    'question':r'\?|\b(?:выполняем|выполняете|выпалняем)\s+ли\b',
    'condition':r'\bесли\b|\bпри\s+(?:наличии|необходимости|заказе)|по\s+желанию',
    'correction_or_retraction':r'исправлени|раньше.{0,80}(?:теперь|сейчас)|старой\s+версии|это\s+(?:неверно|верно)|информац\w*\s+о\s+работах',
    'surface_conflicting_mentions':r'(?:не\s+выполн\w*.{0,120}выполн\w*|выполн\w*.{0,120}не\s+выполн\w*)',
}

def read(path): return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(text): return ' '.join(text.casefold().split())
def words(text): return TOKENS.findall(text.casefold().replace('ё','е'))
def found(row): return set(c['mcId'] for c in row['found_microcategories']) & set(IDS)
def offered(row): return set(row.get('offered_microcategories', [c['mcId'] for c in row['found_microcategories']])) & set(IDS)


def realism(text):
    """Expert, textual, deterministic; neither type/depth nor predictions enter."""
    t=norm(text)
    if 'опроверж' in t:
        return 'artificial','Оценка истинности опровержения: формальный метаязык вместо прямого предложения услуги'
    if re.search(r'фраз\w*.*соответств\w*\s+действительност',t):
        return 'artificial','Истинность цитируемой фразы определяется через соответствие действительности'
    if 'предложение выполн' in t or 'предложение выпалн' in t:
        return 'artificial','Метаутверждение о подтверждении предложения работы'
    if re.search(r'(?:неверно|верно)\s*,?\s*что\s+мы\s+не\s+только\s+не',t):
        return 'artificial','Вложенное отрицание высказывания с не только не'
    if re.search(r'сообщение.{0,25}будто\s+неверно',t):
        return 'artificial','Цепочка сообщение → неверно → предложение/отказ'
    if len(re.findall(r'\b(?:неверно|верно)\s*,?\s*что',t))>=3:
        return 'artificial','Серия из ≥3 однотипных метаутверждений истинности статуса услуг'
    if 'это порядок переговоров' in t and 'перв' in t and 'втор' in t:
        return 'artificial','Явно учебное рассуждение о первой/второй позиции после отвлекающего текста'
    mentioned=sum(bool(re.search(p,t)) for p in MENTIONS.values())
    if mentioned>=4 and re.search(r'не\s+(?:выполн|выпалн|предлаг|предлог)|\bнет\b',t):
        return 'plausible_hard','Многопозиционный список с независимыми положительными/отрицательными статусами'
    if re.search(r'сообщени[еяю]\s+(?:о\s+том|будто)|неверно\s*,?\s*что|верно\s*,?\s*что|говорят|фраз[ауы]\b|не\s+могу\s+сказать|никто\s+не\s+может\s+сказать|нельзя\s+считать|не\s+отрица',t):
        return 'plausible_hard','Единичное сообщение/метаутверждение/речевая модальность, требующие интерпретации'
    if re.search(r'\?\s*(?:да|нет|тоже)|(?:выполняем|выпалняем)\s+ли|клиент\s+написал|заказчик:|(?:первую|вторую)\s+(?:работу|услугу)|кроме\s+(?:первой|второй)|только\s+(?:первую|вторую)|с\s+первой\s+услугой|исправлени|старой\s+версии|это\s+мы\s+(?:не\s+)?делаем|если\s+бы|не\s+оставим.{0,35}\bбез\b',t):
        return 'plausible_hard','FAQ, цитата, анафора, исправление, контрфактическое условие или область отрицания'
    # Several natural scopes in a short utterance are plausible, though demanding.
    if sum(1 for _ in re.finditer(r'\b(?:не|ни|нет)\b',t))>=4 and mentioned>=2:
        return 'plausible_hard','Несколько независимых ограничений/отрицаний в одном коротком тексте'
    return 'realistic','Прямое предложение/отказ, обычное условие либо явно разграниченный исполнитель'


def negation_tier(text):
    t=norm(text)
    if re.search(FLAGS['formal_meta'],t): return 'formal_nested_meta'
    if re.search(FLAGS['double_negation_candidate'],t): return 'double_or_reversed_scope_candidate'
    if re.search(FLAGS['negation_particle'],t): return 'simple_or_multiple_flat_scopes'
    return 'no_separate_ne_ni_net'


def summarize(rows):
    lengths=np.asarray([len(r['description']) for r in rows])
    mentioned=[sum(bool(re.search(p,r['description'],re.I)) for p in MENTIONS.values()) for r in rows]
    positive_raw=[]; negative_raw=[]
    for r in rows:
        text=r['description']; pos=neg=0
        for match in PRED.finditer(text):
            before=text[max(0,match.start()-25):match.start()].casefold()
            if re.search(r'\bне\s+(?:\w+\s+)?$',before) and not re.search(r'\bне\s+только\s*$',before): neg+=1
            else: pos+=1
        positive_raw.append(pos);negative_raw.append(neg)
    return {
        'rows':len(rows), 'length_chars':dict(zip(['min','p10','p25','median','p75','p90','p95','p99','max'],map(float,np.quantile(lengths,[0,.1,.25,.5,.75,.9,.95,.99,1])))),
        'mentioned_categories_proxy':dict(sorted(collections.Counter(mentioned).items())),
        'mentioned_categories_mean':float(np.mean(mentioned)),
        'offered_labels_mean':float(np.mean([len(offered(r)) for r in rows])),
        'mentioned_but_not_offered_proxy_mean':float(np.mean([max(0,n-len(offered(r))) for r,n in zip(rows,mentioned)])),
        'predicate_candidates':{'affirmative_surface_count':sum(positive_raw),'negated_surface_count':sum(negative_raw),
            'affirmative_per_doc':float(np.mean(positive_raw)),'negated_per_doc':float(np.mean(negative_raw)),
            'note':'Surface predicates only, not semantically resolved assertions. Quotations/meta-negation can invert their interpretation.'},
        'negation_tokens_per_100_words':100*sum(len(re.findall(FLAGS['negation_particle'],r['description'],re.I)) for r in rows)/sum(len(words(r['description'])) for r in rows),
        'question_marks_total':sum(r['description'].count('?') for r in rows),
        'text_flags':{name:sum(bool(re.search(pattern,r['description'],re.I)) for r in rows) for name,pattern in FLAGS.items()},
        'negation_surface_tiers':dict(collections.Counter(negation_tier(r['description']) for r in rows)),
        'realism':dict(collections.Counter(realism(r['description'])[0] for r in rows)),
    }


def metric(gold,pred,prob=None,source=None):
    if not len(gold):return {'rows':0}
    result={'rows':len(gold),'micro_f1':float(f1_score(gold,pred,average='micro',zero_division=0)),
            'macro_f1':float(f1_score(gold,pred,average='macro',zero_division=0)),
            'precision':float(precision_score(gold,pred,average='micro',zero_division=0)),
            'recall':float(recall_score(gold,pred,average='micro',zero_division=0)),
            'tp':int(((gold==1)&(pred==1)).sum()),'fp':int(((gold==0)&(pred==1)).sum()),'fn':int(((gold==1)&(pred==0)).sum()),
            'exact_set_accuracy':float(accuracy_score(gold,pred)),
            'empty_predictions':int((pred.sum(1)==0).sum()),
            'empty_on_positive_rows':int(((pred.sum(1)==0)&(gold.sum(1)>0)).sum()),
            'positive_rows':int((gold.sum(1)>0).sum()),'gold_empty_rows':int((gold.sum(1)==0).sum())}
    return result


def lexical(rows):
    tokenized=[words(r['description']) for r in rows]
    counts=collections.Counter(w for doc in tokenized for w in doc)
    eligible=[doc for doc in tokenized if len(doc)>=20]
    rng=np.random.default_rng(42);samples=[]
    n=min(400,len(eligible))
    for _ in range(30):
        selected=rng.choice(len(eligible),n,replace=False)
        window=[]
        for index in selected:
            doc=eligible[int(index)];start=int(rng.integers(0,len(doc)-19));window.extend(doc[start:start+20])
        samples.append(len(set(window))/len(window))
    return {'word_tokens':sum(counts.values()),'word_types':len(counts),
            'matched_20word_windows':{'eligible_docs':len(eligible),'sample_docs':n,'tokens_per_resample':n*20,'resamples':30,
                'mean_TTR':float(np.mean(samples)),'std_TTR':float(np.std(samples)),
                'note':'One random 20-word window from each of 400 sampled documents. Documents shorter than20 words excluded; coverage differs.'},
            'top20_word_token_share':sum(n for _,n in counts.most_common(20))/sum(counts.values())}


def main():
    paths={name:DATA/f'{name}.jsonl' for name in ['train','validation','test','challenge']}
    hashes={str(path):sha(path) for path in [*paths.values(),PUBLIC,MODEL]}
    splits={name:read(path) for name,path in paths.items()}
    raw=read(PUBLIC);public=list({norm(r['description']):r for r in reversed(raw)}.values())
    # Preserve first occurrence for normalized deduplication.
    first={}
    for row in raw:first.setdefault(norm(row['description']),row)
    public=list(first.values())
    all_v2=[r for rows in splits.values() for r in rows]
    stats={'definitions':{'scope':IDS,'model':'frozen mixed public+v2 char_wb46k, five SGD heads, thresholds0.5',
        'realism':'Expert textual heuristic, not measured prevalence on Avito; independent of labels/types/model outcomes.',
        'mention_count':'Number of distinct category-specific lexical patterns, not number of actual operations or offers.',
        'negation':'Textual surface tiers are distinct from generator semantic depth; neither particle parity nor metadata alone proves meaning.'},
        'datasets':{name:summarize(rows) for name,rows in [('public_raw',raw),('public_dedup',public),*[(f'v2_{k}',v) for k,v in splits.items()],('v2_all',all_v2)]},
        'lexical':{},'template':{},'evaluation':{},'source_hashes':hashes}
    for name,rows in [('public_dedup',public),('v2_train',splits['train']),('v2_test',splits['test'])]:
        stats['lexical'][name]=lexical(rows)

    # Canonical service phrases removed consistently in both corpora, after reversing
    # the documented spelling/grammar substitutions. Skeleton is an analytic proxy.
    sys.path.insert(0,str(REPO/'datasets/hard_dataset'))
    import generate, expanded_cases
    inverse={v:k for table in [expanded_cases.GRAMMAR,expanded_cases.SPELLING] for k,v in table.items()}
    def spelling_norm(text):return ' '.join(inverse.get(w,w) for w in words(text))
    service_phrases=sorted({spelling_norm(s) for ss in generate.SERVICES.values() for s in ss},key=len,reverse=True)
    def skeleton(text):
        t=' '+spelling_norm(text)+' '
        for phrase in service_phrases:t=t.replace(' '+phrase+' ',' <SERVICE> ')
        return ' '.join(t.split())
    for name,rows in [('public_dedup',public),('v2_train',splits['train']),('v2_test',splits['test']),('v2_all',all_v2)]:
        counts=collections.Counter(skeleton(r['description']) for r in rows)
        stats['template'][name]={'unique_skeletons':len(counts),'rows':len(rows),
            'top20_skeleton_row_share':sum(n for _,n in counts.most_common(20))/len(rows),
            'top5':counts.most_common(5),
            'note':'Heuristic skeleton: documented error variants normalized,18 canonical service phrases replaced by<SERVICE>, punctuation removed. Not full semantic deduplication.'}

    model=joblib.load(MODEL);pipe=model['pipeline'];assert tuple(model['ids'])==IDS;assert np.array_equal(model['thresholds'],np.full(5,.5))
    branches=dict(pipe.named_steps['vectorizer'].transformer_list)
    public_holdout=[dict(r,mcId=101,found_microcategories=[{'mcId':x} for x in r['labels']]) for r in read(ROOT/'output/ngram-comparison/splits/validation.jsonl')]
    catalog=[]
    evaluated={}
    for split,rows in [('public_holdout_729',public_holdout),('v2_test',splits['test']),('challenge',splits['challenge'])]:
        x=[r['description'][:6000] for r in rows]
        y=np.asarray([[int(mid in found(r)) for mid in IDS] for r in rows],dtype=np.int8)
        p=pipe.predict_proba(x);pred=(p>=.5).astype(np.int8)
        eligible=np.asarray([[r.get('mcId',101)!=mid for mid in IDS] for r in rows])
        pred[~eligible]=0
        result={'overall':metric(y,pred),'realism':{},'surface_negation':{},'generator_depth':{},'type':{},'scores':{},'features':{},'contrast_pairs':{}}
        classifications=[realism(t)[0] for t in x]
        for name,values in [('realism',classifications),('surface_negation',[negation_tier(t) for t in x]),
                            ('generator_depth',[str(r.get('negation_depth','unannotated')) for r in rows]),
                            ('type',[r.get('type','public') for r in rows])]:
            for value in sorted(set(values)):
                mask=np.asarray([v==value for v in values]);result[name][value]=metric(y[mask],pred[mask])
        for i,mid in enumerate(IDS):
            m=eligible[:,i];yp=y[m,i];pp=p[m,i]
            pos=pp[yp==1];neg=pp[yp==0]
            result['scores'][str(mid)]={'roc_auc':float(roc_auc_score(yp,pp)) if len(set(yp))>1 else None,
                'average_precision':float(average_precision_score(yp,pp)) if yp.any() else None,
                'positive_p10_median_p90':list(map(float,np.quantile(pos,[.1,.5,.9]))) if len(pos) else [],
                'negative_p10_median_p90':list(map(float,np.quantile(neg,[.1,.5,.9]))) if len(neg) else [],
                'positive_below_0_2':int((pos<.2).sum()),'positive_0_2_to_0_4':int(((pos>=.2)&(pos<.4)).sum()),
                'positive_0_4_to_0_5':int(((pos>=.4)&(pos<.5)).sum()),'positive_at_least_0_5':int((pos>=.5).sum()),
                'positive_negative_score_ranges_overlap':bool(len(pos) and len(neg) and pos.min()<=neg.max())}
        for branch,v in branches.items():
            analyze=v.build_analyzer();tokens=[analyze(t) for t in x]
            known=sum(sum(token in v.vocabulary_ for token in doc) for doc in tokens);total=sum(map(len,tokens))
            unique={token for doc in tokens for token in doc}
            matrix=v.transform(x)
            result['features'][branch]={'candidate_occurrences':total,'known_occurrences':known,'occurrence_coverage':known/total,
                'candidate_types':len(unique),'known_types':sum(t in v.vocabulary_ for t in unique),
                'zero_vectors':int((np.diff(matrix.indptr)==0).sum()),'median_nonzero_features':float(np.median(np.diff(matrix.indptr)))}
        matrix=pipe.named_steps['vectorizer'].transform(x).tocsr()
        collisions=collections.defaultdict(list)
        for i,r in enumerate(rows):
            a,b=matrix.indptr[i:i+2];fingerprint=hashlib.sha256(matrix.indices[a:b].tobytes()+matrix.data[a:b].tobytes()).hexdigest()
            collisions[(r.get('mcId',101),fingerprint)].append(i)
        impossible=[indices for indices in collisions.values() if len({tuple(y[i]) for i in indices})>1]
        collision_examples=[]
        for indices in impossible:
            ordered=sorted(indices,key=lambda i:rows[i].get('augmentation','clean')!='clean')
            a=ordered[0]
            b=next(i for i in ordered if not np.array_equal(y[i],y[a]))
            assert np.array_equal(matrix[a].indices,matrix[b].indices)
            assert np.array_equal(matrix[a].data,matrix[b].data)
            assert np.array_equal(p[a],p[b])
            if len(collision_examples)<3:
                collision_examples.append({'rows':[{'item_id':rows[i].get('item_id',i),'text':rows[i]['description'],'source':rows[i].get('mcId',101),'gold':sorted(found(rows[i])),'type':rows[i].get('type'),'scores':p[i].tolist()} for i in (a,b)],
                    'sparse_indices_bitwise_equal':True,'sparse_data_bitwise_equal':True,'probabilities_bitwise_equal':True,
                    'nonzero_elements_of_vector_difference':int((matrix[a]-matrix[b]).nnz)})
        unavoidable_errors=sum(int(np.minimum(y[indices].sum(0),len(indices)-y[indices].sum(0)).sum()) for indices in impossible)
        result['features']['exact_representation_collisions']={'conflicting_groups':len(impossible),'rows':sum(map(len,impossible)),
            'rows_by_type':dict(collections.Counter(rows[i].get('type','public') for indices in impossible for i in indices)),
            'groups_by_types':dict(collections.Counter('|'.join(sorted(set(rows[i].get('type','public') for i in indices))) for indices in impossible)),
            'minimum_unavoidable_label_errors_for_any_deterministic_classifier_of_this_representation':unavoidable_errors,
            'examples':collision_examples,
            'note':'Exact identical CSR indices and float data bytes plus same source category but different gold. No rounding. Selected examples additionally checked with np.array_equal on sparse arrays and probability vectors. Zero collisions does not prove linear separability.'}
        if split!='public_holdout_729':
            pair_groups=collections.defaultdict(list)
            for i,r in enumerate(rows):
                if r.get('augmentation','clean') in ('clean','authored_errors'):
                    pair_groups[r['group_id']].append(i)
            pair_stats=collections.defaultdict(lambda:collections.Counter())
            for group,indices in pair_groups.items():
                if len(indices)!=2:continue
                a,b=indices;changed=np.flatnonzero(y[a]!=y[b]);typ=rows[a]['type']
                if not len(changed):
                    pair_stats[typ]['unobservable_after_scope']+=1;continue
                pair_stats[typ]['observable_pairs']+=1
                pair_stats[typ]['both_exact']+=int(np.array_equal(y[a],pred[a]) and np.array_equal(y[b],pred[b]))
                pair_stats[typ]['both_empty']+=int(not pred[a].any() and not pred[b].any())
                for j in changed:
                    positive,negative=(a,b) if y[a,j] else (b,a)
                    pair_stats[typ]['changed_label_comparisons']+=1
                    pair_stats[typ]['correct_score_order']+=int(p[positive,j]>p[negative,j])
                    pair_stats[typ]['equal_score']+=int(p[positive,j]==p[negative,j])
            result['contrast_pairs']={k:dict(v) for k,v in pair_stats.items()}
        stats['evaluation'][split]=result;evaluated[split]=(rows,y,pred,p,classifications)
    # Outcome-independent example selection: rotate split/type, length strata.
    def select_examples(rows,n=20):
        buckets=collections.defaultdict(list)
        for row in rows:buckets[(row.get('_split','public'),row.get('type','public'))].append(row)
        chosen=[];seen=set()
        for key,bucket in sorted(buckets.items()):
            ordered=sorted(bucket,key=lambda r:(len(r['description']),str(r['item_id'])))
            for fraction in (.5,.1,.9):
                r=ordered[min(len(ordered)-1,int(fraction*len(ordered)))];k=norm(r['description'])
                if k not in seen:chosen.append(r);seen.add(k);break
            if len(chosen)>=n:return chosen
        ordered=sorted(rows,key=lambda r:(len(r['description']),str(r['item_id'])))
        for i in np.linspace(0,len(ordered)-1,min(len(ordered),n*3),dtype=int):
            r=ordered[int(i)];k=norm(r['description'])
            if k not in seen:chosen.append(r);seen.add(k)
            if len(chosen)>=n:break
        return chosen
    public_candidates=[]
    for row in public:
        copy=dict(row,mcId=101,_split='public_raw_dedup')
        # Stratify by text cues rather than label balance, already known.
        copy['type']=negation_tier(row['description'])+('/long' if len(row['description'])>=1000 else '/short')
        public_candidates.append(copy)
    example_sets={'original_public':select_examples(public_candidates)}
    enriched=[dict(r,_split=name) for name,rs in splits.items() for r in rs]
    for group in GROUP_NAMES:
        example_sets[group]=select_examples([r for r in enriched if realism(r['description'])[0]==group])
    for group,rows in example_sets.items():
        for i,row in enumerate(rows,1):
            detected=[mid for mid,pattern in MENTIONS.items() if re.search(pattern,row['description'],re.I)]
            category,reason=realism(row['description'])
            catalog.append({'section':group,'number':i,'item_id':row['item_id'],'split':row['_split'],'type':row.get('type'),
                'source_mcId':row.get('mcId',101),'text':row['description'],'gold_102_106':sorted(found(row)),
                'offered_102_106':sorted(offered(row)),'mentioned_categories_proxy':detected,
                'realism_group':category,'realism_reason':reason,'negation_surface_tier':negation_tier(row['description']),
                'ambiguity_review':'Авторская формальная истинность не гарантирует естественную интерпретацию предложения услуги' if 'опроверж' in row['description'].casefold() else 'Подробная экспертная проверка в примечаниях каталога; метки исходника сохранены'})
    stats['example_catalog_counts']=dict(collections.Counter(r['section'] for r in catalog))
    stats['hashes_unchanged']=all(sha(Path(path))==expected for path,expected in hashes.items())
    assert stats['hashes_unchanged']
    (OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    (OUT/'example_catalog.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'realism':{k:v['realism'] for k,v in stats['datasets'].items()},
                      'model':{k:v['overall'] for k,v in stats['evaluation'].items()},
                      'group_metrics':stats['evaluation']['v2_test']['realism'],
                      'lexical':stats['lexical'],'template':stats['template'],
                      'scores':stats['evaluation']['v2_test']['scores'],
                      'features':stats['evaluation']['v2_test']['features'],
                      'catalog':stats['example_catalog_counts']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
