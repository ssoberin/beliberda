"""Read-only EDA of the committed v2 corpus; standard library only."""
import collections
import hashlib
import itertools
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'datasets/hard_dataset/v2'
OLD = ROOT / 'datasets/public/public-train-annotated.jsonl'
if not OLD.exists():
    OLD = ROOT / 'mvp/innoglobalhack_avito/innoglobalhack_avito/public-train-annotated.jsonl'
OUT = ROOT / 'output/v2-eda'
OUT.mkdir(parents=True, exist_ok=True)
SCOPE = set(range(102,107))

def norm(text): return ' '.join(text.casefold().split())
def found(row): return set(x['mcId'] for x in row['found_microcategories'])
def percentile(numbers, q):
    numbers=sorted(numbers); index=(len(numbers)-1)*q
    low=math.floor(index); high=math.ceil(index)
    return numbers[low]+(numbers[high]-numbers[low])*(index-low)
def distribution(values): return dict(sorted(collections.Counter(values).items()))
def summary(numbers):
    return {k: percentile(numbers,q) for k,q in [('min',0),('p10',.1),('p50',.5),('p90',.9),('p95',.95),('max',1)]}
def card(rows, projected=True): return distribution(len(found(r)&SCOPE if projected else found(r)) for r in rows)
def combo(rows, projected=True):
    c=collections.Counter(tuple(sorted(found(r)&SCOPE if projected else found(r))) for r in rows)
    return {','.join(map(str,k)) or 'empty':v for k,v in c.most_common()}

splits={name:[json.loads(s) for s in (SOURCE/f'{name}.jsonl').read_text().splitlines() if s.strip()] for name in ['train','validation','test','challenge']}
old=[json.loads(s) for s in OLD.read_text().splitlines() if s.strip()]
stats={'source_commit':'0b1be5d','scope':sorted(SCOPE),'splits':{},'checks':{},'overlaps':{}}
exceptions=[]; seen_ids={}; by_id={r['item_id']:(name,r) for name,rs in splits.items() for r in rs}
for name, rows in splits.items():
    lengths=[len(r['description']) for r in rows]
    words=[len(re.findall(r'[А-Яа-яЁёA-Za-z]+',r['description'])) for r in rows]
    groups=collections.defaultdict(list)
    for r in rows: groups[r['group_id']].append(r)
    texts=collections.Counter(norm(r['description']) for r in rows)
    inputs=collections.defaultdict(set)
    for r in rows: inputs[(norm(r['description']),r['mcId'])].add(tuple(sorted(found(r))))
    projected_card=card(rows)
    full_card=card(rows,False)
    aug=distribution(r.get('augmentation','clean') for r in rows)
    neg=[len(re.findall(r'\b(?:не|ни|нет)\b',r['description'],re.I)) for r in rows]
    all_service_masks=set(tuple(sorted(r['offered_microcategories'])) for r in rows)
    stats['splits'][name]={
        'rows':len(rows),'sha256':hashlib.sha256((SOURCE/f'{name}.jsonl').read_bytes()).hexdigest(),
        'unique_normalized_descriptions':len(texts),'duplicate_descriptions_beyond_first':sum(n-1 for n in texts.values()),
        'unique_full_inputs':len(inputs),'conflicting_full_inputs':sum(len(x)>1 for x in inputs.values()),
        'target_cardinality_102_106':projected_card,'target_cardinality_all6':full_card,
        'offered_cardinality_102_106':distribution(len(set(r['offered_microcategories'])&SCOPE) for r in rows),
        'share_zero_or_five_projected':sum(projected_card.get(x,0) for x in [0,5])/len(rows),
        'share_one_or_two_projected':sum(projected_card.get(x,0) for x in [1,2])/len(rows),
        'found_counts_102_106':{str(k):sum(k in found(r) for r in rows) for k in sorted(SCOPE)},
        'offered_counts_102_106':{str(k):sum(k in r['offered_microcategories'] for r in rows) for k in sorted(SCOPE)},
        'source_categories':distribution(r['mcId'] for r in rows),
        'characters':summary(lengths),'words':summary(words),
        'length_buckets':{'lt20':sum(n<20 for n in lengths),'20_99':sum(20<=n<100 for n in lengths),'100_299':sum(100<=n<300 for n in lengths),'300_599':sum(300<=n<600 for n in lengths),'600_999':sum(600<=n<1000 for n in lengths),'1000_plus':sum(n>=1000 for n in lengths),'over6000':sum(n>6000 for n in lengths)},
        'augmentation':aug,'topics':distribution(r.get('topic','original_semantics') for r in rows),
        'negation_depth_metadata':distribution(r.get('negation_depth',0) for r in rows),
        'negation_token_count':distribution(neg),'rows_with_ne_ni_net':sum(n>0 for n in neg),
        'rows_with_pod_klyuch':sum(bool(re.search(r'под\s+ключ',r['description'],re.I)) for r in rows),
        'type_family_count':len(set(r['type'] for r in rows)),'types':distribution(r['type'] for r in rows),
        'semantic_groups':len(groups),'clean_semantic_variants':len(set((r['group_id'],r['variant']) for r in rows)),
        'offered_subsets_all6':len(all_service_masks),'projected_target_combinations':combo(rows),
        'confidence':distribution(r.get('confidence','missing') for r in rows),'provenance':distribution(r.get('provenance','missing') for r in rows),
    }
    degenerate=0; modepairs=0
    for gid, group in groups.items():
        modes=collections.defaultdict(list)
        for r in group:modes[r.get('augmentation','clean')].append(r)
        for mode,pair in modes.items():
            modepairs+=1
            if len(pair)!=2:exceptions.append({'split':name,'group':gid,'mode':mode,'error':'pair_size','size':len(pair)})
            elif (found(pair[0])&SCOPE)==(found(pair[1])&SCOPE):degenerate+=1
    stats['splits'][name]['contrast_pairs_with_augmentations']=modepairs
    stats['splits'][name]['contrast_pairs_indistinguishable_in_5_head_scope']=degenerate
    for r in rows:
        rid=r['item_id']
        if rid in seen_ids:exceptions.append({'id':rid,'error':'duplicate_id'})
        seen_ids[rid]=name
        if found(r)!=(set(r['offered_microcategories'])-{r['mcId']}):exceptions.append({'id':rid,'error':'found_offered_source_mismatch'})
        if len(found(r))!=len(r['found_microcategories']):exceptions.append({'id':rid,'error':'duplicate_labels'})
        if r['mcId'] not in range(101,107) or not set(r['offered_microcategories'])<=set(range(101,107)):exceptions.append({'id':rid,'error':'invalid_category'})
        if not isinstance(r['description'],str) or not r['description'].strip():exceptions.append({'id':rid,'error':'empty_description'})
        if 'clean_item_id' in r:
            clean=by_id.get(r['clean_item_id'])
            if clean is None:exceptions.append({'id':rid,'error':'missing_clean'})
            elif clean[0]!=name:exceptions.append({'id':rid,'error':'cross_split_clean'})
            else:
                cr=clean[1]
                if any(r[f]!=cr[f] for f in ['mcId','found_microcategories','offered_microcategories','group_id','variant']):exceptions.append({'id':rid,'error':'clean_label_metadata_mismatch'})

for a,b in itertools.combinations(splits,2):
    aa,bb=splits[a],splits[b]
    comparisons={
        'normalized_description':(set(norm(r['description']) for r in aa),set(norm(r['description']) for r in bb)),
        'normalized_effective6000_description':(set(norm(r['description'][:6000]) for r in aa),set(norm(r['description'][:6000]) for r in bb)),
        'group_id':(set(r['group_id'] for r in aa),set(r['group_id'] for r in bb)),
        'type':(set(r['type'] for r in aa),set(r['type'] for r in bb)),
        'item_id':(set(r['item_id'] for r in aa),set(r['item_id'] for r in bb)),
        'clean_item_id':(set(r.get('clean_item_id',r['item_id']) for r in aa),set(r.get('clean_item_id',r['item_id']) for r in bb)),
    }
    stats['overlaps'][f'{a}__{b}']={key:len(sets[0]&sets[1]) for key,sets in comparisons.items()}

oldnorm=set(norm(r['description']) for r in old)
stats['old_public_train']={'rows':len(old),'unique_normalized_descriptions':len(oldnorm),'target_cardinality':card(old),
    'share_zero_or_five':sum(len(found(r)) in [0,5] for r in old)/len(old),'characters':summary([len(r['description']) for r in old]),
    'found_counts_102_106':{str(k):sum(k in found(r) for r in old) for k in sorted(SCOPE)}}
stats['old_public_train']['normalized_overlap_v2']={name:len(oldnorm&set(norm(r['description']) for r in rs)) for name,rs in splits.items()}
stats['checks']['structural_exceptions']=exceptions
stats['checks']['all_source_aware_label_sets_consistent']=not any(x['error']=='found_offered_source_mismatch' for x in exceptions)
stats['checks']['cross_split_overlap_counts_all_zero']=all(all(n==0 for n in v.values()) for v in stats['overlaps'].values())
manifest=json.loads((SOURCE/'stats.json').read_text())
stats['checks']['all_sha256_match_committed_manifest']=all(stats['splits'][name]['sha256']==manifest['sha256'][f'{name}.jsonl'] for name in splits)

selected={}
for name,rows in splits.items():
    tags=['colloquial_particles','nested_negation','multicategory','original_semantics']
    selected[name]={tag:[r for r in sorted(rows,key=lambda r:(len(r['description']),r['item_id'])) if r.get('topic','original_semantics')==tag][:5] for tag in tags}
(OUT/'statistics.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
(OUT/'examples.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stats,ensure_ascii=False,indent=2))
