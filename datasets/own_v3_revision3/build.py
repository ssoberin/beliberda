"""Replace 250 simple train rows, retaining offered training labels and targets."""
import argparse,collections,hashlib,json,re
from pathlib import Path
from scenarios import SCENARIOS
ROOT=Path(__file__).resolve().parent
NAMES={101:'Ремонт под ключ',102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',105:'Электрика',106:'Поклейка обоев и малярные работы'}
SCOPE=set(range(102,107))
def read(path):return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
def norm(text):return ' '.join(text.casefold().split())
def encoded(rows):return ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def build(base):
    assert len(base)==3000 and len(SCENARIOS)==50
    def signature(r):
        offered=set(r['offered_microcategories'])&SCOPE
        gold={c['mcId'] for c in r['found_microcategories']}&SCOPE
        return tuple(sorted(offered)),tuple(sorted(gold))
    def basic_score(r):
        text=r['description'].casefold()
        # Prefer little contextual machinery and short descriptions, without model scores.
        context_markers=len(re.findall(r'\b(?:если|раньше|теперь|друг\w*|заказчик\w*|хотя|кроме|только|вместо|отдельно|спросил\w*)\b',text))
        return (context_markers,len(r.get('context_spans',[])),len(text),r['item_id'])
    buckets=collections.defaultdict(list)
    for index,row in enumerate(base):buckets[signature(row)].append((index,row))
    for bucket in buckets.values():bucket.sort(key=lambda p:basic_score(p[1]))
    result=list(base);new=[];manifest=[];used=set()
    for group,case in enumerate(SCENARIOS,1):
        for variant,marked in enumerate(case['variants'],1):
            assert marked[:2] in ('+ ','- ')
            offered=[case['category']] if marked[0]=='+' else []
            desired=(tuple(offered),tuple(offered))
            candidates=buckets[desired]
            assert candidates,('Not enough matching simple rows',desired)
            index,old=candidates.pop(0);assert index not in used;used.add(index)
            text=case['context']+'\n\n'+marked[2:]
            full_offered=sorted(set(offered)|({101} if case['topic']=='bundle_boundary' else set()))
            source=old['mcId'];assert source not in offered
            row={'item_id':f'own-v3-r3-{group:03d}-{variant}','group_id':f'own-v3-r3-scene-{group:03d}',
                 'title':marked[2:].split('.')[0][:90],'description':text,'mcId':source,
                 'offered_microcategories':full_offered,
                 'found_microcategories':[{'mcId':c,'label':NAMES[c]} for c in full_offered if c!=source],
                 'evidence':[{'mcId':c,'start':0,'end':len(text),'quote':text} for c in full_offered],
                 'topic':case['topic'],'variant':variant,
                 'annotation_rationale':('Услуга доступна как текущий самостоятельный заказ.' if offered else 'Нет текущего самостоятельного предложения целевых услуг; решающий контекст в последнем абзаце.'),
                 'provenance':'revision3_authored_contrastive_scenario',
                 'annotation_status':'author_labeled_self_reviewed_not_independent_gold',
                 'replaces_item_id':old['item_id']}
            # Source 101 is outside our five heads; source-specific category101 can remain in raw gold.
            assert signature(row)==signature(old),(row,signature(old))
            assert len(text)<=6000
            result[index]=row;new.append(row)
            manifest.append({'old_item_id':old['item_id'],'new_item_id':row['item_id'],'train_index':index,'group_id':row['group_id'],'source_mcId':source,'scope_training_labels':offered,'scope_target_labels':offered,'old_characters':len(old['description']),'new_characters':len(text),'old_basic_score':list(basic_score(old))})
    assert len(result)==3000 and len(new)==250 and len({norm(r['description']) for r in result})==3000
    assert collections.Counter(signature(r) for r in base)==collections.Counter(signature(r) for r in result)
    assert collections.Counter(r['mcId'] for r in base)==collections.Counter(r['mcId'] for r in result)
    return result,new,manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);args=p.parse_args()
    base=read(args.base/'train.jsonl');result,new,manifest=build(base)
    (ROOT/'train.jsonl').write_text(encoded(result));(ROOT/'replacements_250.jsonl').write_text(encoded(new))
    (ROOT/'replacement_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    stats={'base_revision':'dataset_v3@0882f80','base_files_sha256':{s:digest(args.base/f'{s}.jsonl') for s in ['train','validation','test','challenge']},'train_rows':3000,'replaced':250,'new_scenario_groups':50,'variants_per_group':5,'scope':sorted(SCOPE),'topic_counts':dict(collections.Counter(r['topic'] for r in new)),'replacement_training_positives':{str(c):sum(c in r['offered_microcategories'] for r in new) for c in sorted(SCOPE)},'empty_scope_replacements':sum(not (set(r['offered_microcategories'])&SCOPE) for r in new),'new_train_sha256':digest(ROOT/'train.jsonl'),'new_replacements_sha256':digest(ROOT/'replacements_250.jsonl'),'scope_training_and_target_histograms_preserved':True,'source_histogram_preserved':True,'old_selected_median_chars':float(__import__('statistics').median(m['old_characters'] for m in manifest)),'new_median_chars':float(__import__('statistics').median(len(r['description']) for r in new)),'annotation':'Author labels and self-review; no independent human gold. Related variants count as groups, not independent examples.'}
    (ROOT/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
