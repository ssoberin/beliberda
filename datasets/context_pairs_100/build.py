"""Serialize 50 authored context pairs. No model use and no random text generation."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from cases import PAIRS

ROOT = Path(__file__).resolve().parent
NAMES = {102:'Штукатурные работы',103:'Плиточные работы',104:'Сантехника',
         105:'Электрика',106:'Поклейка обоев и малярные работы'}
TOPICS = {'comparison':30,'partial_refusal':20,'actor':20,'status_binding':20,'time_condition':10}


def normalize(text):
    return ' '.join(text.casefold().split())


def build():
    rows=[]
    for i,case in enumerate(PAIRS,1):
        for variant in ('a','b'):
            prefix=case['context']+'\n\n'
            statement=case[variant]
            description=prefix+statement+'\n\n'+case['closing']
            offered=sorted(case['labels_'+variant])
            rows.append({'item_id':f'ctx100-{i:03d}-{variant}', 'group_id':f'ctx100-{i:03d}',
                'variant':variant,'description':description,'mcId':101,
                'offered_microcategories':offered,
                'found_microcategories':[{'mcId':c,'label':NAMES[c]} for c in offered],
                'topic':case['topic'],'pair_rationale':case['change'],
                'discriminating_span':{'start':len(prefix),'end':len(prefix)+len(statement),'quote':statement},
                'evidence':[{'mcId':c,'start':0,'end':len(prefix)+len(statement),
                             'quote':prefix+statement} for c in offered],
                'provenance':'individually_authored_context_pair',
                'annotation_status':'author_labeled_and_self_reviewed_not_independent_gold',
                'recommended_role':'training_supplement_keep_pairs_together'})
    return rows


def validate(rows):
    assert len(rows)==100
    assert len({r['item_id'] for r in rows})==100
    assert len({normalize(r['description']) for r in rows})==100
    assert collections.Counter(r['topic'] for r in rows)==TOPICS
    groups=collections.defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
        assert row['mcId']==101
        offered=row['offered_microcategories']
        assert offered==sorted(set(offered)) and set(offered)<=set(NAMES)
        assert row['found_microcategories']==[{'mcId':c,'label':NAMES[c]} for c in offered]
        assert len(row['description'])<=6000
        for e in [row['discriminating_span'],*row['evidence']]:
            assert row['description'][e['start']:e['end']]==e['quote']
        assert {e['mcId'] for e in row['evidence']}==set(offered)
    assert len(groups)==50
    for group in groups.values():
        assert len(group)==2 and {r['variant'] for r in group}=={'a','b'}
        assert group[0]['offered_microcategories']!=group[1]['offered_microcategories']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT)
    args=parser.parse_args()
    rows=build();validate(rows)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
    path=args.output_dir/'data.jsonl';path.write_text(data)
    from test_cases import build_test
    controls=build_test()
    assert len(controls)==20 and len({normalize(r['description']) for r in controls})==20
    assert not {normalize(r['description']) for r in rows} & {normalize(r['description']) for r in controls}
    control_path=args.output_dir/'test_20.jsonl'
    control_path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in controls))
    lengths=sorted(len(r['description']) for r in rows)
    stats={'rows':100,'independent_scenarios':50,'source_mcId':101,'scope':list(NAMES),
        'topic_counts':dict(collections.Counter(r['topic'] for r in rows)),
        'positive_category_counts':{str(c):sum(c in r['offered_microcategories'] for r in rows) for c in NAMES},
        'target_cardinality':dict(sorted(collections.Counter(len(r['offered_microcategories']) for r in rows).items())),
        'characters':{'min':lengths[0],'median':(lengths[49]+lengths[50])/2,'max':lengths[-1]},
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'separate_test':{'rows':20,'scenarios':10,'sha256':hashlib.sha256(control_path.read_bytes()).hexdigest()},
        'annotation':'Author labels plus explicit self-review; not independent human double annotation.',
        'role':'User-authorized training supplement; pairs remain together. Separate new20 test is held out. Not representative of production traffic.'}
    (args.output_dir/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    lines=['# 50авторских контрастных пар —100текстов','',
           'Вход модели: толькоdescription иmcId. Остальные поля раскрывают разметку. Source101; цели102–106. Метки авторские, независимой двойной человеческой разметки не было.','']
    for i,case in enumerate(PAIRS,1):
        lines += [f"## {i}. {case['topic']}",'',case['change'],'']
        for row in rows[(i-1)*2:i*2]:
            lines += [f"### {row['item_id']} — метки{row['offered_microcategories']}",'',row['description'],'']
    (args.output_dir/'examples.md').write_text('\n'.join(lines).rstrip()+'\n')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
