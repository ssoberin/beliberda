"""Read-only consistency audit. No training, predictions, threshold tuning or label edits."""
import collections
import hashlib
import json
from pathlib import Path
import duckdb

P = Path('output/public-test-quality-audit')
SOURCE = Path('datasets/public/public-test-dataset.parquet')
IDS = {'Штукатурные работы':102, 'Плиточные работы':103, 'Сантехника':104,
       'Электрика':105, 'Поклейка обоев и малярные работы':106}

def norm(text):
    return ' '.join(text.casefold().split())

def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

def group_stats(rows, key):
    groups = collections.defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    conflicts = []
    min_label_errors = min_exact_errors = 0
    cells = []
    positive_total = sum(len(r['gold']) for r in rows)
    for group in groups.values():
        votes = collections.Counter(tuple(r['gold']) for r in group)
        min_exact_errors += len(group) - max(votes.values())
        for category in range(102,107):
            pos = sum(category in r['gold'] for r in group)
            min_label_errors += min(pos, len(group)-pos)
            cells.append((pos, len(group)))
        if len(votes)>1:
            conflicts.append({'rows':[r['row'] for r in group],
                              'label_variants':[{'gold':list(g), 'count':n} for g,n in votes.items()],
                              'title':group[0]['title'], 'description':group[0]['description']})
    # Optimistic combinatorial upper bound for a deterministic lookup on this key.
    # For selected group/category cells: F1=2*sum(positive votes)/(all gold positives + predicted count).
    # Maximizer is a prefix of cells ordered by positive-vote proportion; this is not a trained model.
    tp = predicted = 0
    best = 0.
    for pos, total in sorted(cells, key=lambda cell: cell[0]/cell[1], reverse=True):
        tp += pos
        predicted += total
        best = max(best, 2*tp/(positive_total+predicted))
    return {'unique':len(groups), 'duplicate_groups':sum(len(g)>1 for g in groups.values()),
            'rows_in_duplicate_groups':sum(len(g) for g in groups.values() if len(g)>1),
            'excess_duplicate_rows':len(rows)-len(groups),
            'conflicting_groups':len(conflicts), 'conflicting_rows':sum(len(g['rows']) for g in conflicts),
            'minimum_unavoidable_binary_label_errors':min_label_errors,
            'minimum_unavoidable_exact_match_errors':min_exact_errors,
            'maximum_possible_micro_f1_same_key_same_prediction':best}, conflicts

def main():
    P.mkdir(exist_ok=True, parents=True)
    con = duckdb.connect()
    schema = con.execute('describe select * from read_parquet(?)',[str(SOURCE)]).fetchall()
    raw = con.execute('select title,description,categories from read_parquet(?)',[str(SOURCE)]).fetchall()
    assert all(t is not None and d is not None and cats is not None for t,d,cats in raw)
    assert all(c in IDS for _,_,cats in raw for c in cats)
    assert all(len(cats)==len(set(cats)) for _,_,cats in raw)
    rows = [{'row':i,'title':t,'description':d,'gold':sorted(IDS[c] for c in cats)}
            for i,(t,d,cats) in enumerate(raw)]
    stats = {}
    conflict_details = {}
    for name,key in [('raw_description',lambda r:r['description']),
                     ('normalized_description',lambda r:norm(r['description'])),
                     ('raw_title_description',lambda r:(r['title'],r['description'])),
                     ('normalized_title_description',lambda r:(norm(r['title']),norm(r['description'])))]:
        stats[name], conflict_details[name] = group_stats(rows,key)
    overlaps = {}
    for name,path in [('public_raw_10000','datasets/public/public-train-annotated.jsonl'),
                      ('actual_pub_train_3398','output/ngram-comparison/splits/train.jsonl'),
                      ('public_calibration_728','output/ngram-comparison/splits/calibration.jsonl'),
                      ('public_validation_729','output/ngram-comparison/splits/validation.jsonl'),
                      ('v3_train_2921','datasets/own_v3_clean/train.jsonl')]:
        keys = {norm(r['description']) for r in read(path)}
        ids = [r['row'] for r in rows if norm(r['description']) in keys]
        overlaps[name] = {'rows':len(ids), 'unique_descriptions':len({norm(rows[i]['description']) for i in ids}), 'row_ids':ids}
    summary = {'source':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
               'rows':len(rows),'schema':schema,'null_fields':0,'unknown_categories':0,
               'duplicate_labels_within_row':0,'empty_description':sum(not r['description'].strip() for r in rows),
               'empty_gold':sum(not r['gold'] for r in rows),'gold_label_total':sum(len(r['gold']) for r in rows),
               'cardinality':dict(collections.Counter(len(r['gold']) for r in rows)),
               'groups':stats,'overlaps':overlaps,
               'bound_scope':'Label-consistency audit only: same observed input, same deterministic prediction. Optimistic oracle bound using gold; not model performance, not a suggested predictor. source_mcId=101 is assumed from README and missing from parquet.'}
    (P/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    (P/'conflicts.json').write_text(json.dumps(conflict_details['raw_title_description'],ensure_ascii=False,indent=2)+'\n')
    (P/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
