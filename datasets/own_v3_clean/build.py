"""Exclude the user-approved 79 realism candidates from original v3 train only."""
import argparse,collections,hashlib,json,shutil,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCOPE=set(range(102,107))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def parse(lines):return [json.loads(line) for line in lines if line.strip()]
def counts(rows):
    return {'rows':len(rows),'sources':dict(sorted(collections.Counter(str(r['mcId']) for r in rows).items())),
            'offered_scope_positives':{str(c):sum(c in r['offered_microcategories'] for r in rows) for c in sorted(SCOPE)},
            'target_scope_positives':{str(c):sum(c in {x['mcId'] for x in r['found_microcategories']} for r in rows) for c in sorted(SCOPE)},
            'target_scope_cardinality':dict(sorted(collections.Counter(str(len({x['mcId'] for x in r['found_microcategories']}&SCOPE)) for r in rows).items())),
            'median_characters':statistics.median(len(r['description']) for r in rows)}
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--audit',type=Path,default=Path('output/v3-realism-screen/priority79.json'))
    p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();assert args.base.resolve()!=args.output.resolve()
    audit=json.loads(args.audit.read_text());ids=set(audit['priority_ids']);assert len(ids)==79
    source_paths={s:args.base/f'{s}.jsonl' for s in ['train','validation','test','challenge']}
    source_hashes={s:sha(p) for s,p in source_paths.items()}
    raw=source_paths['train'].read_text().splitlines(keepends=True);rows=parse(raw)
    assert len(rows)==3000 and len({r['item_id'] for r in rows})==3000
    keep_lines=[];removed_lines=[]
    for line in raw:
        if not line.strip():continue
        row=json.loads(line)
        (removed_lines if row['item_id'] in ids else keep_lines).append(line)
    kept=parse(keep_lines);removed=parse(removed_lines)
    assert len(kept)==2921 and len(removed)==79
    assert {r['item_id'] for r in removed}==ids
    assert not {r['item_id'] for r in kept}&ids
    args.output.mkdir(parents=True,exist_ok=True)
    quarantine=args.output/'quarantine';quarantine.mkdir(exist_ok=True)
    (args.output/'train.jsonl').write_text(''.join(keep_lines))
    (quarantine/'excluded_train_79.jsonl').write_text(''.join(removed_lines))
    for split in ['validation','test','challenge']:
        shutil.copyfile(source_paths[split],args.output/f'{split}.jsonl')
        assert sha(args.output/f'{split}.jsonl')==source_hashes[split]
    # No text/label edits, no train-to-test moves, no source mutation.
    assert source_hashes=={s:sha(p) for s,p in source_paths.items()}
    assert parse((args.output/'train.jsonl').read_text().splitlines())==[r for r in rows if r['item_id'] not in ids]
    heldouts=[r for s in ['validation','test','challenge'] for r in parse((args.output/f'{s}.jsonl').read_text().splitlines())]
    norm=lambda t:' '.join(t.casefold().split())
    assert not {norm(r['description']) for r in kept}&{norm(r['description']) for r in heldouts}
    stats={'base':'dataset_v3@0882f80 revision2','operation':'User-approved exclusion of 79 extreme repetition candidates from train only; original files and all heldout files preserved.',
           'before':counts(rows),'after':counts(kept),'removed':counts(removed),
           'removed_percent':100*len(removed)/len(rows),'split_rows':{'train':len(kept),**{s:len(parse((args.output/f'{s}.jsonl').read_text().splitlines())) for s in ['validation','test','challenge']}},
           'source_sha256':source_hashes,'output_sha256':{s:sha(args.output/f'{s}.jsonl') for s in source_paths},
           'audit_sha256':sha(args.audit),'quarantine_sha256':sha(quarantine/'excluded_train_79.jsonl'),
           'checks':{'exactly79_excluded':True,'retained_records_unchanged':True,'heldout_bytes_unchanged':True,'original_files_unchanged':True,'train_heldout_text_overlap':0},
           'warning':'Realism judgement is heuristic, not proof of wrong labels. Label/source balance changes; no model improvement has been measured. Quarantine rows were seen in previous training and are not an independent test.'}
    (args.output/'stats.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2)+'\n')
    manifest={'action':'EXCLUDED_FROM_CLEAN_TRAIN','base':'dataset_v3@0882f80 revision2','excluded_ids':sorted(ids),'reason':'At least5 identical actor/FAQ/old-offer-correction/price constructions in one text. User approved exclusion; not asserted label errors.','quarantine':'quarantine/excluded_train_79.jsonl','all_other_rows_and_holdouts_unchanged':True}
    (args.output/'exclusion_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'before':stats['before'],'after':stats['after'],'removed':stats['removed'],'checks':stats['checks']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
