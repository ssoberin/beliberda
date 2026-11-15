"""Compare exported categories against a separate reference; never trains a model."""
import argparse
import csv
import json
from pathlib import Path
from collections import Counter


def rows_by_case(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    result = {}
    for row in rows:
        key = row['testcase']
        if key in result:
            raise ValueError(f'Duplicate testcase: {key}')
        result[key] = row
    return result


def audit(drafts, reference, source):
    predicted, expected, inputs = map(rows_by_case, (drafts, reference, source))
    if predicted.keys() != expected.keys() or predicted.keys() != inputs.keys():
        raise ValueError('Input files contain different testcase sets.')
    correct = true_positive = false_positive = false_negative = 0
    known_rows = known_correct = known_tp = known_fp = known_fn = 0
    per_category = {cid:Counter() for cid in range(102,107)}
    errors, mismatches, reasons = [], [], Counter()
    for case, row in predicted.items():
        got, target = set(json.loads(row['microcats']) or []), set(json.loads(expected[case]['microcats']))
        error = json.loads(row.get('error') or 'null')
        known = json.loads(row['microcats']) is not None and not (
            isinstance(error, dict) and error.get('code') in ('validation_error','internal_error','service_unavailable'))
        size = len(inputs[case]['description'].strip())
        correct += got == target
        true_positive += len(got & target)
        false_positive += len(got - target)
        false_negative += len(target - got)
        if known:
            known_rows += 1
            known_correct += got == target
            known_tp += len(got & target)
            known_fp += len(got - target)
            known_fn += len(target - got)
        for cid, counts in per_category.items():
            counts['tp'] += cid in got and cid in target
            counts['fp'] += cid in got and cid not in target
            counts['fn'] += cid not in got and cid in target
        detail = {'testcase':case, 'description_chars':size, 'predicted':sorted(got),
                  'reference':sorted(target), 'missing':sorted(target-got), 'extra':sorted(got-target),
                  'classification_available':known}
        if error:
            errors.append({**detail, 'error':error})
            for item in error if isinstance(error, list) else [error]:
                reasons[item.get('code', 'unknown')] += 1
        if got != target:
            mismatches.append({**detail, 'has_error':bool(error)})
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
    denominator = 2 * true_positive + false_positive + false_negative
    return {'sources':{'drafts':str(Path(drafts).resolve()), 'reference':str(Path(reference).resolve()),
                       'input':str(Path(source).resolve())},
            'metrics':{'rows':len(predicted), 'exact_matches':correct, 'exact_match_rate':correct / len(predicted),
                       'true_positive_labels':true_positive, 'false_positive_labels':false_positive,
                       'false_negative_labels':false_negative, 'micro_precision':precision, 'micro_recall':recall,
                       'micro_f1':2*true_positive/denominator if denominator else 0,
                       'classification_available_rows':known_rows, 'available_exact_matches':known_correct,
                       'available_exact_rate':known_correct/known_rows if known_rows else None,
                       'available_micro_f1':2*known_tp/(2*known_tp+known_fp+known_fn) if 2*known_tp+known_fp+known_fn else 0},
            'per_category':{cid:dict(counts) for cid,counts in per_category.items()},
            'error_counts':dict(reasons), 'errors':errors, 'mismatches':mismatches}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drafts', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.drafts, args.reference, args.input)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'metrics':result['metrics'], 'error_counts':result['error_counts'],
                      'report':str(path)}, ensure_ascii=False))
