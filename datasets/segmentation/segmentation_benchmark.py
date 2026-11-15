"""Read-only TF-IDF segmentation ablation; fit decision thresholds on calibration only.

Run on the server from anywhere with the trusted local Python environment.
The saved classifier and API are never changed. Public test is diagnostic only.
"""
import collections
import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path('/srv/Hakaton')
REPO = ROOT / 'innopolis_2026'
IDS = (102, 103, 104, 105, 106)
NAMES = {'Штукатурные работы': 102, 'Плиточные работы': 103, 'Сантехника': 104,
         'Электрика': 105, 'Поклейка обоев и малярные работы': 106}
OUT = ROOT / 'results/segmentation_v1'
# Punctuation boundaries, but not decimal numbers or arbitrary commas/semicolons.
SENTENCE = re.compile(r'(?<=[.!?])\s+(?=[«"А-ЯЁA-Z0-9])')
BULLET = re.compile(r'^\s*(?:[-–—•*▪▫◦✔✅☑🔹🔸🔨🛠⭐]+\s*|\d{1,2}[.)]\s+)')
ANSWER = re.compile(r'^(?:нет|да|не\s+(?:делаем|выполняем|берём|занимаемся)|конечно|выполняем)\b', re.I)
ELLIPTIC_DENIAL = re.compile(r'^(?:нет|не\s+(?:делаем|выполняем|берём|занимаемся))\s*[.!?]?$', re.I)


def units(text):
    """Split on lines, bullets and sentences; keep an answer with its question.

    This is deliberately generic: no knowledge of category words or the synthetic
    generator's 'What we do' marker. Negation is NOT solved by this split.
    """
    result = []
    for line in text.splitlines():
        line = BULLET.sub('', line).strip()
        if not line:
            continue
        parts = [part.strip() for part in SENTENCE.split(line) if part.strip()]
        for part in parts:
            if result and ((result[-1].rstrip().endswith('?') and ANSWER.match(part))
                           or ELLIPTIC_DENIAL.fullmatch(part)):
                result[-1] += ' ' + part
            else:
                result.append(part)
    return result or [text.strip()]


def fragments(text):
    base = units(text)
    return {
        'units': base,
        'window2': [' '.join(base[i:i+2]) for i in range(len(base))] if len(base)>1 else base,
        'window3': [' '.join(base[i:i+3]) for i in range(len(base))] if len(base)>2 else base,
    }


def read_jsonl(path):
    return [json.loads(line) for line in path.open(encoding='utf-8') if line.strip()]


def read_parquet(path):
    import pyarrow.parquet as pq
    return [{'title': row['title'], 'description': row['description'], 'mcId': 101,
             'labels': [NAMES[name] for name in row['categories']]}
            for row in pq.read_table(path).to_pylist()]


def truth(rows):
    return np.asarray([[int(mid in (set(row['labels']) if 'labels' in row else
                                    {item['mcId'] for item in row['found_microcategories']})
                          and mid != row.get('mcId', 101)) for mid in IDS] for row in rows], dtype=np.int8)


def infer_rows(rows, model):
    """Batch vectorization; retain maximum score and its evidence, not every score."""
    pipeline = model['pipeline']
    start = perf_counter()
    full = pipeline.predict_proba([row['description'] for row in rows])
    timings = {'full': round(perf_counter() - start, 3)}
    outputs = {'full': full}
    evidence = {}
    sizes = []
    segment_sets = [fragments(row['description']) for row in rows]
    sizes = [len(s['units']) for s in segment_sets]
    for method in ('units', 'window2', 'window3'):
        start = perf_counter()
        scores = np.zeros((len(rows), len(IDS)))
        best = [[''] * len(IDS) for _ in rows]
        for batch_start in range(0, len(rows), 24):
            block = segment_sets[batch_start:batch_start+24]
            flat = [fragment for row in block for fragment in row[method]]
            probabilities = np.concatenate([pipeline.predict_proba(flat[i:i+512])
                                             for i in range(0, len(flat), 512)])
            cursor = 0
            for offset, segment_set in enumerate(block):
                texts = segment_set[method]
                chunk = probabilities[cursor:cursor+len(texts)]
                cursor += len(texts)
                indexes = np.argmax(chunk, axis=0)
                scores[batch_start+offset] = chunk[indexes, np.arange(len(IDS))]
                best[batch_start+offset] = [texts[int(i)] for i in indexes]
        outputs[method] = scores
        evidence[method] = best
        timings[method] = round(perf_counter() - start, 3)
    for method in ('units', 'window2', 'window3'):
        outputs['full+'+method] = np.maximum(full, outputs[method])
        timings['full+'+method] = round(timings['full'] + timings[method], 3)
    return outputs, evidence, timings, sizes


def calibrate(scores, y):
    # Fixed grid, selected exclusively using public calibration labels.
    # Prefer higher threshold when F1 ties, to avoid gratuitous false positives.
    grid = np.round(np.arange(.30, 1.0, .02), 2)
    chosen = []
    for j in range(len(IDS)):
        values = [(f1_score(y[:,j], scores[:,j] >= th, zero_division=0), float(th)) for th in grid]
        chosen.append(max(values)[1])
    return chosen


def evaluate(scores, rows, thresholds):
    y = truth(rows)
    yp = (scores >= thresholds).astype(np.int8)
    for i, row in enumerate(rows):
        if row.get('mcId', 101) in IDS:
            yp[i, IDS.index(row['mcId'])] = 0
    return {
        'n': len(rows), 'precision': round(precision_score(y, yp, average='micro', zero_division=0), 4),
        'recall': round(recall_score(y, yp, average='micro', zero_division=0), 4),
        'micro_f1': round(f1_score(y, yp, average='micro', zero_division=0), 4),
        'macro_f1': round(f1_score(y, yp, average='macro', zero_division=0), 4),
        'fp': int(((yp == 1) & (y == 0)).sum()), 'fn': int(((yp == 0) & (y == 1)).sum()),
        'empty': int((yp.sum(axis=1) == 0).sum()),
        'exact': round(float(np.all(y == yp, axis=1).mean()), 4),
        'f1_by_category': {str(mid): round(f1_score(y[:,j], yp[:,j], zero_division=0), 4)
                           for j,mid in enumerate(IDS)},
    }


def top_cases(rows, full, segmented, best, thresholds, limit=12):
    """Find per-category recovered labels and introduced FP; no post-hoc filter."""
    y = truth(rows)
    old = full >= .5
    new = segmented >= thresholds
    recovered, introduced = [], []
    for i, row in enumerate(rows):
        for j, mid in enumerate(IDS):
            if mid == row.get('mcId', 101):
                continue
            if not old[i,j] and new[i,j] and y[i,j]:
                target = recovered
            elif not old[i,j] and new[i,j] and not y[i,j]:
                target = introduced
            else:
                continue
            target.append({'index': i, 'item_id': row.get('item_id'), 'mcId': mid,
                           'title': row.get('title'),
                           'description': row['description'],
                           'gold': [m for k,m in enumerate(IDS) if y[i,k]],
                           'full_score': round(float(full[i,j]), 3),
                           'fragment_score': round(float(segmented[i,j]), 3),
                           'best_fragment': best[i][j]})
    recovered.sort(key=lambda r: r['fragment_score']-r['full_score'], reverse=True)
    introduced.sort(key=lambda r: r['fragment_score']-r['full_score'], reverse=True)
    def diverse(examples):
        selected, seen = [], set()
        for item in examples:
            key = (item['mcId'], ' '.join(item['best_fragment'].casefold().split()))
            if key not in seen:
                selected.append(item)
                seen.add(key)
            if len(selected) == limit:
                break
        return selected
    return {'recovered_total': len(recovered), 'introduced_fp_total': len(introduced),
            'recovered_examples': diverse(recovered), 'introduced_fp_examples': diverse(introduced)}


def quantiles(scores, rows):
    y = truth(rows)
    return {str(mid): {'positive': [round(float(q),3) for q in np.quantile(scores[y[:,j]==1,j], [.1,.5,.9])],
                       'negative': [round(float(q),3) for q in np.quantile(scores[y[:,j]==0,j], [.5,.9,.99])]}
            for j,mid in enumerate(IDS)}


def main():
    model = joblib.load(ROOT/'models/baseline/tfidf_baseline.joblib')  # Trusted local artifact.
    assert tuple(model['ids']) == IDS and np.allclose(model['thresholds'], .5)
    datasets = {
        'calibration': read_jsonl(ROOT/'data/processed/calibration.jsonl'),
        'public_validation': read_jsonl(ROOT/'data/processed/validation.jsonl'),
        'synthetic_300': read_parquet(REPO/'datasets/public/public-test-synthetic-300.parquet'),
        'public_test_diagnostic': read_parquet(REPO/'datasets/public/public-test-dataset.parquet'),
    }
    longer = REPO/'datasets/postoyanka/splits/test.jsonl'
    if longer.exists():
        datasets['new_long_test_diagnostic'] = read_jsonl(longer)
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    per_set = {}
    for name, rows in datasets.items():
        scores, evidence, times, lengths = infer_rows(rows, model)
        per_set[name] = (rows, scores, evidence)
        results[name] = {'n': len(rows), 'fragment_count': {'median': int(np.median(lengths)),
                         'p90': int(np.percentile(lengths, 90)), 'max': max(lengths)},
                         'inference_seconds': times}
        print('inferred', name, len(rows), 'median units',int(np.median(lengths)), 'times',times,flush=True)
    cal_rows, cal_scores, _ = per_set['calibration']
    targets = {'full': [.5]*5}
    for method in ('units','window2','window3','full+units','full+window2','full+window3'):
        targets[method] = calibrate(cal_scores[method], truth(cal_rows))
    for name, (rows, scores, evidence) in per_set.items():
        results[name]['metrics'] = {}
        for method, threshold in targets.items():
            results[name]['metrics'][method] = {
                'calibrated': evaluate(scores[method], rows, threshold),
                'raw_0_5_diagnostic': evaluate(scores[method], rows, [.5]*5),
            }
        print(name, [(m, d['calibrated']['micro_f1'],d['calibrated']['fp'],d['calibrated']['fn'])
                     for m,d in results[name]['metrics'].items()],flush=True)
    # Diagnostic strata only; never used to choose a threshold or model variant.
    synthetic_rows, synthetic_scores, _ = per_set['synthetic_300']
    strata = {
        'question_then_refusal': [i for i,r in enumerate(synthetic_rows)
                                  if 'В сообщениях часто спрашивают:' in r['description']],
        'other_contractor': [i for i,r in enumerate(synthetic_rows)
                             if 'заказчик поручает другим' in r['description']],
        'customer_wishes': [i for i,r in enumerate(synthetic_rows)
                            if 'это пожелания к готовому помещению' in r['description']],
        'positive_with_excluded_service': [i for i,r in enumerate(synthetic_rows)
                                           if r['labels'] and 'Обратите внимание:' in r['description']],
        'positive_all_five': [i for i,r in enumerate(synthetic_rows) if len(r['labels']) == 5],
    }
    results['synthetic_300']['strata'] = {
        group: {method: evaluate(synthetic_scores[method][index],
                                 [synthetic_rows[i] for i in index], targets[method])
                for method in ('full','units','window3','full+window3')}
        for group,index in strata.items() if index
    }
    result = {'model': 'trusted public-only baseline, five heads, no retraining',
              'calibration_only_thresholds': targets,
              'calibration_score_quantiles': {m: quantiles(cal_scores[m], cal_rows)
                                              for m in ('full','units','window2','window3')},
              'datasets': results,
              'caveat': 'Synthetic/public tests are diagnostics, never used for thresholds.'}
    (OUT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    cases = {}
    for name in ('public_validation','synthetic_300','public_test_diagnostic','new_long_test_diagnostic'):
        if name not in per_set:
            continue
        rows, scores, evidence = per_set[name]
        cases[name] = top_cases(rows,scores['full'],scores['units'],evidence['units'],
                                targets['units'])
    (OUT/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2),encoding='utf-8')
    print('saved',OUT/'metrics.json',OUT/'cases.json')


if __name__ == '__main__':
    main()
