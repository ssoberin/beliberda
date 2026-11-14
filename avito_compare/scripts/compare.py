"""Fair paired evaluation of TF-IDF vs Qwen, with resumable JSONL cache."""
import argparse
import json
import time
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, f1_score
from scripts.qwen_classifier import classify, IDS, as_drafts

TARGETS = (102, 103, 104, 105, 106)


def read_rows(path, count, seed):
    data = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not data:
        raise ValueError('validation dataset is empty')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data))[:count]
    return [(int(i), data[int(i)]) for i in order]


def report(gold, predicted):
    yt = np.asarray([[int(mid in row) for mid in TARGETS] for row in gold])
    yp = np.asarray([[int(mid in row) for mid in TARGETS] for row in predicted])
    p, r, f, _ = precision_recall_fscore_support(yt, yp, average='micro', zero_division=0)
    by_label = {}
    for i, mid in enumerate(TARGETS):
        pi, ri, fi, _ = precision_recall_fscore_support(yt[:, i], yp[:, i], average='binary', zero_division=0)
        by_label[str(mid)] = {'precision': round(float(pi), 4), 'recall': round(float(ri), 4), 'f1': round(float(fi), 4)}
    return {'micro_precision': round(float(p), 5), 'micro_recall': round(float(r), 5),
            'micro_f1': round(float(f), 5), 'by_label': by_label}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validation', type=Path, default=Path('data/processed/validation.jsonl'))
    ap.add_argument('--baseline', type=Path, default=Path('artifacts/tfidf_baseline.joblib'))
    ap.add_argument('--cache', type=Path, default=Path('artifacts/qwen_predictions.jsonl'))
    ap.add_argument('--output', type=Path, default=Path('artifacts/comparison.json'))
    ap.add_argument('--count', type=int, default=50)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--base-url', default='http://127.0.0.1:8000')
    ap.add_argument('--model', default='avito-qwen')
    args = ap.parse_args()
    if args.count <= 0:
        raise ValueError('--count must be positive')
    selection = read_rows(args.validation, args.count, args.seed)
    baseline = joblib.load(args.baseline)  # Trusted locally-generated model only.
    existing = {}
    if args.cache.exists():
        for line in args.cache.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                existing[item['key']] = item
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    gold, baseline_pred, qwen_pred, rows_out = [], [], [], []
    with args.cache.open('a', encoding='utf-8') as cache_file:
        for n, (idx, row) in enumerate(selection, 1):
            desc = row['description'][:6000]
            # source=101 for the weak train set supplied by organizers.
            source_mc_id = 101
            key = f'{args.validation.resolve()}|{idx}|{args.model}|{args.base_url}|{source_mc_id}'
            probs = baseline['pipeline'].predict_proba([desc])[0]
            b = sorted(mid for mid, prob, th in zip(baseline['ids'], probs, baseline['thresholds'])
                       if mid != source_mc_id and prob >= th)
            if key in existing:
                q = existing[key]['mcIds']
                elapsed = existing[key]['seconds']
            else:
                t = time.perf_counter()
                answer = classify(desc, source_mc_id, args.base_url, args.model)
                elapsed = time.perf_counter() - t
                q = answer['mcIds']
                entry = {'key': key, 'mcIds': q, 'seconds': round(elapsed, 3),
                         'rejected': answer['rejected'], 'usage': answer['usage']}
                cache_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                cache_file.flush()
                existing[key] = entry
            gold.append(row['labels'])
            baseline_pred.append(b)
            qwen_pred.append(q)
            rows_out.append({'index': idx, 'gold': row['labels'], 'baseline': b, 'qwen': q,
                             'seconds': elapsed, 'description': desc[:300]})
            print(f'[{n}/{len(selection)}] gold={row["labels"]} baseline={b} qwen={q} time={elapsed:.1f}s', flush=True)
    result = {'sample_size': len(selection), 'source_mcId': 101,
              'evaluation_note': 'Weak automated labels, not official or manually verified gold; fixed paired sample.',
              'baseline': report(gold, baseline_pred), 'qwen': report(gold, qwen_pred),
              'qwen_latency_seconds': {'median': round(float(np.median([r['seconds'] for r in rows_out])), 3),
                                       'p95': round(float(np.percentile([r['seconds'] for r in rows_out], 95)), 3)},
              'disagreements': [r for r in rows_out if r['baseline'] != r['qwen']]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nRESULT:')
    print(json.dumps({k: result[k] for k in ('sample_size', 'baseline', 'qwen', 'qwen_latency_seconds')}, ensure_ascii=False, indent=2))
    print('Saved:', args.output)

if __name__ == '__main__':
    main()
