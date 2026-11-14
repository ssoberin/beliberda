"""Report real and synthetic held-out quality separately for mixed training."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .evaluate_v2_experiment import IDS, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/mixed-charwb-comparison'))
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.root / 'splits/validation.jsonl').read_text().splitlines()]
    challenge = [json.loads(line) for line in (args.root / 'splits/challenge.jsonl').read_text().splitlines()]
    sets = {'combined_test': rows,
            'public_test': [row for row in rows if row['dataset'] == 'public'],
            'v2_test': [row for row in rows if row['dataset'] == 'v2'], 'challenge': challenge}
    result = {'models': {}, 'evaluation_scope': IDS, 'threshold': 0.5,
              'note': 'Training is public deduplicated train plus v2 train. Ready held-out splits unchanged. Public labels are weak; v2 and challenge are synthetic.'}
    for name in ('char_wb', 'char_wb_60k', 'char_wb_100k'):
        model = joblib.load(args.root / 'runs' / name / 'tfidf_baseline.joblib')
        if tuple(model['ids']) != IDS or not np.array_equal(model['thresholds'], np.full(5, 0.5)):
            raise ValueError('Unexpected heads or thresholds')
        scored = {}
        for label, split in sets.items():
            gold = np.asarray([[int(mid in row['evaluation_labels']) for mid in IDS] for row in split], dtype=np.int8)
            predicted = (model['pipeline'].predict_proba([row['description'][:6000] for row in split]) >= model['thresholds']).astype(np.int8)
            for i, row in enumerate(split):
                if row['mcId'] in IDS:
                    predicted[i, IDS.index(row['mcId'])] = 0
            scored[label] = metrics(gold, predicted)
        saved = json.loads((args.root / 'runs' / name / 'measurement.json').read_text())
        for key in ('micro_f1', 'macro_f1', 'micro_precision', 'micro_recall'):
            if saved[key] != scored['combined_test'][key]:
                raise ValueError('Independent combined evaluation differs')
        result['models'][name] = scored
        print(name, {label: round(value['micro_f1'], 6) for label, value in scored.items()}, flush=True)
    (args.root / 'evaluation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Обучение public + v2: результаты по источникам', '',
             'Для каждого размера заново обучены TF-IDF и пять SGD. Обучение: 3398 уникальных public-объявлений + 8464 v2-примера; остальные части сохранены для проверки. Все пороги 0.5, word (1,3), char_wb (3,5), sublinear_tf=True; SGD random_state42.',
             'Дедупликация public выполнена до старого разбиения: 10000 исходных строк →4855 уникальных нормализованных описаний. Источники и готовые части v2 не пересмешивались; веса строк не менялись.', '',
             '| Вариант | Проверка | Строк | Micro-F1 | Macro-F1 | Precision micro | Recall micro | FP | FN | Exact set accuracy |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, scored in result['models'].items():
        for label, m in scored.items():
            lines.append(f"| {name} | {label} | {m['rows']} | {m['micro_f1']:.6f} | {m['macro_f1']:.6f} | {m['micro_precision']:.6f} | {m['micro_recall']:.6f} | {m['fp']} | {m['fn']} | {m['exact_set_accuracy']:.6f} |")
    lines += ['', 'Общий результат объединяет разные распределения; он не заменяет отдельные public/v2/challenge метрики. Public-разметка автоматическая, синтетический v2 не представляет реальный поток. Нормализованных точных повторов и пересечений между частями нет; похожие public-шаблоны могут остаться.',
              'Фактические размерности, время и память: runs/comparison.md. Происхождение, состав смеси и SHA-256: splits/stats.json. Рабочие модели не заменены.']
    (args.root / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
