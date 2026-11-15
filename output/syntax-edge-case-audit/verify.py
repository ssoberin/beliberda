"""Verify recorded inference and protected inputs, without training or new inference.

Semantic failures are recorded, not hidden or asserted to be successes.
"""
import hashlib
import json
from pathlib import Path

P = Path('output/syntax-edge-case-audit')


def main():
    checked_hashes = 0
    checked_predictions = 0
    semantic_failures = []
    for relative in ['probe_predictions.json', 'public-inference/probe_predictions.json']:
        payload = json.loads((P / relative).read_text())
        assert payload['no_training'] and payload['threshold'] == .5
        for path, expected in payload['sha256'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
            checked_hashes += 1
        for row in payload['rows']:
            expected = sorted(int(c) for c, probability in row['probabilities'].items()
                              if probability >= .5 and int(c) != row['mcId'])
            assert row['prediction'] == expected, row['id']
            checked_predictions += 1
            if 'expert_expected' in row and row['prediction'] != row['expert_expected']:
                semantic_failures.append(row['id'])
    assert semantic_failures == ['partial_ellipsis'], semantic_failures
    old = json.loads((P / 'probe_predictions.json').read_text())
    assert len(old['reproduction']) == 2
    assert all(x['max_abs_probability_difference'] == 0
               and x['prediction_identical'] for x in old['reproduction'].values())
    public = {r['original_row_idx']: r for r in json.loads(
        (P / 'public-inference/probe_predictions.json').read_text())['rows']}
    assert 106 in public[1128]['prediction']
    assert 105 in public[7250]['prediction']
    assert 103 not in public[6401]['prediction']
    # Known genuine refusal failure must remain explicit in this frozen audit.
    assert {104, 105} <= set(public[1060]['prediction'])
    result = dict(artifact_consistency_passed=True, protected_hashes_checked=checked_hashes,
                  predictions_checked=checked_predictions,
                  identical_cached_control_predictions=2,
                  known_failed_manual_semantic_probes=semantic_failures,
                  known_public_refusal_failure=dict(row_idx=1060, false_positive_categories=[104, 105]),
                  note='Passing artifact checks does not mean the model passes all semantic cases.')
    (P / 'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
