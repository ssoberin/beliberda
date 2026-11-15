# Hard-negative V3 experiment

Run from repository root with the existing experiment Python environment.

1. `python datasets/v3_hard_negatives_200/build.py`
2. `python output/hard-negatives-v3-experiment/audit_dataset.py`
3. `python output/hard-negatives-v3-experiment/prepare.py` (freezes config; requires baseline caches)
4. `colab new -s avito-hn200 --gpu T4`
5. `colab exec -s avito-hn200 -f output/hard-negatives-v3-experiment/colab_setup.py --timeout 300`
6. `colab upload -s avito-hn200 output/hard-negatives-v3-experiment/cache/delta_chunks.json /content/hn200_delta.json`
7. `colab exec -s avito-hn200 -f output/hard-negatives-v3-experiment/colab_encode.py --timeout 300`
8. `colab download -s avito-hn200 /content/hn200_delta.npz output/hard-negatives-v3-experiment/cache/delta_checkpoint.npz`
9. `colab stop -s avito-hn200` (also stop on failure; do not leave a VM running)
10. `python output/hard-negatives-v3-experiment/train_b.py`
11. After training validation succeeds, `python output/hard-negatives-v3-experiment/evaluate_once.py`

The final scoring script refuses to overwrite existing metrics. Preserve prior
results when reproducing. Do not use test scores to revise data or thresholds.
Models and large caches remain local; hashes, config, folds, code, audits and
results are saved for review. Source directories are read-only inputs.

`prepare.py` reuses existing embeddings; only new chunk texts are sent to Colab.
The existing public-test embedding cache is accessed as a lookup only for needed
chunks; no public_synthetic_300 records, labels, fitting or metrics are used.
Baseline predictions are reused from the saved C_syntax evaluation and checked
against gold/index alignment, while its weights are retained unmodified.
OOF is recomputed completely with the existing grouped nested-CV routine.
Feature names and CatBoost parameters are unchanged; calibration is for original
early stopping only. No alternative thresholds or ablations are trained.

Read dataset_audit.md before interpreting results: supporting ad paragraphs
repeat across the new data, which limits diversity despite unique full ads.
