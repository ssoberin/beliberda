# Fixed public train with synthetic additions

Results: [report.md](report.md). Executed notebook: [fractions.ipynb](fractions.ipynb).
Train/held-out diagnostic: [overfitting_diagnostic.json](overfitting_diagnostic.json).
The notebook reads saved measurements and does not fit models.

The original v3 files were not edited. The training pool combines
`dataset_v3@0882f80:datasets/own_v3/train.jsonl` (3000 rows) with
`datasets/context_pairs_100/data.jsonl` (100 rows).
The separate `datasets/context_pairs_100/test_20.jsonl` is never used for fitting.
All public training rows remain in all four runs. Thresholds remain 0.5.

## Reproduction

Run from the repository root in an environment with the versions recorded in
`p000/model/measurement.json` (plus duckdb; nbformat, nbclient, ipykernel and
matplotlib for rendering). Use fresh output directories for new measurements.

```bash
git worktree add --detach /tmp/innopolis-dataset-v3-review 0882f80
python -B -m unittest discover -s datasets/context_pairs_100 -p 'test_*.py' -v
python -m avito_compare.scripts.prepare_dataset --input datasets/public/public-train-annotated.jsonl --output output/v3-fraction-public-splits --seed 42
python -m avito_compare.scripts.run_v3_fraction_study --v3 /tmp/innopolis-dataset-v3-review/datasets/own_v3 --public output/v3-fraction-public-splits
python output/v3-fraction-study/check_fit_gap.py
python output/v3-fraction-study/build_notebook.py
```

The runner requires a new output folder: move the saved report folder aside or
pass `--output` to an empty destination when rerunning. The diagnostic/rendering
scripts refer to the canonical `output/v3-fraction-study` folder.
Weights and prepared JSONL split copies are local, excluded from Git; metrics,
source hashes, selected synthetic IDs, reports and plots are committed.
Calibration-search results in trainer diagnostics are not the scores reported
here: predictions use the fixed saved thresholds 0.5 in all runs.
Public 729 is validation. Official 208 has known conflicts and training overlap;
cleaned 114 is a separate, smaller diagnostic, not a new representative test.
The authored control contains 10 paired scenarios, not 20 independent scenarios.
