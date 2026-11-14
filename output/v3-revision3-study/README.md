# Revision3 A/B experiment

[report.md](report.md) contains the results and limitations. [comparison.ipynb](comparison.ipynb)
is executed from saved JSON only and does not fit models.

Two fits: public3398 + v3train3000 + previous authored100. The after fit replaces
250 original v3 training rows while preserving source/training-target label
histograms and row order. Both retain fixed .5 thresholds and 46k features.
No tests or working model are edited; the revision is not recommended as a replacement.

Reproduction from repo root:

```bash
git worktree add --detach /tmp/innopolis-dataset-v3-review 0882f80
python datasets/own_v3_revision3/build.py --base /tmp/innopolis-dataset-v3-review/datasets/own_v3
python -B -m unittest discover -s datasets/own_v3_revision3 -p 'test_*.py' -v
python -m avito_compare.scripts.run_v3_revision_study
python output/v3-revision3-study/build_notebook.py
```

The runner expects the local prepared `output/v3-fraction-study/p100/splits` and
`output/v3-fraction-public-splits` from the earlier fraction study. Recreate these
using that study's [instructions](../v3-fraction-study/README.md). Its p100 model
is not used for prediction: both models are fitted again and baseline held-out
metrics must reproduce the saved p100 metrics exactly. Dependencies and versions
are recorded in `before/model/measurement.json`. The runner requires a fresh
output directory: move report artifacts aside before reproduction, then restore
this notebook builder. Model weights, prediction NPZ and prepared splits are local
and excluded from Git. `quality.json` records the overlap and vocabulary audit;
`summary.json` contains source hashes, all metrics and fit diagnostics;
`deltas.json` records absolute and relative changes.
