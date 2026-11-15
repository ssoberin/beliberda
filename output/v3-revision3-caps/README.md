# Revision3 feature caps and thresholds

[report.md](report.md) and [summary.json](summary.json) record fixed-threshold
46k/60k/100k comparisons plus a modest threshold choice on calibration only.
All runs use word ngrams(1,3), char_wb(3,5), the same 6498 train rows/order/seed.
46k reuses the revision A/B artifact. Only 60k and 100k are newly fitted here.

Run `python -m avito_compare.scripts.run_v3_revision_caps` from repo root after
reproducing `output/v3-revision3-study`. Output must be a fresh directory;
move saved reports aside before rerunning. Weights and intermediate split files
are not committed. The combined executed notebook is
`../v3-revision3-study/comparison.ipynb`.

Threshold candidates .5/.475/.45 are evaluated on public calibration728 plus
original v3 validation600, total1328. Highest calibration Micro-F1 wins; exact
ties prefer the higher threshold. Tests never select thresholds. Saved model
thresholds remain .5; alternative evaluation does not replace working weights.
V3 validation600 is calibration in this series and must not be presented as
an independent evaluation. Known organizer test conflicts/overlaps remain.
All tested thresholds are retained in JSON as diagnostics, not selections by test.
