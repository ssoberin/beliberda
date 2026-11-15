# Tail-service data-only C_syntax A/B

Run from the project root with its existing Python environment. Model A is the original C_syntax without hard-negative200. Model B uses exactly the same architecture/features/training protocol and baseline full-head SGD parameters, with 60 additional training ads. Models stay separate.

1. `python datasets/v3_tail_services_60/build.py`
2. `python output/tail-services-v3-experiment/audit_dataset.py`
3. `python output/tail-services-v3-experiment/prepare.py`
4. `python output/tail-services-v3-experiment/encode_cpu.py`
5. `python output/tail-services-v3-experiment/train_b.py`
6. After successful training validation: `python output/tail-services-v3-experiment/evaluate_once.py`
7. `python output/tail-services-v3-experiment/build_report.py`
8. `python output/tail-services-v3-experiment/verify.py`

Keep a fresh output directory for a new reproduction rather than overwriting the frozen result. `evaluate_once.py` refuses to overwrite metrics. No threshold selection, extra fitting, iterative test optimization, or changes to source labels are performed. Calibration is only used by the original CatBoost early stopping; nested OOF alpha selection is the unchanged train-only baseline routine.

Embeddings reuse the existing cache. Only 223 new chunks are encoded. Colab CLI could not open its log in the read-only config directory, including after escalation; no Colab VM was created. The existing supported CPU path was used instead: the same local frozen Qwen snapshot, float32 CPU inference, normalized output stored float16, max length1024. This differs in numerical execution from earlier CUDA float16 chunks and is a disclosed limitation; all original cached chunks remain unchanged.

Separate new test20 is small, author-labeled and never used for fitting/early stopping. Existing public/V3 tests were already inspected historically. Official208 has missing source_mcId, conflicting labels, and train intersections; source101 follows the existing convention. Both full and fixed-conflict-excluded152 results are reported with unchanged gold. `public_synthetic_300` is not used as data or scored. The historical public-test delta lookup may include other cached chunks; only needed input chunks are selected.

Initial inputs/config were committed as `0703f0e`; before any scoring, one ambiguous decorative-plaster example was replaced by explicit painting after re-reading the category guide. The first OOF pass was interrupted and discarded; corrected inputs were refrozen as `2ddc6c3` and training fully restarted. `quality_correction/` records provenance, not an additional scored candidate. Its previously encoded chunks are reused. For a clean reproduction without these optional intermediate cache files, prepare.py simply encodes all missing final chunks. Original datasets, models and caches are protected by hashes. New weights and large intermediate caches remain local; report, JSON metrics, predictions, annotations and reproduction scripts are versioned.
