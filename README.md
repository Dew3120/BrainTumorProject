# Brain-tumor MRI research pipeline

This repository contains a frozen research pipeline for BRISC-style brain MRI classification, tumor segmentation, pixel calibration, case-level reliability estimation, out-of-distribution review, and cost-sensitive display policy.

> **Research prototype — not for clinical use.** This is not a clinical tool or medical device, has not been validated for patient care, has not established generalizability beyond BRISC, and is not FDA approved or CE marked.

## Final stack

- **Classifier:** v1 ResNet-18 with raw four-class softmax probabilities
- **Segmenter:** E08 ensemble of E05, E06, and classifier-routed E07/v1
- **Pixel calibration:** C03 isotonic mapping
- **Case reliability:** R02 ten-feature L2 logistic regression
- **OOD detection:** O05 secondary logistic ensemble
- **Thresholds:** segmentation `0.50`; reliability `0.9490250173235835`; OOD review `0.36854275476100085`

The complete, hash-verified inference package is in [`runs/final_stack/`](runs/final_stack/). Do not edit that directory.

## Start here

- [Narrative report](docs/report.md)
- [Methods and formulas](docs/methods.md)
- [Consolidated limitations](docs/limitations.md)
- [Artifact and experiment appendix](docs/appendix.md)
- [Machine-readable metric ledger](docs/stage9_source_ledger.json)
- [Figures and figure manifest](docs/figures/)
- [Pre-registration](audit/preregistration.md)
- [Review log](audit/review_log.md)
- [Final freeze report](audit/stage8_final_freezing.md)

## Headline findings

- Classifier macro F1 was 0.9779 on BRISC internal development validation; patient independence is unverified.
- Historical v1 image-mean segmentation Dice was 0.7694. Selected E08 category Dice was 0.6324 glioma, 0.8822 meningioma, and 0.8183 pituitary.
- R02 failure-ranking AUROC was 0.9555 internally, but glioma was its weakest subgroup.
- Independent spatial transport failed: OpenNeuro Dice 0.1272 and UPENN-GBM Dice 0.2208. H8 was rejected for the evaluated independent cohorts.
- Reliability ranking transferred better than segmentation, with AUROC 0.8719 on OpenNeuro and 0.8316 on UPENN-GBM, while absolute reliability calibration did not transfer.
- O05 AUROC was 0.9957 on the registered retrospective OOD benchmark. The detector may rely on dataset-source and encoding shortcuts.
- The reliability policy cost was 205 at its registered threshold versus 406 at 0.50, but sensitivity analysis classified the selected threshold as fragile.

Every value above links to a source through [`docs/stage9_source_ledger.json`](docs/stage9_source_ledger.json).

## Reproduce documentation and verification

```powershell
uv run python docs/build_stage9_figures.py
uv run python docs/build_stage9_appendix.py
uv run python docs/verify_stage9_docs.py
uv run python scripts/verify_final_stack.py
uv run ruff check .
$env:PYTHONPATH=((Resolve-Path '.').Path + ';' + (Resolve-Path 'scripts').Path)
uv run pytest -q
```

The documentation builders read committed evidence and must not write into `runs/final_stack/`. The verifier checks that the frozen tree and manifest hash remain unchanged.

## Repository map

- `audit/` — preregistration, amendments, review log, audits, findings, and stage reports
- `experiments/` — one registered package per experiment
- `runs/` — training/evaluation artifacts and the immutable final stack
- `models/` — deployable reliability artifacts and model provenance
- `scripts/` — training, analysis, evaluation, policy, and verification code
- `tests/` — regression and integrity tests
- `docs/` — report, methods, limitations, appendix, literature review, and figures

## Binding scope

No BRISC sealed-test image was accessed during development or documentation. The external evaluations are retrospective. The system cannot provide diagnosis, causation, danger, prognosis, tumor depth, physical size without verified spacing, or treatment advice.
