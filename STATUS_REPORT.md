# BrainTumorProject — Complete Status Report

**Generated:** 2026-09-24T21:29:50+05:30
**Scope:** Repository state only. No experiment was started by this report.
**Clinical posture:** Research prototype using BRISC internal validation; no clinical validity, patient-independent generalization, or external generalization is established.

## Executive state

- Stages 0–2 are complete.
- C01 is complete and passed its registered calibration gate.
- C02 is complete and failed only its mandatory hard-mask identity checks; its result exists locally but is not committed.
- C03–C09, all E08-specific Stage 4 studies, and all Stage 5 policy studies are pending.
- Frozen v1 remains the accepted segmenter. E08 remains the locked experimental v2 candidate.
- The sealed test partition has not been accessed.

## 1. Files and artifacts

### Repository directory tree

Caches, `.venv`, `.git`, `__pycache__`, `.pytest_cache`, and `.ruff_cache` are excluded. Directory annotations show direct file count, recursive file count, and recursive size. Bulk image/checkpoint files are counted here; file-by-file descriptions are provided for `audit/`, `experiments/`, `docs/`, and `scripts/` below.

```text
E:/BrainTumorProject/
├── .gitattributes
├── .gitignore
├── .python-version
├── pyproject.toml
├── uv.lock
├── STATUS_REPORT.md (this report)
├── audit/ [50 direct; 99 recursive files; 17.66 MiB]
│   ├── archive/ [0 direct; 12 recursive files; 5.52 MiB]
│   │   ├── before-mask-fix-20260921-182626/ [7 direct; 7 recursive files; 1.58 MiB]
│   │   └── superseded-root-output/ [5 direct; 5 recursive files; 3.94 MiB]
│   ├── environment/ [0 direct; 1 recursive files; 948.00 B]
│   │   └── 20260921-213814/ [1 direct; 1 recursive files; 948.00 B]
│   ├── environment-before-20260921-185535/ [2 direct; 2 recursive files; 155.11 KiB]
│   ├── phase3_review/ [1 direct; 1 recursive files; 269.40 KiB]
│   ├── phase4_review/ [1 direct; 1 recursive files; 72.33 KiB]
│   ├── reviews/ [0 direct; 24 recursive files; 3.84 MiB]
│   │   └── 20260921-183932/ [24 direct; 24 recursive files; 3.84 MiB]
│   └── runs/ [0 direct; 8 recursive files; 4.30 MiB]
│       └── 20260921-182736-884368/ [8 direct; 8 recursive files; 4.30 MiB]
├── data/ [0 direct; 15593 recursive files; 266.26 MiB]
│   ├── original_downloads/ [0 direct; 0 recursive files; 0.00 B]
│   ├── raw/ [0 direct; 15592 recursive files; 263.77 MiB]
│   │   └── brisc/ [0 direct; 15592 recursive files; 263.77 MiB]
│   │       ├── .complete/ [0 direct; 1 recursive files; 0.00 B]
│   │       │   └── datasets/ [0 direct; 1 recursive files; 0.00 B]
│   │       │       └── briscdataset/ [0 direct; 1 recursive files; 0.00 B]
│   │       │           └── brisc2025/ [0 direct; 1 recursive files; 0.00 B]
│   │       │               └── 6/ [1 direct; 1 recursive files; 0.00 B]
│   │       └── brisc2025/ [5 direct; 15591 recursive files; 263.77 MiB]
│   │           ├── classification_task/ [0 direct; 6000 recursive files; 133.04 MiB]
│   │           │   ├── test/ [0 direct; 1000 recursive files; 20.87 MiB]
│   │           │   │   ├── glioma/ [254 direct; 254 recursive files; 4.89 MiB]
│   │           │   │   ├── meningioma/ [306 direct; 306 recursive files; 6.32 MiB]
│   │           │   │   ├── no_tumor/ [140 direct; 140 recursive files; 2.10 MiB]
│   │           │   │   └── pituitary/ [300 direct; 300 recursive files; 7.55 MiB]
│   │           │   └── train/ [0 direct; 5000 recursive files; 112.17 MiB]
│   │           │       ├── glioma/ [1147 direct; 1147 recursive files; 20.66 MiB]
│   │           │       ├── meningioma/ [1329 direct; 1329 recursive files; 28.67 MiB]
│   │           │       ├── no_tumor/ [1067 direct; 1067 recursive files; 21.76 MiB]
│   │           │       └── pituitary/ [1457 direct; 1457 recursive files; 41.08 MiB]
│   │           └── segmentation_task/ [0 direct; 9586 recursive files; 118.09 MiB]
│   │               ├── test/ [0 direct; 1720 recursive files; 20.31 MiB]
│   │               │   ├── images/ [860 direct; 860 recursive files; 18.77 MiB]
│   │               │   └── masks/ [860 direct; 860 recursive files; 1.53 MiB]
│   │               └── train/ [0 direct; 7866 recursive files; 97.78 MiB]
│   │                   ├── images/ [3933 direct; 3933 recursive files; 90.41 MiB]
│   │                   └── masks/ [3933 direct; 3933 recursive files; 7.37 MiB]
│   └── splits/ [1 direct; 1 recursive files; 2.49 MiB]
├── docs/ [1 direct; 1 recursive files; 12.64 KiB]
├── experiments/ [2 direct; 167 recursive files; 9.29 MiB]
│   ├── _template/ [4 direct; 5 recursive files; 1.02 KiB]
│   │   └── artifacts/ [1 direct; 1 recursive files; 1.00 B]
│   ├── application_integrity_validation/ [4 direct; 5 recursive files; 2.17 KiB]
│   │   └── artifacts/ [1 direct; 1 recursive files; 1.00 B]
│   ├── c01_vector_scaling/ [4 direct; 9 recursive files; 326.04 KiB]
│   │   └── artifacts/ [5 direct; 5 recursive files; 247.16 KiB]
│   ├── c02_dirichlet/ [4 direct; 8 recursive files; 227.77 KiB]
│   │   └── artifacts/ [4 direct; 4 recursive files; 177.74 KiB]
│   ├── e01_lcc_filter/ [4 direct; 8 recursive files; 252.20 KiB]
│   │   └── artifacts/ [4 direct; 4 recursive files; 248.02 KiB]
│   ├── e02_morphology/ [4 direct; 8 recursive files; 370.42 KiB]
│   │   └── artifacts/ [4 direct; 4 recursive files; 364.75 KiB]
│   ├── e03_tta/ [4 direct; 8 recursive files; 262.13 KiB]
│   │   └── artifacts/ [4 direct; 4 recursive files; 257.64 KiB]
│   ├── e04_multiscale/ [4 direct; 8 recursive files; 286.24 KiB]
│   │   └── artifacts/ [4 direct; 4 recursive files; 281.79 KiB]
│   ├── e05_glioma_weighted/ [4 direct; 24 recursive files; 1.96 MiB]
│   │   └── artifacts/ [20 direct; 20 recursive files; 1.95 MiB]
│   ├── e06_boundary_loss/ [4 direct; 24 recursive files; 1.96 MiB]
│   │   └── artifacts/ [20 direct; 20 recursive files; 1.96 MiB]
│   ├── e07_glioma_specialized/ [4 direct; 26 recursive files; 2.09 MiB]
│   │   └── artifacts/ [22 direct; 22 recursive files; 2.08 MiB]
│   ├── e08_ensemble/ [4 direct; 12 recursive files; 1.57 MiB]
│   │   └── artifacts/ [8 direct; 8 recursive files; 1.57 MiB]
│   ├── e08_final_selection/ [4 direct; 5 recursive files; 1.22 KiB]
│   │   └── artifacts/ [1 direct; 1 recursive files; 0.00 B]
│   ├── glioma_segmentation_improvement/ [4 direct; 5 recursive files; 2.71 KiB]
│   │   └── artifacts/ [1 direct; 1 recursive files; 1.00 B]
│   ├── measurement_engine_validation/ [4 direct; 5 recursive files; 1.81 KiB]
│   │   └── artifacts/ [1 direct; 1 recursive files; 1.00 B]
│   └── sealed_final_evaluation/ [4 direct; 5 recursive files; 3.37 KiB]
│       └── artifacts/ [1 direct; 1 recursive files; 1.00 B]
├── runs/ [1 direct; 4028 recursive files; 564.79 MiB]
│   ├── baseline_v1/ [2 direct; 61 recursive files; 57.43 MiB]
│   │   ├── classifier/ [18 direct; 18 recursive files; 46.50 MiB]
│   │   ├── environment/ [3 direct; 3 recursive files; 237.90 KiB]
│   │   ├── evidence/ [4 direct; 4 recursive files; 32.24 KiB]
│   │   └── segmenter/ [31 direct; 34 recursive files; 10.65 MiB]
│   │       └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   ├── classifier/ [2 direct; 36 recursive files; 137.49 MiB]
│   │   ├── 20260921-170534-443939-smoke/ [10 direct; 10 recursive files; 45.54 MiB]
│   │   ├── 20260921-170607-243133-baseline/ [18 direct; 18 recursive files; 46.50 MiB]
│   │   └── 20260921-172945-476206-baseline/ [6 direct; 6 recursive files; 45.46 MiB]
│   ├── mlflow_artifacts/ [0 direct; 258 recursive files; 52.14 MiB]
│   │   ├── classifier/ [0 direct; 22 recursive files; 1.18 MiB]
│   │   │   ├── 152095f77c6e4d1eaa64151a1a3e6a4c/ [0 direct; 6 recursive files; 81.68 KiB]
│   │   │   │   └── artifacts/ [6 direct; 6 recursive files; 81.68 KiB]
│   │   │   └── f8bb3cf4e8f7499f9a9c5f17b7d4d937/ [0 direct; 16 recursive files; 1.11 MiB]
│   │   │       └── artifacts/ [16 direct; 16 recursive files; 1.11 MiB]
│   │   ├── db8d9b4befc14b46a1eb946493cd4958/ [0 direct; 1 recursive files; 948.00 B]
│   │   │   └── artifacts/ [1 direct; 1 recursive files; 948.00 B]
│   │   ├── segmentation-full/ [0 direct; 60 recursive files; 16.26 MiB]
│   │   │   ├── 142af7eee98943adb2a0434b8012c268/ [0 direct; 56 recursive files; 11.28 MiB]
│   │   │   │   └── artifacts/ [53 direct; 56 recursive files; 11.28 MiB]
│   │   │   │       └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   │   │   ├── 3c5a1a97bf664cf2b2e49910892130a9/ [0 direct; 2 recursive files; 2.49 MiB]
│   │   │   │   └── artifacts/ [2 direct; 2 recursive files; 2.49 MiB]
│   │   │   └── 60f3ebbbb806472d8d8b5ca429ac4860/ [0 direct; 2 recursive files; 2.49 MiB]
│   │   │       └── artifacts/ [2 direct; 2 recursive files; 2.49 MiB]
│   │   ├── segmentation-glioma-specialist/ [0 direct; 32 recursive files; 10.06 MiB]
│   │   │   └── f3c33524283e4258b23c25a28382b326/ [0 direct; 32 recursive files; 10.06 MiB]
│   │   │       └── artifacts/ [29 direct; 32 recursive files; 10.06 MiB]
│   │   │           └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   │   ├── segmentation-overfit/ [0 direct; 52 recursive files; 3.25 MiB]
│   │   │   ├── 30ac0615363e4fc0811db6a31a0a1910/ [0 direct; 7 recursive files; 324.92 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 324.92 KiB]
│   │   │   ├── 5fe7f81842d74c2b877da374d7cd6e50/ [0 direct; 7 recursive files; 529.34 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 529.34 KiB]
│   │   │   ├── 9ab735a414af4fe9ae7ca68ad01c63ed/ [0 direct; 8 recursive files; 411.00 KiB]
│   │   │   │   └── artifacts/ [8 direct; 8 recursive files; 411.00 KiB]
│   │   │   ├── 9c41eeb86bf14ea59efa66838ce0d2a4/ [0 direct; 7 recursive files; 319.68 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 319.68 KiB]
│   │   │   ├── b41edb07a43c47bcadec674c22b0077b/ [0 direct; 8 recursive files; 451.32 KiB]
│   │   │   │   └── artifacts/ [8 direct; 8 recursive files; 451.32 KiB]
│   │   │   ├── f9683bf7b16e4b8c8250ce3ac3b0afc0/ [0 direct; 7 recursive files; 441.87 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 441.87 KiB]
│   │   │   └── fe164382767f4afd92f5197f7d6a8a34/ [0 direct; 8 recursive files; 848.31 KiB]
│   │   │       └── artifacts/ [8 direct; 8 recursive files; 848.31 KiB]
│   │   ├── segmentation-single-pair/ [0 direct; 21 recursive files; 482.72 KiB]
│   │   │   ├── 3e1e9ea510a44664aeb4d679fc486d4d/ [0 direct; 7 recursive files; 159.94 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 159.94 KiB]
│   │   │   ├── c7d9731cef3b4918b31766d79793902c/ [0 direct; 7 recursive files; 182.81 KiB]
│   │   │   │   └── artifacts/ [7 direct; 7 recursive files; 182.81 KiB]
│   │   │   └── e27809de820746cba0178f9e3dd34bf6/ [0 direct; 7 recursive files; 139.97 KiB]
│   │   │       └── artifacts/ [7 direct; 7 recursive files; 139.97 KiB]
│   │   ├── segmentation-smoke/ [0 direct; 6 recursive files; 510.60 KiB]
│   │   │   └── ff3ff108eb854c759deb31d5ce020be2/ [0 direct; 6 recursive files; 510.60 KiB]
│   │   │       └── artifacts/ [6 direct; 6 recursive files; 510.60 KiB]
│   │   ├── segmentation-v2-boundary-loss/ [0 direct; 32 recursive files; 10.19 MiB]
│   │   │   └── a92cfe0f0d2542f1bcc28593f9a91312/ [0 direct; 32 recursive files; 10.19 MiB]
│   │   │       └── artifacts/ [29 direct; 32 recursive files; 10.19 MiB]
│   │   │           └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   │   └── segmentation-v2-weighted/ [0 direct; 32 recursive files; 10.23 MiB]
│   │       └── fc5ac87288904a039968b751e256b728/ [0 direct; 32 recursive files; 10.23 MiB]
│   │           └── artifacts/ [29 direct; 32 recursive files; 10.23 MiB]
│   │               └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   ├── segmentation/ [4 direct; 170 recursive files; 27.07 MiB]
│   │   ├── 20260921-174907-092984-smoke/ [8 direct; 8 recursive files; 583.15 KiB]
│   │   ├── 20260921-175439-356139-overfit/ [11 direct; 11 recursive files; 615.82 KiB]
│   │   ├── 20260922-060600-323325-overfit/ [10 direct; 10 recursive files; 405.59 KiB]
│   │   ├── 20260922-060723-749607-overfit/ [10 direct; 10 recursive files; 410.53 KiB]
│   │   ├── 20260922-061931-668446-batch-inspection/ [1 direct; 1 recursive files; 2.12 KiB]
│   │   ├── 20260922-062033-862162-single-pair/ [10 direct; 10 recursive files; 224.79 KiB]
│   │   ├── 20260922-062534-350737-single-pair/ [10 direct; 10 recursive files; 246.85 KiB]
│   │   ├── 20260923-062340-683704-single-pair/ [10 direct; 10 recursive files; 269.21 KiB]
│   │   ├── 20260923-063527-150497-overfit/ [10 direct; 10 recursive files; 531.31 KiB]
│   │   ├── 20260923-064720-473343-overfit-256/ [11 direct; 11 recursive files; 941.50 KiB]
│   │   ├── 20260923-065751-622110-overfit-128/ [11 direct; 11 recursive files; 546.42 KiB]
│   │   ├── 20260923-070137-629306-overfit-128-compact/ [11 direct; 11 recursive files; 1.99 MiB]
│   │   ├── 20260923-071442-892340-full/ [7 direct; 7 recursive files; 2.55 MiB]
│   │   │   └── top3_checkpoints/ [0 direct; 0 recursive files; 0.00 B]
│   │   ├── 20260923-071703-220400-full/ [9 direct; 12 recursive files; 7.21 MiB]
│   │   │   └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   │   └── 20260923-072439-397263-full/ [31 direct; 34 recursive files; 10.65 MiB]
│   │       └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   ├── segmentation_v2_boundary_loss/ [1 direct; 3431 recursive files; 224.04 MiB]
│   │   └── 20260924-063051-647842-boundary/ [28 direct; 3430 recursive files; 224.04 MiB]
│   │       ├── distance_cache/ [0 direct; 3399 recursive files; 212.85 MiB]
│   │       │   ├── train/ [2719 direct; 2719 recursive files; 170.27 MiB]
│   │       │   └── validation/ [680 direct; 680 recursive files; 42.58 MiB]
│   │       └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   ├── segmentation_v2_ensemble/ [1 direct; 5 recursive files; 39.38 MiB]
│   │   └── 20260924-122739-908245-calibration-cache/ [4 direct; 4 recursive files; 39.38 MiB]
│   ├── segmentation_v2_glioma_specialized/ [1 direct; 35 recursive files; 10.32 MiB]
│   │   └── 20260924-105229-241365-specialist/ [31 direct; 34 recursive files; 10.32 MiB]
│   │       └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
│   └── segmentation_v2_weighted/ [1 direct; 31 recursive files; 10.21 MiB]
│       └── 20260924-055049-892543-weighted/ [27 direct; 30 recursive files; 10.21 MiB]
│           └── top3_checkpoints/ [3 direct; 3 recursive files; 4.65 MiB]
├── scripts/ [55 direct; 56 recursive files; 780.77 KiB]
│   └── snapshots/ [1 direct; 1 recursive files; 32.12 KiB]
└── tests/ [32 direct; 32 recursive files; 59.36 KiB]
```

### Top-level folder summary

| Folder | Files | Total size | Description |
|---|---:|---:|---|
| `audit/` | 99 | 17.66 MiB | Governance, dataset review, findings, figures, and consolidated evidence. |
| `data/` | 15,593 | 266.26 MiB | BRISC working images, masks, manifests, and split snapshots. |
| `docs/` | 1 | 12.64 KiB | Research literature review. |
| `experiments/` | 167 | 9.29 MiB | Preregistered experiment packages and master registry. |
| `runs/` | 4,028 | 564.79 MiB | Training, inference, MLflow, checkpoint, prediction, and frozen-baseline artifacts. |
| `scripts/` | 56 | 780.77 KiB | Training, evaluation, calibration, analysis, and verification programs. |
| `tests/` | 32 | 59.36 KiB | Automated unit, integration, governance, and artifact-verification tests. |

### `audit/` file inventory

- `audit/archive/before-mask-fix-20260921-182626/audit_brisc.py` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/before-mask-fix-20260921-182626/dataset_source.json` — Dataset source, version, license, and provenance.
- `audit/archive/before-mask-fix-20260921-182626/inventory.csv` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/before-mask-fix-20260921-182626/issues.csv` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/before-mask-fix-20260921-182626/preview_no_tumor.png` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/before-mask-fix-20260921-182626/similarity_candidates.csv` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/before-mask-fix-20260921-182626/summary.json` — Archived pre-mask-fix audit artifact retained for provenance.
- `audit/archive/superseded-root-output/inventory.csv` — Archived superseded root audit output retained for provenance.
- `audit/archive/superseded-root-output/issues.csv` — Archived superseded root audit output retained for provenance.
- `audit/archive/superseded-root-output/preview_no_tumor.png` — Archived superseded root audit output retained for provenance.
- `audit/archive/superseded-root-output/similarity_candidates.csv` — Archived superseded root audit output retained for provenance.
- `audit/archive/superseded-root-output/summary.json` — Archived superseded root audit output retained for provenance.
- `audit/classifier_binary_calibration.json` — Binary classifier calibration metrics and gate evidence.
- `audit/classifier_binary_calibration.md` — Binary classifier calibration conclusion; raw output retained.
- `audit/classifier_calibration.json` — Four-class classifier calibration metrics.
- `audit/classifier_calibration.md` — Four-class classifier calibration conclusion.
- `audit/classifier_segmenter_agreement.json` — Machine-readable classifier/segmenter agreement analysis.
- `audit/classifier_segmenter_agreement.md` — Finding that correct classification does not imply correct segmentation.
- `audit/classifier_training_findings.json` — Frozen classifier metrics, shortcut diagnostic, and limitations.
- `audit/confidence_calibration_completion.md` — Earlier v1 confidence-calibration completion report.
- `audit/confidence_calibration_findings.json` — Consolidated v1 classifier, pixel, and case-reliability calibration findings.
- `audit/dataset_source.json` — Dataset source, version, license, and provenance.
- `audit/environment-before-20260921-185535/pyproject.toml` — Dependency snapshot from before environment setup.
- `audit/environment-before-20260921-185535/uv.lock` — Dependency snapshot from before environment setup.
- `audit/environment/20260921-213814/environment_check.json` — Captured Python/CUDA/GPU/dependency environment verification.
- `audit/foundation_reset_verification.json` — Verification of preregistration, baseline freeze, tracking, and review controls.
- `audit/glioma_boundary_sharpness.csv` — Boundary-gradient measurements for all 204 glioma validation cases.
- `audit/glioma_case_study.md` — Narrative for ten area-matched glioma success/failure pairs.
- `audit/glioma_case_study_pairs.png` — Rendered ten-pair glioma comparison panel.
- `audit/glioma_failure_analysis_verification.json` — Integrity checks for glioma-analysis outputs.
- `audit/glioma_failure_artifact_manifest.json` — SHA256 manifest for glioma-analysis artifacts.
- `audit/glioma_failure_decomposition.csv` — Per-case geometry/topology/boundary audit for 60 weak glioma cases.
- `audit/glioma_failure_findings.md` — Consolidated evidenced glioma failure findings.
- `audit/glioma_failure_review_01.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_failure_review_02.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_failure_review_03.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_failure_review_04.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_failure_review_05.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_failure_review_06.png` — Manual visual-review panel for weak glioma cases.
- `audit/glioma_subgroup_analysis.md` — Bootstrap subgroup analysis; small lesions show clearest elevated risk.
- `audit/glioma_visual_review_annotations.csv` — Human visual annotations with unassessable clinical fields explicitly marked.
- `audit/phase3_review/saved_model_binary_errors.png` — Rendered saved-classifier error panel.
- `audit/phase4_review/overfit_comparison.png` — Rendered tiny-set/full-run overfit comparison.
- `audit/preregistration.md` — Primary hypotheses H1–H6, gates, metrics, data, and analysis rules.
- `audit/preregistration_amendment_01.md` — Controlled glioma, calibration, reliability, and policy program.
- `audit/preregistration_amendment_02.md` — Local H7 conditional E08 selection amendment; ignored and untracked by Git.
- `audit/review_log.md` — Chronological hash-linked governance decisions.
- `audit/reviews/20260921-183932/exclusion_reasons.csv` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/group_1013_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1078_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1079_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_109_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1151_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1169_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_16_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1901_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_1903_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_2066_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_2088_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_2525_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_2835_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_3350_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_3398_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_585_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_67_train.png` — Manual similarity-group review panel.
- `audit/reviews/20260921-183932/group_review.csv` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/input_quality_flags.csv` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/integrity_issues.csv` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/mask_threshold_review.csv` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/review_decisions.json` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/reviews/20260921-183932/review_summary.json` — Human dataset-review decision, flag, summary, or integrity artifact.
- `audit/runs/20260921-182736-884368/inventory.csv` — Canonical audited inventory with split/hash/label/similarity metadata.
- `audit/runs/20260921-182736-884368/issues.csv` — Dataset technical issue list.
- `audit/runs/20260921-182736-884368/preview_glioma.png` — Class-specific image/mask alignment preview.
- `audit/runs/20260921-182736-884368/preview_meningioma.png` — Class-specific image/mask alignment preview.
- `audit/runs/20260921-182736-884368/preview_no_tumor.png` — Class-specific image/mask alignment preview.
- `audit/runs/20260921-182736-884368/preview_pituitary.png` — Class-specific image/mask alignment preview.
- `audit/runs/20260921-182736-884368/similarity_candidates.csv` — Candidate duplicate/similarity pairs.
- `audit/runs/20260921-182736-884368/summary.json` — Dataset-audit counts and constraints.
- `audit/segmentation_capacity_gate_findings.json` — Laptop/GPU segmentation-capacity result.
- `audit/segmentation_failure_modes.md` — Topology breakdown of 83 v1 cases below Dice 0.50.
- `audit/segmentation_overfit_gate_findings.json` — Tiny-set memorization-gate evidence.
- `audit/segmentation_reproducibility.md` — Determinism, environment, manifest, and reload findings.
- `audit/segmentation_resolution_gate_findings.json` — 128-versus-256 resolution diagnostic.
- `audit/segmentation_single_pair_findings.json` — One-pair tensor/loss/gradient diagnostic.
- `audit/segmentation_six_pair_gate_findings.json` — Six-pair overfit-gate findings.
- `audit/segmentation_smoke_findings.json` — Initial segmentation pipeline smoke results.
- `audit/segmentation_training_findings.json` — Consolidated v1 training, metrics, failure, uncertainty, and reproducibility evidence.
- `audit/segmentation_uncertainty_signals.json` — Machine-readable extreme-case uncertainty analysis.
- `audit/segmentation_uncertainty_signals.md` — Finding that zero-Dice errors are confident spatial failures.
- `audit/segmentation_visual_review_guide.md` — Manual overlay-review instructions and stopping rules.
- `audit/segmentation_weighted_single_pair_findings.json` — Weighted-loss single-pair diagnostic.
- `audit/segmenter_case_reliability.json` — Full v1 OOF reliability model, features, coefficients, metrics, and verification.
- `audit/segmenter_case_reliability.md` — Readable v1 reliability report.
- `audit/segmenter_pixel_calibration.json` — Full v1 pixel-temperature calibration result.
- `audit/segmenter_pixel_calibration.md` — Readable v1 pixel-calibration decision.
- `audit/technical_debt.md` — Resolved and deferred engineering cleanup ledger.
- `audit/v2_model_selection.json` — Machine-readable E01–E08 selection evidence.
- `audit/v2_model_selection.md` — E08 experimental selection and v1 retention decision.

### `experiments/` file inventory

- `experiments/_template/artifacts/.gitkeep` — Reusable experiment-package template file.
- `experiments/_template/config.json` — Reusable experiment-package template file.
- `experiments/_template/decision.md` — Reusable experiment-package template file.
- `experiments/_template/hypothesis.md` — Reusable experiment-package template file.
- `experiments/_template/result.json` — Reusable experiment-package template file.
- `experiments/application_integrity_validation/artifacts/.gitkeep` — Placeholder preserving an empty artifacts directory.
- `experiments/application_integrity_validation/config.json` — application_integrity_validation parameters, hashes, gates, and lifecycle status.
- `experiments/application_integrity_validation/decision.md` — application_integrity_validation decision and rationale.
- `experiments/application_integrity_validation/hypothesis.md` — application_integrity_validation falsifiable hypothesis and protocol.
- `experiments/application_integrity_validation/result.json` — application_integrity_validation metrics, checks, constraints, and result state.
- `experiments/c01_vector_scaling/artifacts/classifier_reliability.png` — C01 classifier reliability diagram.
- `experiments/c01_vector_scaling/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/c01_vector_scaling/artifacts/segmenter_fold_parameters.csv` — C01 per-fold coefficients.
- `experiments/c01_vector_scaling/artifacts/segmenter_lesion_centered_reliability.png` — C01 lesion-centred reliability diagram.
- `experiments/c01_vector_scaling/artifacts/segmenter_whole_image_reliability.png` — C01 whole-image reliability diagram.
- `experiments/c01_vector_scaling/config.json` — c01_vector_scaling parameters, hashes, gates, and lifecycle status.
- `experiments/c01_vector_scaling/decision.md` — c01_vector_scaling decision and rationale.
- `experiments/c01_vector_scaling/hypothesis.md` — c01_vector_scaling falsifiable hypothesis and protocol.
- `experiments/c01_vector_scaling/result.json` — c01_vector_scaling metrics, checks, constraints, and result state.
- `experiments/c02_dirichlet/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/c02_dirichlet/artifacts/fold_parameters.csv` — C02 per-fold Dirichlet coefficients.
- `experiments/c02_dirichlet/artifacts/lesion_centered_reliability.png` — C02 lesion-centred reliability diagram.
- `experiments/c02_dirichlet/artifacts/whole_image_reliability.png` — C02 whole-image reliability diagram.
- `experiments/c02_dirichlet/config.json` — c02_dirichlet parameters, hashes, gates, and lifecycle status.
- `experiments/c02_dirichlet/decision.md` — c02_dirichlet decision and rationale.
- `experiments/c02_dirichlet/hypothesis.md` — c02_dirichlet falsifiable hypothesis and protocol.
- `experiments/c02_dirichlet/result.json` — c02_dirichlet metrics, checks, constraints, and result state.
- `experiments/e01_lcc_filter/artifacts/dice_delta_distribution.png` — Paired Dice-change distribution.
- `experiments/e01_lcc_filter/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e01_lcc_filter/artifacts/per_case_metrics.csv` — Per-case baseline and candidate metrics.
- `experiments/e01_lcc_filter/artifacts/summary.md` — Readable experiment summary.
- `experiments/e01_lcc_filter/config.json` — e01_lcc_filter parameters, hashes, gates, and lifecycle status.
- `experiments/e01_lcc_filter/decision.md` — e01_lcc_filter decision and rationale.
- `experiments/e01_lcc_filter/hypothesis.md` — e01_lcc_filter falsifiable hypothesis and protocol.
- `experiments/e01_lcc_filter/result.json` — e01_lcc_filter metrics, checks, constraints, and result state.
- `experiments/e02_morphology/artifacts/dice_delta_distribution.png` — Paired Dice-change distribution.
- `experiments/e02_morphology/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e02_morphology/artifacts/per_case_metrics.csv` — Per-case baseline and candidate metrics.
- `experiments/e02_morphology/artifacts/summary.md` — Readable experiment summary.
- `experiments/e02_morphology/config.json` — e02_morphology parameters, hashes, gates, and lifecycle status.
- `experiments/e02_morphology/decision.md` — e02_morphology decision and rationale.
- `experiments/e02_morphology/hypothesis.md` — e02_morphology falsifiable hypothesis and protocol.
- `experiments/e02_morphology/result.json` — e02_morphology metrics, checks, constraints, and result state.
- `experiments/e03_tta/artifacts/dice_delta_distribution.png` — Paired Dice-change distribution.
- `experiments/e03_tta/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e03_tta/artifacts/per_case_metrics.csv` — Per-case baseline and candidate metrics.
- `experiments/e03_tta/artifacts/summary.md` — Readable experiment summary.
- `experiments/e03_tta/config.json` — e03_tta parameters, hashes, gates, and lifecycle status.
- `experiments/e03_tta/decision.md` — e03_tta decision and rationale.
- `experiments/e03_tta/hypothesis.md` — e03_tta falsifiable hypothesis and protocol.
- `experiments/e03_tta/result.json` — e03_tta metrics, checks, constraints, and result state.
- `experiments/e04_multiscale/artifacts/dice_delta_distribution.png` — Paired Dice-change distribution.
- `experiments/e04_multiscale/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e04_multiscale/artifacts/per_case_metrics.csv` — Per-case baseline and candidate metrics.
- `experiments/e04_multiscale/artifacts/summary.md` — Readable experiment summary.
- `experiments/e04_multiscale/config.json` — e04_multiscale parameters, hashes, gates, and lifecycle status.
- `experiments/e04_multiscale/decision.md` — e04_multiscale decision and rationale.
- `experiments/e04_multiscale/hypothesis.md` — e04_multiscale falsifiable hypothesis and protocol.
- `experiments/e04_multiscale/result.json` — e04_multiscale metrics, checks, constraints, and result state.
- `experiments/e05_glioma_weighted/artifacts/best_validation_by_category.png` — Best-case overlay panel by category.
- `experiments/e05_glioma_weighted/artifacts/category_dice_comparison.png` — Category baseline-versus-candidate Dice figure.
- `experiments/e05_glioma_weighted/artifacts/confidence_calibration_findings.json` — Consolidated candidate confidence findings.
- `experiments/e05_glioma_weighted/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e05_glioma_weighted/artifacts/per_case_comparison.csv` — Per-case comparison with frozen baseline.
- `experiments/e05_glioma_weighted/artifacts/run_manifest.json` — Copied training-run manifest.
- `experiments/e05_glioma_weighted/artifacts/segmentation_failure_modes.csv` — Per-case weak-segmentation failure flags.
- `experiments/e05_glioma_weighted/artifacts/segmentation_failure_modes.md` — Readable failure-mode summary.
- `experiments/e05_glioma_weighted/artifacts/segmenter_case_reliability.json` — Candidate case-reliability model and metrics.
- `experiments/e05_glioma_weighted/artifacts/segmenter_case_reliability.md` — Candidate reliability report.
- `experiments/e05_glioma_weighted/artifacts/segmenter_case_reliability_diagram.png` — Case-reliability diagram.
- `experiments/e05_glioma_weighted/artifacts/segmenter_foreground_reliability_diagram.png` — Foreground pixel-reliability diagram.
- `experiments/e05_glioma_weighted/artifacts/segmenter_pixel_calibration.json` — Candidate pixel-calibration metrics.
- `experiments/e05_glioma_weighted/artifacts/segmenter_pixel_calibration.md` — Candidate pixel-calibration decision.
- `experiments/e05_glioma_weighted/artifacts/segmenter_whole_image_reliability_diagram.png` — Whole-image pixel-reliability diagram.
- `experiments/e05_glioma_weighted/artifacts/selected_checkpoint.json` — Selected checkpoint identity and hash.
- `experiments/e05_glioma_weighted/artifacts/summary.md` — Readable experiment summary.
- `experiments/e05_glioma_weighted/artifacts/visual_audit_selections.csv` — Deterministic visual-review selections and duplicate flags.
- `experiments/e05_glioma_weighted/artifacts/visual_review.json` — Machine-readable visual-review conclusion.
- `experiments/e05_glioma_weighted/artifacts/worst_validation_by_category.png` — Worst-case overlay panel by category.
- `experiments/e05_glioma_weighted/config.json` — e05_glioma_weighted parameters, hashes, gates, and lifecycle status.
- `experiments/e05_glioma_weighted/decision.md` — e05_glioma_weighted decision and rationale.
- `experiments/e05_glioma_weighted/hypothesis.md` — e05_glioma_weighted falsifiable hypothesis and protocol.
- `experiments/e05_glioma_weighted/result.json` — e05_glioma_weighted metrics, checks, constraints, and result state.
- `experiments/e06_boundary_loss/artifacts/best_validation_by_category.png` — Best-case overlay panel by category.
- `experiments/e06_boundary_loss/artifacts/confidence_calibration_findings.json` — Consolidated candidate confidence findings.
- `experiments/e06_boundary_loss/artifacts/dice_and_boundary_f1_comparison.png` — Glioma Dice and boundary-F1 comparison.
- `experiments/e06_boundary_loss/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e06_boundary_loss/artifacts/per_case_comparison.csv` — Per-case comparison with frozen baseline.
- `experiments/e06_boundary_loss/artifacts/run_manifest.json` — Copied training-run manifest.
- `experiments/e06_boundary_loss/artifacts/segmentation_failure_modes.csv` — Per-case weak-segmentation failure flags.
- `experiments/e06_boundary_loss/artifacts/segmentation_failure_modes.md` — Readable failure-mode summary.
- `experiments/e06_boundary_loss/artifacts/segmenter_case_reliability.json` — Candidate case-reliability model and metrics.
- `experiments/e06_boundary_loss/artifacts/segmenter_case_reliability.md` — Candidate reliability report.
- `experiments/e06_boundary_loss/artifacts/segmenter_case_reliability_diagram.png` — Case-reliability diagram.
- `experiments/e06_boundary_loss/artifacts/segmenter_foreground_reliability_diagram.png` — Foreground pixel-reliability diagram.
- `experiments/e06_boundary_loss/artifacts/segmenter_pixel_calibration.json` — Candidate pixel-calibration metrics.
- `experiments/e06_boundary_loss/artifacts/segmenter_pixel_calibration.md` — Candidate pixel-calibration decision.
- `experiments/e06_boundary_loss/artifacts/segmenter_whole_image_reliability_diagram.png` — Whole-image pixel-reliability diagram.
- `experiments/e06_boundary_loss/artifacts/selected_checkpoint.json` — Selected checkpoint identity and hash.
- `experiments/e06_boundary_loss/artifacts/summary.md` — Readable experiment summary.
- `experiments/e06_boundary_loss/artifacts/visual_audit_selections.csv` — Deterministic visual-review selections and duplicate flags.
- `experiments/e06_boundary_loss/artifacts/visual_review.json` — Machine-readable visual-review conclusion.
- `experiments/e06_boundary_loss/artifacts/worst_validation_by_category.png` — Worst-case overlay panel by category.
- `experiments/e06_boundary_loss/config.json` — e06_boundary_loss parameters, hashes, gates, and lifecycle status.
- `experiments/e06_boundary_loss/decision.md` — e06_boundary_loss decision and rationale.
- `experiments/e06_boundary_loss/hypothesis.md` — e06_boundary_loss falsifiable hypothesis and protocol.
- `experiments/e06_boundary_loss/result.json` — e06_boundary_loss metrics, checks, constraints, and result state.
- `experiments/e07_glioma_specialized/artifacts/best_validation_by_category.png` — Best-case overlay panel by category.
- `experiments/e07_glioma_specialized/artifacts/confidence_calibration_findings.json` — Consolidated candidate confidence findings.
- `experiments/e07_glioma_specialized/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e07_glioma_specialized/artifacts/per_case_comparison.csv` — Per-case comparison with frozen baseline.
- `experiments/e07_glioma_specialized/artifacts/routed_dice_comparison.png` — Routed-system Dice comparison.
- `experiments/e07_glioma_specialized/artifacts/routing.json` — Classifier routing summary.
- `experiments/e07_glioma_specialized/artifacts/routing_assignments.csv` — Per-case routing assignments.
- `experiments/e07_glioma_specialized/artifacts/run_manifest.json` — Copied training-run manifest.
- `experiments/e07_glioma_specialized/artifacts/segmentation_failure_modes.csv` — Per-case weak-segmentation failure flags.
- `experiments/e07_glioma_specialized/artifacts/segmentation_failure_modes.md` — Readable failure-mode summary.
- `experiments/e07_glioma_specialized/artifacts/segmenter_case_reliability.json` — Candidate case-reliability model and metrics.
- `experiments/e07_glioma_specialized/artifacts/segmenter_case_reliability.md` — Candidate reliability report.
- `experiments/e07_glioma_specialized/artifacts/segmenter_case_reliability_diagram.png` — Case-reliability diagram.
- `experiments/e07_glioma_specialized/artifacts/segmenter_foreground_reliability_diagram.png` — Foreground pixel-reliability diagram.
- `experiments/e07_glioma_specialized/artifacts/segmenter_pixel_calibration.json` — Candidate pixel-calibration metrics.
- `experiments/e07_glioma_specialized/artifacts/segmenter_pixel_calibration.md` — Candidate pixel-calibration decision.
- `experiments/e07_glioma_specialized/artifacts/segmenter_whole_image_reliability_diagram.png` — Whole-image pixel-reliability diagram.
- `experiments/e07_glioma_specialized/artifacts/selected_checkpoint.json` — Selected checkpoint identity and hash.
- `experiments/e07_glioma_specialized/artifacts/summary.md` — Readable experiment summary.
- `experiments/e07_glioma_specialized/artifacts/visual_audit_selections.csv` — Deterministic visual-review selections and duplicate flags.
- `experiments/e07_glioma_specialized/artifacts/visual_review.json` — Machine-readable visual-review conclusion.
- `experiments/e07_glioma_specialized/artifacts/worst_validation_by_category.png` — Worst-case overlay panel by category.
- `experiments/e07_glioma_specialized/config.json` — e07_glioma_specialized parameters, hashes, gates, and lifecycle status.
- `experiments/e07_glioma_specialized/decision.md` — e07_glioma_specialized decision and rationale.
- `experiments/e07_glioma_specialized/hypothesis.md` — e07_glioma_specialized falsifiable hypothesis and protocol.
- `experiments/e07_glioma_specialized/result.json` — e07_glioma_specialized metrics, checks, constraints, and result state.
- `experiments/e08_ensemble/artifacts/best_validation_by_category.png` — Best-case overlay panel by category.
- `experiments/e08_ensemble/artifacts/dice_comparison.png` — Category Dice comparison.
- `experiments/e08_ensemble/artifacts/experiment_manifest.json` — SHA256/size manifest for the completed package.
- `experiments/e08_ensemble/artifacts/per_case_comparison.csv` — Per-case comparison with frozen baseline.
- `experiments/e08_ensemble/artifacts/summary.md` — Readable experiment summary.
- `experiments/e08_ensemble/artifacts/visual_audit_selections.csv` — Deterministic visual-review selections and duplicate flags.
- `experiments/e08_ensemble/artifacts/visual_review.json` — Machine-readable visual-review conclusion.
- `experiments/e08_ensemble/artifacts/worst_validation_by_category.png` — Worst-case overlay panel by category.
- `experiments/e08_ensemble/config.json` — e08_ensemble parameters, hashes, gates, and lifecycle status.
- `experiments/e08_ensemble/decision.md` — e08_ensemble decision and rationale.
- `experiments/e08_ensemble/hypothesis.md` — e08_ensemble falsifiable hypothesis and protocol.
- `experiments/e08_ensemble/result.json` — e08_ensemble metrics, checks, constraints, and result state.
- `experiments/e08_final_selection/artifacts/.gitkeep` — Placeholder preserving an empty artifacts directory.
- `experiments/e08_final_selection/config.json` — e08_final_selection parameters, hashes, gates, and lifecycle status.
- `experiments/e08_final_selection/decision.md` — e08_final_selection decision and rationale.
- `experiments/e08_final_selection/hypothesis.md` — e08_final_selection falsifiable hypothesis and protocol.
- `experiments/e08_final_selection/result.json` — e08_final_selection metrics, checks, constraints, and result state.
- `experiments/glioma_segmentation_improvement/artifacts/.gitkeep` — Placeholder preserving an empty artifacts directory.
- `experiments/glioma_segmentation_improvement/config.json` — glioma_segmentation_improvement parameters, hashes, gates, and lifecycle status.
- `experiments/glioma_segmentation_improvement/decision.md` — glioma_segmentation_improvement decision and rationale.
- `experiments/glioma_segmentation_improvement/hypothesis.md` — glioma_segmentation_improvement falsifiable hypothesis and protocol.
- `experiments/glioma_segmentation_improvement/result.json` — glioma_segmentation_improvement metrics, checks, constraints, and result state.
- `experiments/measurement_engine_validation/artifacts/.gitkeep` — Placeholder preserving an empty artifacts directory.
- `experiments/measurement_engine_validation/config.json` — measurement_engine_validation parameters, hashes, gates, and lifecycle status.
- `experiments/measurement_engine_validation/decision.md` — measurement_engine_validation decision and rationale.
- `experiments/measurement_engine_validation/hypothesis.md` — measurement_engine_validation falsifiable hypothesis and protocol.
- `experiments/measurement_engine_validation/result.json` — measurement_engine_validation metrics, checks, constraints, and result state.
- `experiments/README.md` — Experiment lifecycle and directory contract.
- `experiments/registry.json` — Master registry with preregistration and baseline hashes.
- `experiments/sealed_final_evaluation/artifacts/.gitkeep` — Placeholder preserving an empty artifacts directory.
- `experiments/sealed_final_evaluation/config.json` — sealed_final_evaluation parameters, hashes, gates, and lifecycle status.
- `experiments/sealed_final_evaluation/decision.md` — sealed_final_evaluation decision and rationale.
- `experiments/sealed_final_evaluation/hypothesis.md` — sealed_final_evaluation falsifiable hypothesis and protocol.
- `experiments/sealed_final_evaluation/result.json` — sealed_final_evaluation metrics, checks, constraints, and result state.

### `docs/` file inventory

- `docs/glioma_literature_review.md` — Review of 19 primary papers on glioma segmentation, BRISC, boundary losses, diffuse tumors, reported Dice, and limitations.

### `scripts/` file inventory

- `scripts/advanced_calibration_common.py` — Shared cache, fold, region, metric, bootstrap, and plotting utilities for C01 onward.
- `scripts/analyze_glioma_failures.py` — Builds glioma geometry, boundary, case-study, and subgroup analyses.
- `scripts/audit_brisc.py` — Inventories BRISC, validates files/masks, and finds similarity candidates.
- `scripts/calibrate_classifier_binary_confidence.py` — Evaluates binary tumor/no-tumor temperature scaling.
- `scripts/calibrate_classifier_confidence.py` — Evaluates four-class classifier temperature scaling.
- `scripts/calibrate_segmenter_pixels.py` — Evaluates v1 pixel-level segmenter calibration.
- `scripts/check_geometry_shortcut.py` — Tests label association using dimensions without image pixels.
- `scripts/classifier_segmenter_agreement.py` — Computes classifier/segmenter agreement and special flags.
- `scripts/consolidate_confidence_calibration.py` — Combines confidence and reliability findings.
- `scripts/consolidate_segmentation_findings.py` — Combines segmentation metrics, failures, uncertainty, and reproducibility.
- `scripts/evaluate_e07_routed_system.py` — Evaluates classifier-routed glioma-specialist inference.
- `scripts/evaluate_e08_ensemble.py` — Evaluates the fixed E05/E06/E07 probability ensemble.
- `scripts/evaluate_lcc_filter.py` — Evaluates largest-component post-processing.
- `scripts/evaluate_morphology.py` — Evaluates registered closing/opening variants.
- `scripts/evaluate_multiscale.py` — Evaluates fixed 128/256 inference.
- `scripts/evaluate_tta.py` — Evaluates fixed eight-view augmentation.
- `scripts/finalize_e05_weighted.py` — Packages E05 training and downstream audits.
- `scripts/finalize_e06_boundary_loss.py` — Packages E06 training and downstream audits.
- `scripts/finalize_e07_glioma_specialist.py` — Packages E07 training/routing evidence.
- `scripts/finalize_e08_ensemble.py` — Packages E08 metrics, visual review, and decision.
- `scripts/fit_segmenter_case_reliability.py` — Fits the v1 prediction-derived OOF reliability model.
- `scripts/freeze_baseline_v1.py` — Copies and hashes immutable v1 artifacts.
- `scripts/materialize_e08_calibration_cache.py` — Creates verified E08 probabilities/targets for calibration.
- `scripts/review_brisc.py` — Supports manual dataset, duplicate, quality, and mask review.
- `scripts/run_c01_vector_scaling.py` — Runs five-fold C01 vector scaling.
- `scripts/run_c02_dirichlet.py` — Runs five-fold C02 Dirichlet calibration and mask checks.
- `scripts/run_e05_calibration_followup.py` — Runs E05 calibration/reliability follow-up.
- `scripts/run_e06_calibration_followup.py` — Runs E06 calibration/reliability follow-up.
- `scripts/run_e07_calibration_followup.py` — Runs E07 calibration/reliability follow-up.
- `scripts/segmentation_failure_analysis.py` — Classifies weak segmentation topology and overlap modes.
- `scripts/segmentation_overfit_gate.py` — Runs controlled tiny-set memorization gate.
- `scripts/segmentation_reproducibility_audit.py` — Audits determinism, hashes, reload fidelity, and repeatability.
- `scripts/segmentation_single_pair_diagnostic.py` — Diagnoses one-pair tensors, losses, and gradients.
- `scripts/segmentation_smoke_test.py` — Runs minimal segmentation pipeline smoke test.
- `scripts/segmentation_system.py` — Shared dataset, model, loss, metric, and inference utilities.
- `scripts/segmentation_uncertainty_analysis.py` — Extracts classifier/segmenter signals for extreme cases.
- `scripts/segmentation_visual_audit.py` — Recomputes predictions and renders overlay panels.
- `scripts/select_v2_model.py` — Applies E01–E08 model-selection rules.
- `scripts/snapshots/fit_segmenter_case_reliability_v1.py` — Frozen snapshot of accepted v1 reliability fitting.
- `scripts/train_classifier.py` — Trains/evaluates ImageNet-initialized four-class ResNet-18.
- `scripts/train_segmenter.py` — Trains compact MONAI 2D U-Net baseline.
- `scripts/train_segmenter_boundary_loss.py` — Trains BCE/Dice/boundary-loss segmenter.
- `scripts/train_segmenter_glioma_specialist.py` — Trains glioma-only specialist.
- `scripts/train_segmenter_glioma_weighted.py` — Trains with glioma loss multiplied by three.
- `scripts/validate_experiment_tracking.py` — Validates registry, files, lifecycle, and frozen hashes.
- `scripts/verify_baseline_v1.py` — Verifies frozen v1 artifact hashes.
- `scripts/verify_e01_lcc_filter.py` — Verifies E01 artifacts and metrics.
- `scripts/verify_e02_morphology.py` — Verifies E02 artifacts and metrics.
- `scripts/verify_e03_tta.py` — Verifies E03 transforms, artifacts, and metrics.
- `scripts/verify_e04_multiscale.py` — Verifies E04 scale handling and metrics.
- `scripts/verify_e05_weighted.py` — Verifies E05 checkpoint and audit package.
- `scripts/verify_e06_boundary_loss.py` — Verifies E06 checkpoint and boundary metrics.
- `scripts/verify_e07_glioma_specialist.py` — Verifies E07 checkpoint and routing.
- `scripts/verify_e08_ensemble.py` — Verifies component reproduction and ensemble result.
- `scripts/verify_glioma_failure_analysis.py` — Verifies glioma decomposition outputs.
- `scripts/verify_training_environment.py` — Reports Python, packages, CUDA, GPU, and paths.

## 2. Experiments executed

| Registry ID | Registry status | Folder | Result | Key metric/result | Decision file | Gate outcome |
|---|---|---|---|---|---|---|
| `glioma_segmentation_improvement` | REGISTERED_NOT_STARTED | `experiments/glioma_segmentation_improvement` | NOT RUN | Umbrella registration; controlled child experiments E01–E08 completed separately | `experiments/glioma_segmentation_improvement/decision.md` | Pending |
| `e01_lcc_filter` | COMPLETE | `experiments/e01_lcc_filter` | FAIL | Glioma Dice 0.6050; Δ −0.0040 vs v1 | `experiments/e01_lcc_filter/decision.md` | Required Δ ≥+0.02; failed |
| `e02_morphology` | COMPLETE | `experiments/e02_morphology` | FAIL | Best 3×3 glioma Dice 0.6120; Δ +0.0031 | `experiments/e02_morphology/decision.md` | Required Δ ≥+0.02; failed |
| `e03_tta` | COMPLETE | `experiments/e03_tta` | FAIL | Glioma Dice 0.4487; Δ −0.1603; pituitary harm | `experiments/e03_tta/decision.md` | Benefit and harm checks failed |
| `e04_multiscale` | COMPLETE | `experiments/e04_multiscale` | FAIL | Glioma Dice 0.5687; Δ −0.0403; pituitary harm | `experiments/e04_multiscale/decision.md` | Benefit and harm checks failed |
| `e05_glioma_weighted` | COMPLETE | `experiments/e05_glioma_weighted` | FAIL | Glioma Dice 0.5972; Δ −0.0117 | `experiments/e05_glioma_weighted/decision.md` | All three category checks failed |
| `e06_boundary_loss` | COMPLETE | `experiments/e06_boundary_loss` | FAIL | Glioma Dice Δ −0.0296; boundary-F1 Δ −0.0470 | `experiments/e06_boundary_loss/decision.md` | Both primary benefit checks failed |
| `e07_glioma_specialized` | COMPLETE | `experiments/e07_glioma_specialized` | FAIL | Routing accuracy 0.9721 passed; glioma Dice Δ −0.0287 | `experiments/e07_glioma_specialized/decision.md` | Glioma improvement check failed |
| `e08_ensemble` | COMPLETE | `experiments/e08_ensemble` | PASS locally | Glioma Dice 0.6324; +0.0522 vs eligible E07; +0.0234 vs v1 | `experiments/e08_ensemble/decision.md` | Local gate passed; parent G1 failed |
| `c01_vector_scaling` | COMPLETE | `experiments/c01_vector_scaling` | PASS | Lesion ECE 0.0505→0.0382; whole NLL 0.02323→0.01776 | `experiments/c01_vector_scaling/decision.md` | All C01 checks passed; 43,935 hard-mask pixels changed and reported |
| `c02_dirichlet` | COMPLETE locally, uncommitted | `experiments/c02_dirichlet` | FAIL | Lesion ECE 0.0505→0.0361; 43,228 pixels across 680 images changed | `experiments/c02_dirichlet/decision.md` | NLL/ECE checks passed; zero-change mask checks failed |
| `e08_final_selection` | REGISTERED_NOT_STARTED | `experiments/e08_final_selection` | NOT RUN | H7 awaits calibration selection and E08 reliability AUROC | `experiments/e08_final_selection/decision.md` | Pending |
| `sealed_final_evaluation` | REGISTERED_NOT_STARTED | `experiments/sealed_final_evaluation` | NOT RUN | Sealed classifier/segmenter/reliability evaluation unopened | `experiments/sealed_final_evaluation/decision.md` | Pending |
| `application_integrity_validation` | REGISTERED_NOT_STARTED | `experiments/application_integrity_validation` | NOT RUN | No application integrity metrics | `experiments/application_integrity_validation/decision.md` | Pending |
| `measurement_engine_validation` | REGISTERED_NOT_STARTED | `experiments/measurement_engine_validation` | NOT RUN | No synthetic measurement validation | `experiments/measurement_engine_validation/decision.md` | Pending |

## 3. Models trained

### Classifier

- **Architecture:** ImageNet `IMAGENET1K_V1` initialized ResNet-18 with a four-class glioma/meningioma/pituitary/no-tumor head.
- **Input:** grayscale → aspect-preserving 224×224 black padding → RGB → ImageNet normalization.
- **Training:** 3,570 images; batch 8; learning rate 0.0001; weight decay 0.0001; best epoch 7 of 12 completed.
- **Checkpoint:** `runs/classifier/20260921-170607-243133-baseline/best_model.pt`; SHA256 `3bc044027b83b2de225ebde3fb817be8827d2b427c599824697b13ff7a727535`.
- **Internal validation:** accuracy 0.9776; macro F1 0.9779; tumor sensitivity 0.9985; no-tumor specificity 0.9953; tumor ROC AUC 0.9987; one false negative and one false positive.
- **Status:** accepted frozen development classifier, but still experimental. Sealed-test and clinical performance are not established. Raw probabilities remain accepted because multiclass and binary temperature scaling did not show stable improvement.

### Segmenter v1

- **Architecture:** compact MONAI 2D U-Net, channels `[16, 32, 64, 128]`, two residual units, linear logit head, 401,277 parameters.
- **Training:** 2,719 pairs, 128×128 input, batch 8, AdamW, learning rate 0.001, balanced per-image BCE plus foreground Dice, AMP, no augmentation; best epoch 56 of 68 completed.
- **Top checkpoints:** `epoch_056_macro_dice_0.762662.pt` SHA256 `16dc8745…23de`; epoch 54 SHA256 `649761c3…26d`; epoch 57 SHA256 `0b199fea…527`.
- **Internal validation:** Dice 0.7694 overall, 0.7627 macro; glioma 0.6090, meningioma 0.8702, pituitary 0.8088; 83/680 below 0.50 and 17 zero-Dice.
- **Status:** accepted frozen v1 development segmenter and current final-model default; still a research model without sealed-test or clinical validation.

### Segmenter E08

- **Architecture:** no standalone neural checkpoint. E08 averages equal-weight raw probability maps from three compact 2D U-Net systems: E05 glioma-weighted, E06 boundary-loss, and classifier-routed E07. E07 routes predicted glioma to its specialist and other classes to v1.
- **Component checkpoints:** E05 `epoch_045_macro_dice_0.738782.pt` (`e17e7f5a…c6a`); E06 `epoch_040_macro_dice_0.745706.pt` (`b98afabc…3ab`); E07 `epoch_031_glioma_dice_0.580685.pt` (`bb68ce4f…be0`).
- **Internal validation:** glioma 0.6324, meningioma 0.8822, pituitary 0.8183; glioma +0.0234 versus v1.
- **Status:** local E08 experiment passed, but parent G1 failed because +0.0234 is below +0.0500. E08 is locked experimental v2 for calibration/reliability studies and has not replaced v1.

### Other trained segmentation candidates

| Candidate | Training change | Result | Status |
|---|---|---|---|
| E05 | Glioma loss ×3 | Glioma 0.5972 | Rejected |
| E06 | BCE + Dice + boundary loss | Glioma 0.5794; boundary F1 worsened | Rejected |
| E07 | Glioma-only specialist with classifier routing | Routed glioma 0.5802 | Rejected |
| Smoke/tiny-set diagnostics | Minimal capacity and memorization checks | Pipeline learnability established | Diagnostic only |

### Case-level reliability model

- **Model:** standardized logistic regression predicting `P(Dice ≥ 0.50)` with five-fold similarity-aware OOF evaluation.
- **Fourteen prediction-derived features:** empty prediction, predicted area fraction, whole-map mean/SD, predicted-region mean probability and entropy, boundary entropy, high-confidence foreground fraction, uncertain-pixel fraction, log component count, largest-component fraction, bounding-box fill ratio, normalized perimeter, and border-touch fraction.
- **Metrics on v1:** AUROC 0.9410 (95% CI 0.9184–0.9612), ECE 0.0212, Brier skill 0.4761; the highest-risk 20% captured 14/17 zero-Dice and 68/83 sub-0.50 cases.
- **Subgroups:** glioma AUROC 0.8475, meningioma 0.9943, pituitary 0.9741. Glioma is clearly weaker.
- **Status:** v1 reliability model passed its registered gate and is an operational proposal stored in `audit/segmenter_case_reliability.json`; the required E08 Stage 4 reliability model has not been built.

## 4. Findings documented

Every `audit/` file is described in Section 1. The principal findings are:

- **Dataset:** 6,000 readable classification images, 4,793 matched masks, no technical read failures, patient independence unverified, and physical measurements restricted to pixels until spacing is verified.
- **Classifier:** high internal validation performance, but a geometry-only diagnostic achieved ROC AUC 0.8190, demonstrating a source/geometry association that prevents naive clinical interpretation.
- **Glioma:** 60/204 cases are below Dice 0.50. Small masks are the clearest supported risk factor: 21/32 failures, failure-rate risk difference +0.430 with 95% CI +0.258 to +0.592. Diffuse and multifocal cases are descriptively harder but were not statistically established by the registered association rule.
- **Failure topology:** across 83 weak v1 cases, 84.3% are fragmented and 80.7% contain disconnected false-positive components.
- **Classifier/segmenter independence:** 78/83 segmentation failures and 16/17 zero-Dice cases still had correct classification.
- **Uncertainty:** zero-Dice cases are confident spatial errors. Classifier confidence correlates only 0.3252 with Dice and 0.1660 with probability inside the reference lesion; it cannot serve as a segmentation-reliability proxy.
- **Falsified hypothesis:** the proposed skull-base positional bias did not hold across all 17 zero-Dice cases: 11/17 centroids were below midline, only 1/17 below normalized y=0.60, and median lower-third predicted fraction was 0.008.
- **Calibration:** raw classifier outputs were already well calibrated; temperature scaling was rejected. v1 pixel temperature scaling improved foreground metrics but missed the ECE gate by 0.0014 and slightly worsened background NLL. C01 on E08 passed its gate. C02 improved probability metrics but failed mandatory hard-mask identity.
- **Reliability:** v1 prediction-derived logistic reliability passed all gates; boundary entropy was the strongest consistent failure association, and high-confidence foreground fraction was negatively associated with reliability. Coefficients are descriptive, not causal.
- **Reproducibility:** seeds, data hashes, and configurations are controlled, and checkpoint reload reproduced all 680 validation metrics exactly. Full training is controlled but not bitwise guaranteed because strict deterministic algorithms, CUBLAS workspace, and TF32 policy were not enforced or recorded at launch. The exact launch-time Git HEAD was not recorded; a source-equivalent commit is documented.

## 5. Pre-registration and governance

- **Primary preregistration:** `audit/preregistration.md`; registered 2026-09-24; commit `3f09bfd8913a56848401b745ecbfef3a3e99c227`.
- **Amendment 01:** `audit/preregistration_amendment_01.md`; registered 2026-09-24; commit `e4a4af6ccd09747d0cb6216a3ef011eee17dae08`. It defines the controlled E01–E08, C01–C07, R01–R10, and policy protocols.
- **Amendment 02:** `audit/preregistration_amendment_02.md`; registered locally 2026-09-24; SHA256 `c3627c7e25b89441c883ced726ca5f531d5c7833c2799017b9ce53c3e5a66891`. It adds H7. It currently has **no Git commit hash** because `.gitignore` excludes `audit/*`; the registry reference was committed in `9e1ce71`, but the amendment file itself was not force-added.
- **Review log entries:** 22 (`##` decision entries). The C02 outcome does not yet have a review-log closeout entry.

| Hypothesis | Current status |
|---|---|
| H1 — glioma improvement | **Failed/closed.** No candidate met parent G1; E08 improved glioma by +0.0234 rather than +0.0500. |
| H2 — final classification | **Not evaluated.** Sealed classifier test remains unopened. |
| H3 — final segmentation | **Not evaluated.** Sealed segmentation test remains unopened. |
| H4 — reliability generalization | **Not evaluated.** Sealed reliability evaluation remains unopened. |
| H5 — application integrity | **Not started.** Application integrity experiment remains pending. |
| H6 — measurement correctness | **Not started.** Synthetic measurement-engine validation remains pending. |

Additional H7 is registered locally and pending: E08 can be nominated only if an eligible calibration passes and the selected gate-passing E08 reliability model reaches OOF AUROC ≥0.941.

## 6. Stage status

| Stage | Status | Completed | Pending |
|---|---|---|---|
| Stage 0 — Foundation reset | **Complete** | Preregistration, immutable v1 manifest, experiment structure, review log | Amendment 02 Git-tracking defect must be repaired before relying on clone reproducibility |
| Stage 1 — Glioma decomposition | **Complete** | 60-case audit, all-204 boundary analysis, ten matched pairs, subgroup analysis, 19-paper review | No registered item pending |
| Stage 2 — Glioma attempts | **Complete** | E01–E08 and model selection | No intervention reached parent +0.05 gate; v1 retained |
| Stage 3 — Calibration | **Partial** | C01 PASS; C02 FAIL and locally complete | C02 commit/closeout; C03 isotonic; C04 split temperature; C05 reliability-weighted; C06 per-category; C07 hierarchical; C08 unified ECE; C09 six diagrams |
| Stage 4 — Reliability perfection | **Not started on E08** | Existing v1 reliability baseline only | R01–R10 on the locked calibration-selected E08 output |
| Stage 5 — Reliability policy | **Not started** | None | Cost model, threshold selection, three-state rule, UI copy, threshold sensitivity |

## 7. Constraints verified

| Constraint | Current evidence |
|---|---|
| Sealed test access | **0 images and 0 predictions accessed** across recorded experiments. |
| Retraining during Stage 3 | **0.** C01 and C02 fit calibration mappings only; neural weights were frozen. |
| Threshold changes | **0.** The numeric threshold stayed 0.50. C01 and C02 changed probabilities and therefore changed 43,935 and 43,228 hard-mask pixels respectively; those changes are explicitly reported rather than called threshold changes. |
| Hash/data verification | No cache, split, checkpoint, or population mismatch occurred in C01/C02. C02 artifact verification passed. |
| Verification failures | One transient governance-test failure occurred because `experiments/e08_final_selection/result.json` initially contained `{}`. It was corrected locally to a valid `NOT_RUN` record, after which **93 tests passed**. This correction is uncommitted. Experimental gate failures are scientific results, not integrity failures. |
| Clinical claims | None supported. Patient independence, external generalization, and clinical safety remain unverified. |

## 8. Git state

- **HEAD:** `baa94daa88276b64ef4518470a0f72f05c2e24c5`
- **Branch:** `master`
- **Working tree:** not clean. C02 results and one governance-schema correction are uncommitted. `STATUS_REPORT.md` is also untracked after this report is created.
- **Ignored but required:** `audit/preregistration_amendment_02.md` exists locally but is ignored and not tracked.

Current changes before this report was created:

```text
D  experiments/c02_dirichlet/artifacts/.gitkeep
M  experiments/c02_dirichlet/config.json
M  experiments/c02_dirichlet/decision.md
M  experiments/e08_final_selection/result.json
M  experiments/registry.json
?? experiments/c02_dirichlet/artifacts/experiment_manifest.json
?? experiments/c02_dirichlet/artifacts/fold_parameters.csv
?? experiments/c02_dirichlet/artifacts/lesion_centered_reliability.png
?? experiments/c02_dirichlet/artifacts/whole_image_reliability.png
?? experiments/c02_dirichlet/result.json
```

After report creation, Git also reports:

```text
?? STATUS_REPORT.md
```

Last ten commits:

```text
baa94da Implement Dirichlet pixel calibration
9e1ce71 Register conditional E08 selection and Dirichlet calibration
adddfcd Evaluate vector-scaling calibration
2c50d5b Implement vector-scaling calibration
e2d3aef Register vector-scaling calibration
98eeee4 Add verified ensemble calibration cache
73263fb Record glioma model selection evidence
57a58e3 Select experimental glioma candidate
87eb11b Evaluate fixed segmentation ensemble
07153d5 Implement fixed segmentation ensemble evaluation
```

## 9. Test status

- **Pytest:** 93 passed in the most recent complete run.
- **Ruff:** all checks passed in the most recent complete run; no Python source changed afterward.
- **C02 artifact verification:** passed for manifest hashes, result constraints, case count, and zero test access.
- **Known transient failure:** the newly registered H7 placeholder result lacked required identity fields; corrected locally before the 93-test passing run.

## 10. What is missing

### Not yet done

- Commit and push the completed C02 failure package and add its review-log decision.
- Force-add amendment 02 so the file referenced by the committed registry is reproducible from Git.
- Run and close C03–C07 in order.
- Produce C08 unified 15-bin ECE comparison across calibration studies.
- Produce C09 per-category pixel and case-reliability diagrams.
- Run E08-specific R01–R10 reliability studies and evaluate H7 AUROC ≥0.941.
- Build the Stage 5 cost model, select thresholds, define the three-state policy, write UI warnings, and run threshold sensitivity.
- Lock the final model, calibration, reliability model, and policy.
- Perform the one-time sealed classifier/segmenter/reliability evaluation.
- Build and validate the user-facing application.
- Run application integrity and measurement-engine validation.
- Validate on an external, patient-separated, clinically reviewed dataset; none is currently registered.

### Blocked items

- **H7 final E08 decision** is blocked until C03–C09 and Stage 4 finish.
- **One-time sealed evaluation** is intentionally blocked until every final artifact and policy is locked.
- **Clone-level governance reproducibility** is blocked by the untracked amendment 02 file even though its hash is stored in the committed registry.
- **Patient-independent performance claims** are blocked because BRISC does not provide verified patient identifiers for the current split.
- **Clinical and cross-dataset claims** are blocked because there is no external clinical validation dataset or clinician-reviewed prospective evaluation.

### Deferred items

- Sealed-test access is deferred until final lock.
- Application integration is deferred until calibration, reliability, and policy selection are complete.
- Millimetre measurements are deferred unless verified pixel spacing is available; current outputs must remain in pixels.
- Seven-family tumor classification, cause, danger, organ-failure prediction, and depth beneath skull remain outside the registered scope.

## Final integrity statement

The repository contains a well-audited internal research prototype with complete foundation, failure decomposition, and controlled glioma intervention work. It does not yet contain a locked E08 reliability model, a finished calibration comparison, a reliability policy, a completed application, or a sealed-test result. Frozen v1 remains the accepted development segmenter until the registered H7 pathway is completed and passes.
