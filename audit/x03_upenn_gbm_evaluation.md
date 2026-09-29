# X03 UPENN-GBM external evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

This is an independent, retrospective research transport evaluation of the frozen E08 + C03 + R02 stack on the registered UPENN-GBM derivative release. Glioblastoma is a subtype of glioma, so the registered mapping is coarse; these results are not directly comparable to the heterogeneous BRISC glioma category. This is not clinical validation.

## Cohort and execution

- 147 unique subjects, 9,464 tumor-bearing axial slices, and 22,785 total axial slices.
- Eighty-two tumor masks from 54 subjects vanish under the frozen 240×240 → 128×128 nearest-neighbor target transform. They remain in the strict gate with zero scores and are listed by identifier in `target_not_representable.csv`.
- Attempts 1–3 stopped fail-closed for target representability, a native-datatype loader defect, and a bootstrap-membership defect. Amendment 19 authorized attempt 4, which completed from frozen launch commit `a786ad39f93bce7bf679ae8c6c61b1ebed74229c`.
- Attempt 4 inferred all 22,785 slices once. The neural models, C03 mapping, R02 model, image adapter, mask transform, and registered thresholds remained frozen.

## Five analytical views

| View | Population | Subject-macro Dice | 95% CI | Role |
|---|---:|---:|---:|---|
| Strict model-space | 9,464 slices / 147 subjects | 0.2208 | 0.2004–0.2422 | Registered gate; 82 erased targets receive 0 |
| Representable model-space | 9,382 slices / 147 subjects | 0.2226 | 0.2021–0.2441 | Secondary sensitivity |
| Native source-coordinate | 9,464 slices / 147 subjects | 0.2205 | 0.2001–0.2418 | Restored hard predictions at original resolution |
| Exploratory whole volume | 22,785 slices / 147 subjects | 0.2428 | 0.2206–0.2655 | Exploratory; not the gate |
| Confidence transport | Calibration n=9,382; reliability n=9,464 | Reliability AUROC 0.8316 | 0.8120–0.8501 | Pixel calibration + strict reliability outcome |

The 82 erased targets are not the main explanation for the low Dice. Removing them raises subject-macro Dice only from `0.2208` to `0.2226`, and the native-resolution view remains `0.2205`.

## Classification

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Glioma recall / patient accuracy | 0.4218 | ≥0.85 | Fail |
| Top-label ECE | 0.2454 | ≤0.05 | Fail |
| NLL | 0.8542 | Descriptive | — |
| Brier | 0.6265 | Descriptive | — |

The classifier predicts 62/147 subjects as glioma and 85/147 as no tumor. OVR AUROC, no-tumor specificity, and binary tumor/no-tumor AUROC are not estimable in this single-category cohort.

## Segmentation and failure modes

- Strict glioma subject-macro Dice: `0.2208` against the registered `0.55` gate — **fail**.
- Failures below Dice 0.50: 7600/9,464; zero-Dice cases: 4263; empty predictions: 470.
- Complete misses: 4263/7600 (56.1%). Non-empty zero-overlap: 3793 (49.9%).
- Fragmented predictions: 3953 (52.0%); disconnected false positives: 3467 (45.6%).
- The deterministic panels show valid reference alignment and clean best-case overlap. Worst cases are model/task failures without a common coordinate transform defect.

## Pixel calibration

Pixel calibration uses the 9,382 representable references because lesion-centred regions are undefined for the 82 empty model-space targets.

| Region | Metric | Raw | C03 |
|---|---|---:|---:|
| Lesion-centred | NLL | 1.2954 | 0.8634 |
| Lesion-centred | Brier | 0.1740 | 0.1717 |
| Lesion-centred | ECE | 0.1430 | 0.1420 |
| Background | ECE | 0.0108 | 0.0091 |
| Whole image | ECE | 0.0191 | 0.0106 |

C03 lowers NLL, Brier, ECE, and MCE numerically in the lesion-centred region and also improves background and whole-image summaries. Lesion-centred ECE remains high at `0.1420`, so calibration transport remains inadequate despite the improvement.

## Reliability transfer

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Failure AUROC | 0.8316 | ≥0.75 | Pass |
| AUROC lower CI | 0.8120 | ≥0.70 | Pass |
| ECE | 0.3391 | ≤0.05 | Fail |
| Brier skill | 0.5715 | >0 with lower CI >0 | Pass |
| Top-20% failure capture | 1886/7600 (24.8%) | ≥50% | Fail |

The top-20% gate is mathematically unattainable when 7,600/9,464 cases fail because 1,893 risk slots cannot contain half of those failures. The gate is retained and fails. At the fixed policy threshold, 7,808/9,464 cases require review, 6,784 failures are caught, 816 are missed, and total P01 cost is 15,968.

All 82 erased-target cases are retained in reliability evaluation. Their mean score is `0.3729` versus `0.5375` for representable cases, and 20/82 enter the highest-risk 20%. Target representability can affect prediction-derived reliability inputs even though R02 never reads a reference mask.

## Domain gap

- Model-input mean Wasserstein distance: `0.0254`; fixed-histogram JS divergence: `0.1345`.
- Frozen-encoder centroid shift: `13.1125`.
- Reference foreground fraction: UPENN-GBM `0.0215` vs BRISC validation `0.0200`.
- Fragmented reference fraction: UPENN-GBM `0.5280` vs BRISC validation `0.0490`.
- Reliability-score Wasserstein distance: `0.1637`; policy review rate shifts from `69.1%` on BRISC development validation to `82.5%` on X03.

The strong intensity and encoder shifts, much greater reference fragmentation, and broad complete-miss pattern support a genuine domain-transport failure rather than an explanation limited to the 82 erased targets. Mask-derived domain features were bootstrapped on exactly 9,382 representable rows; input, encoder, reliability, and policy features used exactly 9,464 primary rows under Amendment 19.

## Registered decision

Failed gates: `classifier_glioma_recall_at_least_0_85`, `classifier_top_label_ece_at_most_0_05`, `glioma_subject_macro_dice_at_least_0_55`, `reliability_ece_at_most_0_05`, `top20_failure_capture_at_least_half`.

H8 is rejected for X03. No threshold, gate, model, calibration mapping, or reliability model is changed. X03 must not be used for tuning under this protocol.

## Verification and limits

- Independent recomputation checked 45 decision-driving values across 9,464 primary rows, 147 subjects, 22,785 volume slices, classifier aggregation, segmentation views, failure taxonomy, reliability bootstrap, policy counts, and all gates. Maximum absolute difference: `0.0`.
- Pixel-calibration and domain-gap standalone JSON artifacts match the completed result exactly. Per-pixel probability maps were not persisted, so the descriptive pixel-calibration aggregates cannot be independently regenerated without another inference run; no repeat inference was performed.
- BRISC development validation files used for frozen domain reference: 680. BRISC sealed-test files opened: 0.
- Neural retraining: 0. Calibrator refits: 0. Reliability refits: 0. Segmentation-threshold changes: 0. Policy-threshold changes: 0.
- This evaluates one retrospective, single-site, glioma-only cohort. It does not establish prospective generalization, clinical validity, diagnostic safety, or suitability for patient care.
