# X02 OpenNeuro pituitary external evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

This is an independent, retrospective research transport evaluation of the frozen E08 + C03 + R02 stack on OpenNeuro `ds006248` v1.0.0. It is not clinical validation.

## Cohort and execution

- 50 unique pituitary-adenoma subjects, 425 native source-positive coronal CE-T1 slices, and 624 total coronal slices.
- One first launch stopped fail-closed after 80 inferred slices and saved no result. Amendment 15 was registered before the one authorized restart.
- Five native source-positive edge masks (1.18%) vanish under the frozen 128×128 nearest-neighbor transform. They remain in the strict gate with zero scores and are listed in `target_not_representable.csv`.
- Frozen launch commit: `99f17f7ca02af5adf735bd511cb66f9154043110`. Completed external inference runs: 1; aborted attempts: 1.

## Five analytical views

| View | Population | Subject-macro Dice | 95% CI | Role |
|---|---:|---:|---:|---|
| Strict model-space | 425 slices / 50 subjects | 0.1272 | 0.0824–0.1778 | Registered gate; five erased targets receive 0 |
| Representable model-space | 420 slices / 50 subjects | 0.1295 | 0.0837–0.1813 | Secondary sensitivity |
| Native source-coordinate | 425 slices / 50 subjects | 0.1273 | 0.0826–0.1778 | Restored hard predictions at original resolution |
| Exploratory whole volume | 624 slices / 50 subjects | 0.1448 | 0.0922–0.2068 | Exploratory; not the gate |
| Confidence transport | Calibration n=420; reliability n=425 | Reliability AUROC 0.8719 | 0.8112–0.9162 | Pixel calibration + strict reliability outcome |

The five erased cases are not the explanation for the low Dice. Removing them raises subject-macro Dice only from `0.1272` to `0.1295`, and the native-resolution view remains `0.1273`.

## Classification

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Pituitary recall / patient accuracy | 0.9400 | ≥0.85 | Pass |
| Top-label ECE | 0.0762 | ≤0.05 | Fail |
| NLL | 0.1108 | Descriptive | — |
| Brier | 0.0539 | Descriptive | — |

The classifier predicts 47/50 subjects as pituitary and 3/50 as no tumor. OVR AUROC, no-tumor specificity, and binary tumor/no-tumor AUROC are not estimable in this single-category cohort.

## Segmentation and failure modes

- Strict pituitary subject-macro Dice: `0.1272` against the unchanged `0.75` gate — **fail**.
- Failures below Dice 0.50: 357/425; zero-Dice cases: 309; empty predictions: 81.
- Complete misses: 309/357 (86.6%). Non-empty zero-overlap: 228 (63.9%).
- Fragmented predictions: 153 (42.9%); disconnected false positives: 147 (41.2%).
- The deterministic panels show valid reference alignment and clean best-case overlap. Worst cases are model/task failures without a common coordinate transform defect.

## Pixel calibration

Pixel calibration uses the 420 representable references because lesion-centred regions are undefined for the five empty model-space targets.

| Region | Metric | Raw | C03 |
|---|---|---:|---:|
| Lesion-centred | NLL | 1.5332 | 1.0451 |
| Lesion-centred | Brier | 0.1478 | 0.1511 |
| Lesion-centred | ECE | 0.1348 | 0.1414 |
| Background | ECE | 0.0124 | 0.0105 |
| Whole image | ECE | 0.0162 | 0.0098 |

C03 improves lesion NLL, background calibration, and whole-image calibration, but lesion Brier and lesion ECE worsen. Calibration therefore does not transfer uniformly to the lesion region.

## Reliability transfer

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Failure AUROC | 0.8719 | ≥0.75 | Pass |
| AUROC lower CI | 0.8112 | ≥0.70 | Pass |
| ECE | 0.2551 | ≤0.05 | Fail |
| Brier skill | 0.7050 | >0 with lower CI >0 | Pass |
| Top-20% failure capture | 85/357 (23.8%) | ≥50% | Fail |

The top-20% gate is mathematically unattainable when 357/425 cases fail because 85 risk slots cannot contain half of 357 failures. The gate is retained and fails; it is not relaxed after seeing prevalence. At the fixed policy threshold, 385/425 cases require review, 340 failures are caught, 17 are missed, and total P01 cost is 555.

All five erased-target cases are `REVIEW_REQUIRED`, but only 1/5 enters the highest-risk 20%. Their mean reliability score is `0.7197` versus `0.4115` for representable cases. This supports the required limitation: target representability can interact with prediction-derived reliability inputs even though R02 never reads a reference mask.

## Domain gap

- Model-input mean Wasserstein distance: `0.1353`; fixed-histogram JS divergence: `0.9389`.
- Frozen-encoder centroid shift: `18.7692`.
- Reference foreground fraction: OpenNeuro `0.0104` vs BRISC validation `0.0160`.
- Fragmented reference fraction: OpenNeuro `0.1381` vs BRISC validation `0.0412`.
- Reliability-score Wasserstein distance: `0.4643`; policy review rate shifts from `28.8%` on BRISC development validation to `90.6%` on X02.

The strong intensity and encoder shifts, smaller lesions, more fragmented references, and broad complete-miss pattern support a genuine domain-transport failure rather than a five-slice preprocessing artifact.

## Registered decision

Failed gates: `classifier_top_label_ece_at_most_0_05`, `pituitary_subject_macro_dice_at_least_0_75`, `reliability_ece_at_most_0_05`, `top20_failure_capture_at_least_half`.

H8 is rejected for X02. No threshold, gate, model, calibration mapping, or reliability model is changed. X02 must not be used for tuning under this protocol.

## Verification and limits

- Independent recomputation checked 45 decision-driving values across 425 primary rows, 50 subjects, 624 volume slices, classifier aggregation, segmentation views, failure taxonomy, reliability bootstrap, policy counts, and all gates. Maximum absolute difference: `0.0`.
- Pixel-calibration and domain-gap standalone JSON artifacts match the completed result exactly. Per-pixel probability maps were not persisted, so the descriptive pixel-calibration aggregates cannot be independently regenerated without another inference run; no repeat inference was performed.
- BRISC development validation files used for frozen domain reference: 680. BRISC sealed-test files opened: 0.
- Neural retraining: 0. Calibrator refits: 0. Reliability refits: 0. Segmentation-threshold changes: 0. Policy-threshold changes: 0.
- This evaluates one retrospective, single-site, pituitary-only cohort. It does not establish prospective generalization, clinical validity, diagnostic safety, or suitability for patient care.
