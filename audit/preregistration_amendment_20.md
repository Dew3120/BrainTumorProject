# Pre-registration amendment 20 — out-of-distribution detection

**Status:** REGISTERED BEFORE O01–O06 EXECUTION

**Registered:** 2026-09-29

**Parent evidence:** `audit/stage6_cross_dataset.md`

## H9

An OOD detector trained on BRISC in-distribution data can reliably distinguish BRISC-like inputs from external or non-brain inputs without using reference masks, labels, or external performance data.

H9 passes only if the registered O05 held-out ensemble AUROC is at least `0.92`. No subgroup result, fitting result, or post-hoc variant may replace that gate.

## Frozen populations and splits

- **ID fitting:** 600 deterministic, similarity-group-aware samples from the BRISC development-training partition. Labels and reference masks are prohibited detector inputs.
- **ID evaluation:** all 680 positive BRISC segmentation-validation slices. These cases are held out from all detector fitting. BRISC sealed-test files are prohibited.
- **Near OOD:** Cheng/Figshare, OpenNeuro ds006248 v1.0.0, and UPENN-GBM v2. Each contributes 100 fitting and 100 evaluation images selected deterministically with patient-disjoint fitting/evaluation membership. Reference masks, diagnostic labels, and Stage-6 performance data are prohibited detector inputs.
- **Far OOD:** MedMNIST OrganAMNIST abdominal CT, CIFAR-10 natural images, seeded random-pixel images, and seeded uniform-gray images. Each contributes 100 fitting and 100 evaluation images. Official train partitions supply fitting images and official validation/test partitions supply evaluation images where available.
- Every sample is recorded with SHA256, source, release/provenance, detector split, and grouping identifier.

## Registered methods and gates

1. **O01 benchmark:** descriptive construction and integrity audit.
2. **O02 image statistics:** standardized L2 logistic regression using histogram entropy, mean, standard deviation, skewness, kurtosis, four fixed intensity-quartile fractions, aspect ratio, Canny edge density, Harris corner count, and source-file size. Gate: evaluation AUROC `>= 0.85`.
3. **O03 encoder Mahalanobis:** frozen v1-fallback compact segmenter bottleneck, spatially averaged; Ledoit-Wolf Gaussian fitted only to BRISC fitting embeddings. Score is Mahalanobis distance and the descriptive flag threshold is the fitting-ID 95th percentile. Gate: evaluation AUROC `>= 0.90`.
4. **O04 model confidence:** frozen classifier score `1 - max softmax` and frozen E08 segmenter whole-image mean Bernoulli entropy, assessed separately. Each gate: evaluation AUROC `>= 0.85`.
5. **O05 ensemble:** primary score is the equal-weight average of four fitting-ID empirical-CDF-normalized scores (O02, O03, classifier confidence, segmenter entropy). A secondary standardized L2 logistic regression may be fitted on detector-fitting data only. The secondary is selected only if held-out AUROC exceeds the primary by at least `0.01` and a 2,000-resample paired group-bootstrap 95% CI for the difference has lower bound above zero; otherwise the primary remains selected. Gate: selected held-out AUROC `>= 0.92`.
6. **O06 policy:** descriptive threshold and application contract. REVIEW is the 95th percentile and optional REFUSE is the 99th percentile of the selected ensemble score on BRISC fitting images. These thresholds are frozen before evaluation. ACCEPT is below REVIEW, REVIEW is at or above REVIEW and below REFUSE, REFUSE is at or above REFUSE. If REFUSE adds no useful separation it remains documented but is not enabled.

All AUROCs use the held-out evaluation split and 2,000 group-bootstrap percentile confidence intervals with seed `42`. Calibration uses 15 equal-frequency bins. Patient or similarity group is the resampling unit where available; otherwise the source sample is the unit.

## Integration policy

OOD policy executes before classifier, segmentation, C03 calibration, and R02 reliability display. Invalid or unavailable OOD scores fail closed to REVIEW. The segmentation threshold remains `0.50`; the reliability-policy threshold remains exactly `0.9490250173235835`.

## Novel versus expected

- Passing a registered gate is evidence that the corresponding method separates this registered retrospective benchmark.
- Failure is a retained negative result.
- Performance on external images does not establish clinical safety, diagnosis, universal OOD detection, or future-site generalization.

## Non-negotiable constraints

No BRISC sealed-test access; no neural retraining or refitting; no reference masks, labels, or external performance values as detector features; no changes to earlier thresholds; no gate relaxation after results; no experiment substitution or merging. Verification mismatches fail closed.
