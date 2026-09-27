# Pre-registration amendment 07 — R04 per-category reliability models

**Status:** REGISTERED BEFORE R04 FITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R02 source result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R02 experiment manifest:** `experiments/r02_reduced_features/artifacts/experiment_manifest.json`, SHA256 `1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819`

## Prospective R04 hypothesis

Per-category reliability fitting with the locked R02 ten-feature input will achieve failure-detection AUROC of at least `0.80` in glioma, meningioma, and pituitary cases under both of these separately reported analyses:

- **Version A — descriptive:** one model per true tumor category.
- **Version B — operational:** one model per frozen-classifier-predicted tumor category.

Equality passes. The R04 experiment passes only if all six category/version point estimates pass. Failure of any one estimate rejects R04. Confidence intervals, calibration metrics, and comparisons are mandatory reports but do not create additional gates.

## Locked population, inputs, and feature contract

- Population: the same 680 BRISC internal segmentation-validation cases used by R01–R03.
- Calibration: each case's locked C03 five-fold out-of-fold isotonic mapping.
- Threshold: exactly `0.50`.
- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Version A category membership is the true label and is allowed only for this non-operational descriptive comparator. It may not be used as an inference input or operational router.
- Version B category membership is the frozen classifier's predicted label. The single predicted `no_tumor` case is excluded from Version B fitting, evaluation, and its three-category gate; its path and exclusion reason must be recorded. No replacement route or score may be invented for it.
- Reference masks may create the target and evaluation metrics only. They may not create a reliability feature.
- The frozen classifier's discrete predicted category may route Version B. Classifier probabilities, true labels, reference geometry, Dice/IoU, paths, patient/group identities, and dataset metadata are prohibited reliability-model inputs.

The ten inputs, in locked order, are:

1. `empty_prediction`
2. `predicted_area_fraction`
3. `whole_probability_mean`
4. `whole_probability_sd`
5. `predicted_region_entropy_mean`
6. `boundary_entropy_mean`
7. `high_confidence_foreground_fraction`
8. `uncertain_pixel_fraction`
9. `bounding_box_fill_ratio`
10. `normalized_perimeter`

Every feature is computed from the C03 out-of-fold probability map or its threshold-`0.50` prediction and is available at inference.

## Locked cross-fitting protocol

For each outer fold and each category, select the relevant category cases from the four training folds, fit a fresh `StandardScaler` plus L2 logistic regression, and evaluate only the matching category cases in the held-out fold. The estimator uses `C=1`, `lbfgs`, no class weighting, maximum 5,000 iterations, and seed 42. Scaling, prevalence estimation, and model fitting occur only inside that category's outer-training partition. Reported predictions are stitched out-of-fold scores only.

Each category-specific training partition must contain at least one success and one failure in every fold. Each reported category must contain both outcomes. Any violation is `INVALID/STOP`; no pooled, borrowed, or fallback model may replace a missing category fit.

Version A and Version B use the same case folds and feature values but are fitted independently. Version B route assignment comes only from the frozen classifier prediction available before reliability fitting.

## Metrics and comparisons

For each version and each of glioma, meningioma, and pituitary, report:

- failure-detection AUROC using `risk = 1 - reliability`;
- a 95% similarity-group bootstrap interval;
- 15-bin equal-frequency ECE and MCE with stable score ordering;
- Brier score;
- the fold-specific training-prevalence baseline Brier score;
- Brier skill `1 - Brier_model / Brier_baseline`;
- standardized coefficients for each fold, their mean, sample SD, sign counts, and five-fold sign stability.

Bootstrap uses the same 2,000 global similarity-group resamples and seed 42 as R01–R03. Each resample is restricted to the category being evaluated. Resamples lacking either outcome have undefined AUROC and are excluded from the percentile interval, with the valid/invalid counts reported. At least 1,900 of 2,000 resamples must be valid for every reported category/version interval or R04 stops as invalid. Gate decisions use the full-sample OOF point estimates, not confidence-interval bounds.

Compare each Version A category against the R02 aggregate ten-feature score restricted to the same true category. Compare each Version B category against the R02 aggregate ten-feature score restricted to the same classifier-predicted route. Report the per-category AUROC difference and its paired group-bootstrap interval. These comparisons are descriptive and do not alter the `0.80` gate.

The locked R02 comparators are:

| Analysis | Glioma | Meningioma | Pituitary |
|---|---:|---:|---:|
| True-category subgroup | `0.8768434088` | `0.9941558442` | `0.9571026723` |
| Predicted-category subgroup | `0.8709790210` | `0.9968701095` | `0.9639118457` |

The R04 report must explicitly state whether per-category fitting improves glioma over the appropriate R02 comparator and whether pituitary recovers, preserves, or worsens relative to R02. R03's pituitary decrease to `0.9198312236` is a confirmed feature-set-sensitivity finding, not the R04 comparator.

## Routing evaluation

Version B must report:

- four-class frozen-classifier routing accuracy across all 680 cases;
- three-tumor-route accuracy among the 679 cases routed to glioma, meningioma, or pituitary;
- route coverage (`679/680` expected);
- the full true-label by predicted-label confusion matrix.

True labels are used only to evaluate routing accuracy and Version A; they do not select a Version B model at inference.

## Gate and fail-closed decision

R04 passes only when all six conditions hold:

`Version A glioma AUROC >= 0.80`

`Version A meningioma AUROC >= 0.80`

`Version A pituitary AUROC >= 0.80`

`Version B predicted-glioma AUROC >= 0.80`

`Version B predicted-meningioma AUROC >= 0.80`

`Version B predicted-pituitary AUROC >= 0.80`

R04 stops without an accepted result if any source hash, population, C03 transform count, threshold, feature name/order/value, fold, similarity group, route assignment, baseline OOF vector, category membership, outcome support, finiteness, bootstrap-validity, sealed-test, or retraining check fails. The R02 aggregate OOF vector and every locked subgroup comparator must reproduce within `1e-12`. Exactly 22,891 C03-changed pixels across 679 images must reproduce. No threshold change, classifier/neural segmenter retraining, test-set access, post-hoc category pooling, or gate relaxation is permitted.

