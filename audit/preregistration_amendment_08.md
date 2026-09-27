# Pre-registration amendment 08 — R05 category-aware reliability model

**Status:** REGISTERED BEFORE R05 FITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R02 source result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R04 source result:** `experiments/r04_per_category/result.json`, SHA256 `e47d380024953f9e77be1d1916f4eeb0f103a72a12070ecc3a5708d43fc39046`

## Prospective R05 hypothesis

A single category-aware reliability estimator using the locked R02 ten-feature vector plus the frozen classifier's predicted tumor category will achieve failure-detection AUROC of at least `0.88` among cases routed as glioma by the classifier.

The gate is:

`classifier-predicted-glioma OOF AUROC >= 0.88`

Equality passes. This is the only R05 selection gate. Overall, true-category, other predicted-route, calibration, Brier, coefficient, and comparison results are mandatory descriptive reports and cannot override the gate.

## Locked population and exclusion

- Source population: the same 680 BRISC internal segmentation-validation cases used by R01–R04.
- Included population: the 679 cases whose frozen classifier prediction is glioma, meningioma, or pituitary.
- Excluded population: the single frozen-classifier-predicted `no_tumor` case, `data/raw/brisc/brisc2025/classification_task/train/meningioma/brisc2025_train_01380_me_ax_t1.jpg`.
- The excluded case enters neither training nor evaluation, receives no R05 reliability score, and cannot be reassigned or represented by an all-zero category vector.
- Calibration: each included case's locked C03 five-fold out-of-fold isotonic mapping.
- Threshold: exactly `0.50`.
- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.

## Locked feature contract

The first ten inputs are the R02 features in their committed order:

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

Append exactly three mutually exclusive binary columns in this order:

11. `predicted_category_glioma`
12. `predicted_category_meningioma`
13. `predicted_category_pituitary`

All three indicators are retained. They encode only the frozen classifier's discrete predicted label. True tumor type and classifier probabilities are prohibited model inputs. The ten numeric features derive only from the C03 out-of-fold probability map or its threshold-`0.50` prediction. Reference masks create the target and evaluation metrics only.

## Locked cross-fitting protocol

Use the same five outer folds as R02. In each fold, remove the registered `no_tumor` prediction from both partitions, create the three one-hot indicators from the frozen classifier prediction, fit `StandardScaler` across all 13 columns on the four-fold training partition, then fit L2 logistic regression with `C=1`, `lbfgs`, no class weighting, maximum 5,000 iterations, and seed 42. Score only the included cases in the held-out fold. All reported R05 probabilities are stitched out-of-fold predictions.

Every outer training partition and the combined included evaluation cohort must contain both reliability outcomes. Every reported true-category and predicted-category subgroup must contain both outcomes. A violation is `INVALID/STOP`; no pooled, fallback, or alternate estimator may substitute.

## Metrics and comparisons

Report on the included 679-case cohort:

- overall failure-detection AUROC with a 2,000-resample similarity-group bootstrap 95% interval;
- true-category AUROC and bootstrap interval for glioma, meningioma, and pituitary;
- classifier-predicted-category AUROC and bootstrap interval for glioma, meningioma, and pituitary;
- 15-bin equal-frequency ECE and MCE, Brier score, fold-training-prevalence baseline Brier, and Brier skill overall and for every true-category and predicted-category subgroup;
- all 13 standardized coefficients for every fold, plus mean, sample SD, sign counts, and five-fold sign stability;
- route coverage, the excluded-case record, and the frozen classifier confusion matrix.

Bootstrap uses 2,000 global similarity-group resamples, seed 42, and percentile 95% intervals. Overall draws exclude the registered `no_tumor` case. Subgroup draws restrict the same global samples to the evaluated category. Undefined subgroup draws lacking either outcome are recorded and excluded; at least 1,900 valid draws are required for each subgroup interval.

All R05-versus-R02 and R05-versus-R04 comparisons use identical cases and paired bootstrap draws. Published full-cohort values are also displayed where the user's requested comparator differs from the matched 679-case cohort.

### Locked R02 comparators

| Analysis | Overall | Glioma | Meningioma | Pituitary |
|---|---:|---:|---:|---:|
| Matched 679-case overall | `0.9550146579` | — | — | — |
| Published true-category | — | `0.8768434088` | `0.9941558442` | `0.9571026723` |
| Matched true-category | — | `0.8768434088` | `0.9931818182` | `0.9571026723` |
| Predicted-category | — | `0.8709790210` | `0.9968701095` | `0.9639118457` |

### Locked R04 comparators

| Analysis | Overall | Glioma | Meningioma | Pituitary |
|---|---:|---:|---:|---:|
| Matched 679-case Version A | `0.9296007267` | `0.8573680063` | `0.8204545455` | `0.9254571027` |
| Published Version A | — | `0.8573680063` | `0.8454545455` | `0.9254571027` |
| Version B predicted-category | `0.9308600685` | `0.8677156177` | `0.9029733959` | `0.9578512397` |

The R05 report must explicitly state whether predicted-glioma clears `0.88`, whether glioma improves over R02 and R04, and whether pituitary recovers, matches, or degrades relative to the identical-case R02 and R04 comparators.

## Gate and fail-closed decision

R05 passes only when its classifier-predicted-glioma point-estimate AUROC is at least `0.88`. The point estimate, not the confidence-interval bound, controls the registered gate.

R05 stops without an accepted result if any registered source hash, population, excluded path, route count, C03 transform count, threshold, feature name/order/value, one-hot exclusivity, fold, similarity group, baseline OOF vector, comparator, outcome support, finiteness, bootstrap-validity, sealed-test, or retraining check fails. The R02 aggregate OOF vector, R04 OOF vectors, and locked comparators must reproduce within `1e-12`. Exactly 22,891 C03-changed pixels across 679 images must reproduce. No true category, reference-mask value, classifier probability, path, identity, or metadata may enter the reliability estimator. No threshold change, classifier/neural segmenter retraining, test-set access, or gate relaxation is permitted.

