# Pre-registration amendment 10 — R07 coefficient bootstrap

**Status:** REGISTERED BEFORE R07 BOOTSTRAP REFITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R01 source result:** `experiments/r01_ablation/result.json`, SHA256 `d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d`

**Operational R02 source result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R06 prior result:** `experiments/r06_nonlinear_comparison/result.json`, SHA256 `aa09898ee9e46f4d3edeecae34ec87d90d7107416b95376c2a7dee7a77eefb73`

## Prospective R07 question

R07 asks whether the standardized coefficients of the operational R02 ten-feature logistic reliability model have stable magnitude and direction under similarity-group resampling. It is descriptive. It has no performance gate, cannot replace R02, cannot change the Stage 4 model-selection outcome, and does not establish causal effects.

## Locked population, folds, target, and features

- Population: the same 680 BRISC internal segmentation-validation cases used by R01–R06.
- Similarity groups: the same 583 groups used by R01–R06.
- Calibration: the locked C03 five-fold out-of-fold isotonic probability maps.
- Threshold: exactly `0.50`.
- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Estimator: fold-local `StandardScaler` followed by `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight=None, max_iter=5000, random_state=42)`.

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

All ten inputs derive only from the C03 out-of-fold probability map or its threshold-`0.50` prediction. True category, reference-mask pixels or geometry, Dice/IoU, classifier predictions or probabilities, raw image intensity, path, identity, similarity-group identity, and metadata are prohibited model inputs. Reference masks create the binary training target and support evaluation only.

## Locked coefficient analyses

### Five-fold coefficients

Reproduce the five R02 outer-fold standardized coefficient vectors exactly from the same training partitions. For each feature, define fold sign agreement as the fraction of the five fold coefficients whose sign equals the sign of the coefficient from one descriptive full-population R02 pipeline fit. A coefficient equal to exactly zero has sign zero and agrees only with a zero reference sign.

### Similarity-group bootstrap

Use `2,000` resamples with NumPy random generator seed `42`. The resampling unit is the complete similarity group. On each repetition:

1. draw exactly 583 similarity-group identifiers with replacement from the 583 observed groups;
2. include every case belonging to each selected group once per time that group was drawn;
3. refit both `StandardScaler` and the unchanged R02 logistic regression on that duplicated resample;
4. record the ten standardized coefficients.

Every resample must contain both target outcomes, the fit must converge without an exception, and all coefficients must be finite. Any invalid resample makes R07 `INVALID/STOP`; it is not replaced with an extra draw.

For every feature, report the bootstrap mean, sample SD (`ddof=1`), 2.5th and 97.5th percentiles, and sign agreement with the full-population R02 coefficient. Also report the full-population coefficient and the five-fold mean and SD. Rank features by descending absolute bootstrap-mean coefficient, breaking ties by feature name.

Plot one histogram for each feature using all 2,000 bootstrap values, with vertical lines for zero, the bootstrap mean, and both percentile limits. A multi-panel figure is acceptable; no distribution may be omitted.

### Comparison with R01

For the ten shared features, compare the R07/R02 five-fold coefficient mean against the committed R01 fourteen-feature five-fold coefficient mean. Report both standardized means, their signed difference (`R02 - R01`), magnitude change, R01 sign agreement, R02 sign agreement, and whether the mean sign changed. Coefficient changes are descriptive associations under correlated predictors, not causal feature effects.

## Completion rule

R07 has no pass/fail performance gate. It is `DESCRIPTIVE_COMPLETE` only if all 2,000 registered resamples are valid, every required coefficient statistic and comparison is present, all ten distributions are rendered, the feature contract passes, the source and fold hashes match, the R02 five-fold coefficients reproduce within `1e-12`, and all integrity checks pass.

R07 does not select features, tune the estimator, or modify R02. Results will not be used to change the registered R02 operational model.

## Fail-closed conditions

Stop without an accepted R07 result if any pinned hash, population count, group count, case path/order, C03 transform count, threshold, feature name/order/value, fold, target, R02 fold coefficient, estimator parameter, resample count, bootstrap validity, coefficient finiteness, feature contract, sealed-test, or retraining check fails. Exactly 22,891 C03-changed pixels across 679 images must reproduce. No threshold change, classifier/neural-segmenter retraining, hyperparameter tuning, feature substitution, sealed-test access, or post-result protocol change is permitted.
