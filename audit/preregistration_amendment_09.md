# Pre-registration amendment 09 — R06 nonlinear reliability comparison

**Status:** REGISTERED BEFORE R06 DEPENDENCY INSTALLATION OR FITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R02 source result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R05 prior result:** `experiments/r05_category_aware/result.json`, SHA256 `3f08ad6dde731209c53d84c4b01c6709bcec64dca536b801cd4e78d94abf2d8d`

## Prospective R06 hypothesis

At least one of three fixed nonlinear reliability estimators — XGBoost, a two-layer MLP, or a 100-tree random forest — will achieve five-fold out-of-fold failure-detection AUROC of at least `0.9750` on the same 680 C03-calibrated cases and locked R02 ten-feature input.

The user-specified controlling gate is the absolute threshold `AUROC >= 0.9750`; equality passes. The exact full-population R02 comparator is `0.9555020801044132`, while the matched-679 comparator cited when deriving the rounded gate is `0.9550146579132087`. R06 will report exact AUROC changes against the full 680-case R02 comparator, but the fixed absolute `0.9750` threshold controls the decision and will not be shifted after outcomes are observed.

Each nonlinear model receives an independent pass/fail result. R06 passes when at least one passes. If none passes, retain R02's interpretable logistic regression. If more than one passes, the highest aggregate OOF AUROC is the R06 winner, followed by lower ECE, lower Brier, then model order XGBoost, MLP, random forest.

## Locked population and feature contract

- Population: the same 680 BRISC internal segmentation-validation cases used by R01–R04.
- Calibration: each case's locked C03 five-fold out-of-fold isotonic mapping.
- Threshold: exactly `0.50`.
- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Reported probabilities: out-of-fold only.

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

Every input derives only from the C03 out-of-fold probability map or its threshold-`0.50` prediction. True category, reference-mask values or geometry, Dice/IoU, classifier predictions or probabilities, raw image intensity, path, identity, similarity group, and metadata are prohibited model inputs.

## Locked estimators

### XGBoost

Install and lock `xgboost==3.1.2` after plan approval and before any R06 fit. Use `XGBClassifier` with:

- `n_estimators=200`
- `max_depth=3`
- `learning_rate=0.03`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `min_child_weight=1`
- `gamma=0`
- `reg_alpha=0`
- `reg_lambda=1`
- `objective="binary:logistic"`
- `eval_metric="logloss"`
- `tree_method="hist"`
- `random_state=42`
- `n_jobs=1`

No scaling, early stopping, validation-set tuning, or class weighting is permitted.

### Small MLP

Use scikit-learn `MLPClassifier` after fold-local `StandardScaler` with:

- `hidden_layer_sizes=(32, 32)`
- `activation="relu"`
- `solver="adam"`
- `alpha=0.0001`
- `batch_size=64`
- `learning_rate="constant"`
- `learning_rate_init=0.001`
- `max_iter=1000`
- `shuffle=True`
- `random_state=42`
- `tol=0.0001`
- `early_stopping=False`
- `n_iter_no_change=50`

No architecture, learning-rate, stopping, or regularization change is permitted after results appear. A convergence warning is recorded; non-finite or absent probabilities are `INVALID/STOP`.

### Random forest

Use scikit-learn `RandomForestClassifier` with:

- `n_estimators=100`
- `criterion="gini"`
- `max_depth=None`
- `min_samples_split=2`
- `min_samples_leaf=2`
- `max_features="sqrt"`
- `bootstrap=True`
- `class_weight=None`
- `random_state=42`
- `n_jobs=1`

No scaling or hyperparameter tuning is permitted.

R06 fits only reliability estimators. It does not retrain the frozen classifier or any neural segmenter.

## Metrics and comparisons

For R02 logistic regression and each nonlinear model, report:

- overall failure-detection AUROC with a 2,000-resample similarity-group bootstrap 95% interval;
- 15-bin equal-frequency ECE and MCE;
- Brier score, fold-training-prevalence baseline Brier, and Brier skill;
- zero-Dice capture among the 136 highest-risk cases, with normalized path as deterministic tie-breaker;
- true-category and classifier-predicted-category AUROC for glioma, meningioma, and pituitary, with bootstrap intervals and valid/invalid replicate counts.

Use the same 2,000 global similarity-group bootstrap samples, seed 42, for all models. Paired nonlinear-minus-R02 AUROC differences use these identical samples and percentile 95% intervals. At least 1,900 subgroup replicates must contain both outcomes; otherwise R06 is invalid. Full-population bootstrap requires all 2,000 valid draws.

## Feature importance and interpretability

For XGBoost, report fold-mean native gain importance, treating absent features as zero. For random forest, report fold-mean impurity importance. For all three nonlinear models, compute held-out permutation importance for each feature using 30 repetitions per fold and deterministic seed `42 + 1000 × model_index + 100 × feature_index + repetition`. Permutation occurs only within the held-out fold. Stitch permuted fold predictions into a full OOF vector and report `baseline model AUROC - permuted AUROC`, averaged across repetitions, with the repetition 2.5th and 97.5th percentiles.

If a nonlinear model passes, discuss its native and permutation explanations, interaction capacity, stability, and loss of coefficient-level interpretability compared with R02. Passing does not itself establish causal feature importance. If none passes, retain R02 because the registered performance gain did not justify lower interpretability.

## Gate and fail-closed decision

Each nonlinear model passes only when:

`model OOF AUROC >= 0.9750`

R06 passes when any model passes. Equality passes. Calibration, subgroup, capture, and importance metrics are mandatory reports but do not create unregistered gates.

R06 stops without accepted results if `xgboost==3.1.2` cannot be locked; any source hash, package version, population, C03 transform count, threshold, feature name/order/value, fold, similarity group, baseline OOF vector, baseline AUROC, outcome support, probability finiteness/range, bootstrap-validity, sealed-test, or retraining check fails. R02's per-case OOF vector and AUROC must reproduce within `1e-12`. Exactly 22,891 C03-changed pixels across 679 images must reproduce. No threshold change, classifier/neural segmenter retraining, test-set access, hyperparameter tuning, model substitution, or gate relaxation is permitted.

