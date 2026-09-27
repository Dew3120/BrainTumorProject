# Pre-registration amendment 11 — R08 prospective fold evaluation

**Status:** REGISTERED BEFORE R08 METRIC COMPUTATION

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**Operational R02 result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R07 result:** `experiments/r07_coefficient_bootstrap/result.json`, SHA256 `fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948`

## Prospective R08 hypothesis and gate

The accepted R02 ten-feature standardized logistic reliability estimator will achieve an unweighted mean held-out-fold failure-detection AUROC of at least `0.941` across the same five similarity-aware folds. Equality passes. This is the registered H7 reliability condition against the v1 comparator.

The controlling statistic is the arithmetic mean of the five independently computed held-out-fold AUROCs, not the aggregate 680-case OOF AUROC. Overall OOF AUROC and its interval are mandatory supporting evidence. If the controlling gate fails and the failure changes how R09–R10 should be registered, stop after R08 for user review.

## Locked data, estimator, and cross-fitting

- Population: the same 680 C03-calibrated internal-validation cases and 583 similarity groups used by R01–R07.
- Folds: the same five similarity-aware assignments, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Features: the same ten R02 prediction-derived inputs in the same order.
- Target: reliability is `P(Dice >= 0.50)` and failure risk is `1 - reliability`.
- Threshold: exactly `0.50`.
- Estimator: fold-local `StandardScaler` followed by `LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", class_weight=None, max_iter=5000, random_state=42)`.

For each fold, fit the scaler and estimator on the other four folds and evaluate only on the held-out fifth fold. No held-out case enters its own model fit. Do not refit across folds beyond this standard OOF protocol. Reported generalization metrics use the stitched five-fold OOF scores. A later full-population fit, if produced, is an operational proposal only and cannot enter reported metrics.

## Locked metrics

For each held-out fold, report:

- failure-detection AUROC using `risk = 1 - reliability`;
- 15-bin equal-frequency reliability ECE;
- Brier score;
- fold-training-prevalence baseline Brier, where the constant reliability prediction is the success prevalence in that fold's training partition;
- Brier skill `1 - Brier_model / Brier_baseline`.

Report the unweighted arithmetic mean and sample SD (`ddof=1`) across the five folds for AUROC, ECE, Brier, baseline Brier, and Brier skill.

On the stitched 680-case OOF vector, report overall failure-detection AUROC with a 2,000-resample similarity-group bootstrap percentile 95% interval, using seed 42. Use the same resamples to report true-category glioma, meningioma, and pituitary AUROCs with percentile intervals and valid/invalid counts; at least 1,900 subgroup repetitions must contain both outcomes. Full-population overall bootstrap requires all 2,000 valid draws.

Also report capture of zero-Dice cases among the 136 highest-risk cases, using normalized path as the deterministic tie-breaker.

## Gate and fail-closed rule

R08 passes only when:

`mean(fold_1_AUROC, ..., fold_5_AUROC) >= 0.941`

No CI-based substitution or relaxation is allowed. R08 stops without an accepted result if any source hash, population, path/order, C03 transform count, threshold, feature name/order/value, fold, group-isolation, R02 OOF score, estimator parameter, outcome-support, probability, bootstrap-validity, sealed-test, or retraining check fails. R02's complete 680-case OOF reliability vector and aggregate AUROC must reproduce within `1e-12`. Exactly 22,891 C03-changed pixels across 679 images must reproduce. No threshold change, classifier/neural-segmenter retraining, test-set access, hyperparameter tuning, feature substitution, or gate relaxation is permitted.
