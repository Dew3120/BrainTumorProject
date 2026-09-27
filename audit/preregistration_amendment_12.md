# Pre-registration amendment 12 — R09 fold sensitivity

**Status:** REGISTERED BEFORE ALTERNATIVE FOLD GENERATION

**Registered:** 2026-09-28

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**Operational R02 result:** `experiments/r02_reduced_features/result.json`, SHA256 `fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2`

**R08 result:** `experiments/r08_prospective_fold_evaluation/result.json`, SHA256 `8e6ea05a18937949aa55aa4d285e535047e3c1dba3a5a98bc12b7b3e523f34a4`

## Prospective R09 question

R09 descriptively measures how sensitive the accepted R02 ten-feature logistic model is to three alternative, similarity-isolated fold assignments. It has no pass/fail gate and cannot change model selection.

## Locked alternative folds

Use `sklearn.model_selection.StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` independently with seeds `17`, `42`, and `93`. Stratify on the locked binary reliability target `Dice >= 0.50`; group on the same 583 similarity groups. Assign the returned held-out indices to fold numbers 1–5 in splitter iteration order.

Every case appears in exactly one held-out fold for each seed. Every group must occur in exactly one held-out fold. Every training and held-out partition must contain both outcomes. The three assignments must be distinct from one another and from the original fold assignment; otherwise R09 is `INVALID/STOP`.

## Locked data, model, and metrics

- Population: the same 680 C03-calibrated cases.
- Features: the same ten R02 prediction-derived features in the same order.
- Threshold: exactly `0.50`.
- Estimator: fold-local `StandardScaler` plus unchanged L2 logistic regression (`C=1`, `lbfgs`, no class weighting, maximum 5,000 iterations, estimator seed 42).
- Predictions: stitched OOF only; fit on four alternative folds and evaluate on the fifth.

For each alternative seed, report aggregate OOF failure-detection AUROC, 15-bin equal-frequency ECE/MCE, Brier score, fold-training-prevalence baseline Brier, Brier skill, and true-category glioma/meningioma/pituitary AUROCs.

Report the unweighted mean and sample SD across the three alternative-split aggregate AUROCs, ECEs, Briers, and Brier skills. Compare each alternative AUROC to the original R02 AUROC `0.9555020801044132` with an exact point difference and a paired 2,000-resample seed-42 similarity-group bootstrap percentile interval using the same draws for original and alternative scores.

The descriptive label is fixed before results:

- `NARROW`: AUROC sample SD `<=0.01` and maximum absolute alternative-minus-original AUROC difference `<=0.02`;
- `MODERATE`: AUROC sample SD `<=0.02` and maximum absolute difference `<=0.04`, when `NARROW` is not met;
- `MATERIAL`: otherwise.

This label is descriptive, not a gate, and cannot change selection.

## Fail-closed conditions

Stop without an accepted result if any source hash, population, case order, C03 transform count, threshold, feature contract, group count, split identity, fold coverage, group isolation, outcome support, probability, original R02 OOF reproduction, bootstrap, sealed-test, or retraining check fails. R02's original OOF vector and AUROC must reproduce within `1e-12`; exactly 22,891 C03-changed pixels across 679 images must reproduce. No threshold change, classifier/neural-segmenter retraining, test-set access, hyperparameter tuning, alternate seed substitution, or post-result fold regeneration is permitted.
