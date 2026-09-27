# Pre-registration amendment 03 — calibrated E08 reliability program

**Status:** REGISTERED BEFORE STAGE 4 ANALYSIS

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Prior amendments:**

- `audit/preregistration_amendment_01.md`, SHA256 `62abb1d13d3ffd16d3fdf41b461ad48aafc746caf4639f18e4545467ebf8c0fb`
- `audit/preregistration_amendment_02.md`, SHA256 `0153a8709f16ee38e4e2e2cea6044c840bb8b775971e2d090a02537a473e6a43`

## Registration timing and locked Stage 4 input

Stage 3 is complete. This amendment is registered before any R01–R10 result is computed. C03 isotonic calibration is the selected operational E08 calibrator. Every Stage 4 reported score, hard mask, feature, Dice-derived target, and failure label must use the five-fold out-of-fold C03 mapping in `experiments/c03_isotonic/artifacts/isotonic_mappings.json`, SHA256 `fdc1a735d78af8e8eb89392e696dc97fe6632729b44de6cf9e9e2d888ea760fa`. The final all-validation C03 mapping is an operational proposal only and cannot produce reported Stage 4 metrics.

The hard threshold remains exactly `0.50`. C03 changes 22,891 OOF pixels across 679 of 680 images. Stage 4 therefore recomputes reliability features and `Dice >=0.50` targets from the locked C03 OOF probabilities and resulting masks; it does not reuse raw-E08 reliability targets.

Reference masks may create training targets and evaluation metrics. They may never enter a reliability feature or inference input. True tumor type is descriptive only except where R04 explicitly fits a non-operational descriptive comparator. Classifier-predicted type may be an operational feature or router only where an experiment explicitly registers it.

“No retraining” in Stage 4 means no classifier or neural segmenter retraining. Fitting the registered logistic regression, tree, random-forest, and small-MLP reliability estimators is the purpose of R01–R10 and is authorized. The sealed test remains closed.

## Shared protocol

- Target: reliability is `P(Dice >= 0.50)`; failure risk is `1 - reliability`.
- Folds: the same five similarity-aware folds used in Stage 3, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Reported development predictions are out-of-fold. Preprocessing, scaling, imputation, and estimator fitting occur inside each outer training fold.
- Bootstrap: 2,000 paired similarity-group resamples, seed 42, percentile 95% intervals.
- Reliability ECE: 15 equal-frequency bins with stable score ordering.
- Baseline estimator: `StandardScaler` plus L2 logistic regression (`C=1`, `lbfgs`, no class weighting, seed 42), unless an experiment registers another estimator.
- Frozen v1 comparator: failure-detection AUROC `0.9409900910` (95% CI `0.9184–0.9612`), reliability ECE `0.021171`, and Brier skill `0.476052`.
- H7 reliability gate: selected calibrated-E08 reliability AUROC `>=0.941` using five-fold similarity-aware OOF predictions.
- Any hash mismatch, fold mismatch, non-finite score, reference-derived feature, sealed-test access, or threshold change produces `INVALID/STOP` before a decision artifact.

## Registered experiment sequence and prospective gates

R01–R10 run sequentially and independently. Each receives its own registration commit, result, decision, artifacts, and review-log entries. No item may be skipped, merged, or substituted.

| ID | Study | Gate or decision rule |
|---|---|---|
| R01 | Fourteen-feature ablation plus permutation importance | Descriptive; cannot be selected |
| R02 | Reduced logistic feature set selected by the rule registered after R01 and before R02 fitting | AUROC no more than `0.005` below the same-fold 14-feature baseline |
| R03 | Add circularity, solidity, boundary fractal dimension, predicted-boundary image gradient, centre distance, and vertical symmetry | AUROC improvement `>=0.005` over same-fold 14-feature baseline |
| R04 | True-category descriptive models and classifier-predicted-category operational models | AUROC `>=0.80` in each of the three category models for the relevant analysis |
| R05 | One logistic model with classifier-predicted category one-hot features | Predicted-glioma AUROC `>=0.88` |
| R06 | Gradient boosting, 2×32 MLP, and 100-tree random forest versus logistic regression | Best nonlinear AUROC improvement `>=0.02`; otherwise retain logistic interpretability |
| R07 | 2,000 group-bootstrap logistic coefficients | Descriptive; cannot change selection |
| R08 | Five prospective outer-fold reports | Mean fold AUROC `>=0.941`; overall OOF AUROC and CI also reported |
| R09 | Alternative StratifiedGroupKFold seeds 17, 42, and 93 | Descriptive stability; cannot change selection |
| R10 | Estimator seeds 7, 21, 42, 84, and 168 on identical folds | Descriptive stability; cannot change selection |

The R02 threshold, R03 feature list and threshold, R05 gate, and R08 gate supersede the earlier amendment 01 versions at the user's explicit direction. This is prospective: no R01–R10 result exists at registration, so no observed Stage 4 outcome informed these changes.

## Final reliability selection

R01 is descriptive and ineligible. Among R02–R06 models that pass their own gates, rank by highest aggregate OOF AUROC, then lowest ECE, then lowest Brier score, then experiment ID ascending. R07–R10 assess uncertainty and stability and cannot change selection. If none of R02–R06 passes, use the locked calibrated-E08 14-feature baseline for the H7 comparison, but it satisfies H7 only if its OOF AUROC is at least `0.941`.

Gates will not be relaxed after results are seen. Passing H7 does not establish clinical validity, patient-independent generalization, or external generalization; it only nominates E08 for the still-sealed final evaluation pathway.
