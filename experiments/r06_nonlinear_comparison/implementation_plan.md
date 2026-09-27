# R06 implementation plan

1. Verify Amendment 09 and every pinned C03, fold, R02, prior-stage, feature-reference, and common-library hash before changing the environment or fitting.
2. Add exactly `xgboost==3.1.2` through uv, update `pyproject.toml` and `uv.lock`, verify the installed version, and commit the environment lock before model fitting. Stop if resolution or import fails.
3. Reconstruct the same 680 C03 OOF probability maps and reproduce exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
4. Recompute the exact ten R02 features. Reproduce R02's per-case OOF logistic probabilities and AUROC `0.9555020801044132` within `1e-12`.
5. Fit XGBoost, the scaled two-layer MLP, and random forest independently inside the same five similarity-aware outer folds using only the registered fixed hyperparameters. Store all held-out probabilities and fold-support diagnostics.
6. Report overall AUROC with 2,000-resample group-bootstrap CI, ECE/MCE, Brier metrics, and zero-Dice capture for the baseline and all three nonlinear models.
7. Report true-category and classifier-predicted-category AUROCs with bootstrap intervals and valid/invalid draw counts for every model.
8. Compute paired nonlinear-minus-R02 AUROC differences and 95% intervals from identical bootstrap draws.
9. Compute XGBoost native gain, random-forest impurity importance, and 30-repeat held-out permutation importance for every feature in all three nonlinear models.
10. Apply the unchanged absolute gate `AUROC >= 0.9750` independently to each nonlinear model. If none passes, retain R02. If models pass, select the R06 winner by AUROC, ECE, Brier, then fixed model order and document interpretability limits.
11. Write OOF scores, metric tables, subgroup metrics, bootstrap comparisons, feature importance, fold diagnostics, result, decision, and manifest. Update the registry and review log.
12. Run the full test suite and Ruff. Commit and push only if every integrity check passes, then pause before R07.

Any registered mismatch stops R06 without an accepted decision artifact. No dependency installation or fitting occurs until the user approves this plan.

