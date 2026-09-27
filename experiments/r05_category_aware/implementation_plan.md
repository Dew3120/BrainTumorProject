# R05 implementation plan

1. Verify Amendment 08 and every pinned C03, fold, classifier-prediction, R02, R04, feature-reference, and common-library hash before fitting.
2. Reconstruct the same 680 C03 OOF probability maps and reproduce exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
3. Recompute the ten locked R02 features and reproduce the R02 aggregate per-case OOF vector and all registered comparators within 1e-12. Reproduce the pinned R04 OOF vectors and comparators within the same tolerance.
4. Verify the frozen classifier routes. Require exactly 203 glioma, 219 meningioma, 257 pituitary, and one no-tumor prediction. Require the no-tumor prediction to be the registered path, exclude it, and verify 679 included cases.
5. Append three mutually exclusive one-hot columns in the registered order. Verify every included row sums to exactly one and that no true label or classifier probability enters the matrix.
6. Use the same five similarity-aware folds. In each fold, fit `StandardScaler` and L2 logistic regression on the 13 columns from included outer-training cases only, then score included held-out cases.
7. Report overall, true-category, and classifier-predicted-category AUROCs with 2,000-resample similarity-group bootstrap intervals. Apply the fixed gate only to classifier-predicted-glioma AUROC: it must be at least 0.88.
8. Report overall and per-category 15-bin ECE/MCE, Brier, fold-training-prevalence baseline Brier, and Brier skill.
9. Report all 13 standardized coefficients per fold, plus mean, sample SD, sign counts, and five-fold sign stability.
10. Compare R05 with the identical-case R02 aggregate model and R04 Versions A/B using paired bootstrap intervals. Display the requested published R02/R04 subgroup values alongside matched-cohort values where they differ.
11. Write the excluded-case audit, feature matrix, OOF scores, bootstrap comparisons, metric tables, coefficient table, result, decision, and experiment manifest. Update the registry and review log.
12. Run the full test suite and Ruff. Commit and push only if every integrity check passes, then pause before R06.

Any registered mismatch stops R05 without an accepted decision artifact. No R05 fitting occurs until the user approves this plan.

