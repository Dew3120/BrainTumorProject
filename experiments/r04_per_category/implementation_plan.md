# R04 implementation plan

1. Verify Amendment 07 and every pinned C03, fold, classifier-prediction, R02 result, R02 OOF, R02 manifest, feature-reference, and common-library hash before fitting.
2. Reconstruct the same 680 C03 out-of-fold probability maps and reproduce exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
3. Recompute the locked ten R02 features, require exact names/order and finite values, and reproduce R02's per-case aggregate OOF reliability scores, aggregate AUROC, and six subgroup comparator AUROCs within `1e-12`.
4. Verify all five fold assignments and similarity groups. Confirm that every version/category outer-training partition and reported category contains both successes and failures.
5. Fit Version A independently for true glioma, meningioma, and pituitary. Inside every outer fold, select the category from the training folds, fit its scaler and logistic regression, and score only matching held-out cases.
6. Fit Version B independently for classifier-predicted glioma, meningioma, and pituitary using the same fold-local procedure. Exclude and record the single classifier-predicted `no_tumor` case; do not assign a fallback score.
7. For all six models, report OOF failure-detection AUROC, 2,000-resample group-bootstrap CI, 15-bin ECE/MCE, Brier score, fold-training-prevalence baseline Brier, Brier skill, and all fold coefficients with mean, sample SD, sign counts, and sign stability.
8. Compare Version A with the R02 aggregate model inside each true category and Version B with the R02 aggregate model inside each predicted route. Report paired AUROC differences and bootstrap intervals.
9. Report four-class routing accuracy across 680 cases, three-tumor-route accuracy and coverage among routed cases, and the complete confusion matrix. Explicitly assess glioma relative to R02's `0.8768434088`/`0.8709790210` comparators and pituitary relative to R02's `0.9571026723`/`0.9639118457` comparators.
10. Apply the unchanged conjunctive gate: all three Version A and all three Version B AUROCs must be at least `0.80`. Equality passes. Any one failure rejects R04.
11. Write case-level OOF scores, category metrics, bootstrap comparisons, coefficients, routing audit, result, decision, and experiment manifest. Update the registry and review log.
12. Run the full test suite and Ruff. Commit and push only if all integrity checks pass, then pause before R05.

Any registered mismatch stops R04 without an accepted decision artifact. No R04 fitting occurs until the user approves this plan.

