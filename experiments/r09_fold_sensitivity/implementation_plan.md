# R09 implementation plan

1. Verify Amendment 12 and every pinned C03, R02, R08, feature-reference, and shared-code hash.
2. Reconstruct the same 680 cases, 583 groups, C03 target, and ten R02 inference-time features at threshold 0.50.
3. Reproduce the original R02 OOF reliability vector and AUROC within `1e-12`.
4. Generate three five-fold `StratifiedGroupKFold` assignments using seeds 17, 42, and 93. Verify complete coverage, group isolation, outcome support, and distinct assignments.
5. For each alternative assignment, fit the unchanged standardized logistic pipeline inside each training partition and stitch held-out predictions.
6. Report per-seed AUROC, ECE/MCE, Brier, training-prevalence baseline Brier, Brier skill, and true-category AUROCs.
7. Report mean and sample SD across seeds. Compare each AUROC with original R02 using identical 2,000 seed-42 similarity-group bootstrap draws and paired percentile intervals.
8. Apply the preregistered descriptive dispersion label (`NARROW`, `MODERATE`, or `MATERIAL`) without treating it as a gate.
9. Write assignments, OOF scores, metrics, comparisons, bootstrap values, result, decision, `audit/fold_sensitivity.md`, and manifest. Update registry and review log.
10. Independently verify metrics and hashes, run all tests and Ruff, commit and push, report R09, and continue to R10 under standing approval.
