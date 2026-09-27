# R02 implementation plan

1. Verify Amendment 04, R01 result, R01 manifest, R01 OOF table, C03 mapping, calibration-cache, fold-assignment, feature-reference, and classifier-prediction hashes before loading analysis inputs.
2. Read the ten candidate names from the R02 config and independently compare them, including order, with R01's committed candidate screen. Stop if they differ.
3. Reconstruct all 680 C03 out-of-fold probability maps from the locked fold mappings. Reproduce exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
4. Recompute the fourteen prediction-only features and Dice >=0.50 targets from the C03 OOF maps. Verify the feature function accepts only probability and threshold and produces finite values.
5. Refit the full fourteen-feature five-fold OOF baseline with the same StandardScaler plus L2 logistic pipeline. Require AUROC to match R01 within 1e-12 before evaluating R02.
6. Fit the ten-feature reduced model independently inside each of the same five outer training folds and stitch only held-out probabilities into the reported OOF vector.
7. Compute paired full-versus-reduced failure-detection AUROCs. Apply the fixed R02 gate: reduced AUROC must be at least 0.9495028142589118.
8. Using 2,000 paired similarity-group bootstrap samples with seed 42, report AUROC confidence intervals for both models and the reduced-minus-full AUROC difference interval.
9. For both full and reduced models, report 15-bin equal-frequency ECE and MCE, Brier score, training-fold-prevalence baseline Brier, Brier skill, zero-Dice capture in the top 20% risk, and true-category plus classifier-predicted-category AUROCs.
10. Report the ten reduced-model standardized coefficients for all five folds, their mean, sample SD, sign counts, and all-fold sign stability.
11. Write OOF scores, metric comparisons, coefficients, verification evidence, result, decision, and manifest. Update the registry and review log. Run the full test suite and Ruff, then commit and push only if all checks pass.

R02 stops before accepting a result if any hash, population, candidate list, C03 transformation, fold, group, baseline reproduction, feature contract, finiteness, threshold, test-access, or retraining check fails.
