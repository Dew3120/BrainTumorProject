# R03 implementation plan

1. Verify Amendment 05 and every pinned R01, C03, cache, fold, classifier, split-snapshot, feature-reference, and common-library hash before feature extraction.
2. Reconstruct the same 680 C03 out-of-fold probability maps and reproduce exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
3. Verify the validation path order against the pinned cache and split snapshot. Require each source image to remain inside the project raw-data directory and match its split-snapshot SHA256 before reading pixels.
4. Recompute the fourteen unchanged R01 prediction-only features. Reproduce R01's per-case OOF scores and AUROC within 1e-12.
5. Transform each source image to the registered 128×128 grayscale letterbox representation without opening a reference mask. Compute the six registered features exactly, including all empty-mask rules, and require finite values.
6. Fit the twenty-feature model with the same five similarity-aware outer folds. Fit scaling and logistic regression only on each training partition and stitch held-out probabilities.
7. Report extended-model AUROC with 2,000-resample similarity-group bootstrap CI, 15-bin ECE/MCE, Brier metrics, zero-Dice capture, and true-category plus classifier-predicted-category AUROCs.
8. Report extended-minus-baseline AUROC and its paired group-bootstrap CI. Apply the fixed gate: extended AUROC must be at least `0.9595028142589118`.
9. Refit six leave-one-new-feature-out models. Report `extended AUROC - ablated AUROC` for each new feature.
10. For each new feature, run 30 within-held-out-fold permutations using the registered seeds. Report mean AUROC decrease and paired 2,000-resample group-bootstrap 95% CI.
11. Report all twenty standardized coefficients per fold, plus mean, sample SD, sign counts, and five-fold sign stability.
12. Write source-image hash verification, raw and extended feature tables, OOF scores, bootstrap comparisons, ablation/permutation tables, result, decision, and manifest. Update the registry and review log.
13. Run the full test suite and Ruff. Commit and push only if all integrity checks pass, then pause before R04.

Any registered mismatch stops R03 without an accepted decision artifact.
