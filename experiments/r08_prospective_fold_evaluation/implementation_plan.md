# R08 implementation plan

1. Verify Amendment 11 and every pinned C03, fold, R02, R07, feature-reference, and shared-code hash.
2. Reconstruct the same 680 C03-calibrated cases, 583 similarity groups, threshold-0.50 target, and ten inference-time features. Confirm exactly 22,891 changed pixels across 679 images.
3. Verify that every similarity group belongs to exactly one fold and every training and held-out partition contains both outcomes.
4. For each of the five original folds, fit a new scaler and unchanged R02 logistic estimator on the other four folds, then predict only the held-out fifth fold.
5. Reproduce the committed R02 680-case OOF reliability vector and aggregate AUROC within `1e-12` before accepting any R08 result.
6. Compute held-out AUROC, 15-bin ECE/MCE, Brier, fold-training-prevalence baseline Brier, and Brier skill for each fold. Report unweighted means and sample SDs.
7. Compute aggregate OOF AUROC with a 2,000-resample seed-42 similarity-group bootstrap percentile interval. Use the same draws for true-category glioma, meningioma, and pituitary AUROCs.
8. Compute zero-Dice capture among the 136 highest-risk cases with normalized-path tie-breaking.
9. Apply the unchanged controlling gate: unweighted mean of five held-out-fold AUROCs at least `0.941`; equality passes.
10. Write all OOF scores, fold metrics, bootstrap values, subgroup metrics, result, decision, audit summary, and manifest. Update the registry and review log.
11. Independently recompute every metric, run the full tests and Ruff, commit and push, then report R08 before proceeding.

The user's standing approval authorizes execution after this registration commit. If R08 fails and changes the registration basis for R09–R10, stop after R08.
