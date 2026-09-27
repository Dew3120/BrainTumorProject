# R01 implementation plan

1. Verify every registered hash, reconstruct each case's C03 out-of-fold probability map, and bitwise reproduce C03's reported 22,891 changed pixels across 679 images.
2. Recompute the C03 hard-mask Dice target for all 680 cases. Set reliability target to `1` when Dice is at least `0.50` and failure risk target to `1` when Dice is below `0.50`.
3. Extract the fourteen registered features from the calibrated probability map and calibrated `0.50` hard mask. The feature function accepts no reference mask, label, path, metadata, classifier output, or raw image.
4. Fit the full baseline with five-fold similarity-aware cross-fitting. Fit `StandardScaler` and L2 logistic regression only on each outer training fold; emit reliability scores only for that fold's held-out cases.
5. Repeat the identical cross-fitting procedure fourteen times, removing one feature each time. Report `baseline AUROC − ablated AUROC`; positive values indicate useful unique signal.
6. For permutation importance, retain each fitted full-fold model and permute one standardized input feature only within its held-out fold. Use 30 deterministic repetitions per feature. Stitch held-out scores across all folds for every repetition.
7. Compute permutation importance as baseline AUROC minus the mean permuted AUROC. Compute its 95% interval with 2,000 paired similarity-group bootstrap samples; each sample uses the same resampled cases for baseline and every permutation repetition.
8. Report full-baseline AUROC with group-bootstrap CI, 15-bin ECE, MCE, Brier score, training-prevalence baseline Brier, Brier skill, top-risk catastrophic capture, fold coefficients, coefficient sign stability, and descriptive true-label and classifier-predicted-category AUROCs.
9. Write all tables, OOF scores, fold coefficients, verification hashes, and the descriptive decision. Run the full test suite and Ruff, then commit and push R01 before registering R02.

R01 stops without result or decision artifacts if any source hash, fold membership, path alignment, C03 changed-mask count, feature finiteness check, or prediction-only feature contract fails.
