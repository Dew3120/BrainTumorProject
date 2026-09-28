# P02 implementation plan

1. Verify all fixed P01, C03, E08-selection, R02, and OOF-table hashes.
2. Verify 680 unique OOF cases, 598 acceptable and 82 bad outcomes, five folds, finite bounded scores, and exact `Dice >=0.50` target agreement.
3. Verify unchanged binary and three-state cost vectors before writing results.
4. Sweep `0`, each distinct score, and `1` using strict-below review semantics.
5. Apply the registered binary cost and tie-breaking rule.
6. Compare the optimum with flag-none, flag-all, and the conventional `0.50` reference. No current deployed reliability default exists.
7. Exhaustively evaluate every observed-score pair satisfying `theta_low <= theta_high`, then apply the registered three-state cost and tie-breaking rule.
8. Repeat the binary sweep descriptively within true glioma, meningioma, and pituitary groups; retain the global threshold as the operational candidate.
9. Render the expected-cost curve, failure-detection ROC curve, and 15-bin equal-frequency reliability diagram with the selected operating point marked.
10. Independently recompute selections and verify all artifact hashes before committing.

No test artifact is accessed, no model is fit, and the neural segmentation threshold stays `0.50`.
