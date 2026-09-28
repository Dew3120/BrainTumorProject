# R10 implementation plan

1. Verify the Stage 4 parent registration and every pinned C03, original-fold, R02, R09, feature-reference, and shared-code hash.
2. Reconstruct the same 680 C03-calibrated cases, 583 similarity groups, original five folds, threshold-0.50 target, and ten R02 inference-time features.
3. Reproduce the committed R02 seed-42 OOF vector and aggregate AUROC within `1e-12`.
4. Repeat the identical five-fold OOF fitting protocol with logistic-estimator seeds 7, 21, 42, 84, and 168. Standard scaling and all non-seed parameters remain fixed.
5. Report per-seed AUROC, 15-bin ECE/MCE, Brier, fold-training-prevalence baseline Brier, and Brier skill.
6. Report unweighted mean and sample SD across seeds. Compare every seed with R02 aggregate AUROC and its OOF probability vector.
7. Use the same 2,000 seed-42 similarity-group bootstrap draws for paired seed-minus-R02 AUROC comparisons.
8. Apply the preregistered descriptive label: `EXACT` when every OOF probability matches seed 42 within `1e-12`; `NUMERICALLY_STABLE` when AUROC SD and maximum absolute AUROC change are each at most `0.001`; otherwise `SEED_SENSITIVE`. This is not a gate.
9. Write OOF scores, metrics, comparisons, bootstrap results, result, decision, `audit/seed_stability.md`, and manifest. Update the registry and review log.
10. Independently verify results, run all tests and Ruff, commit and push, then perform the registered final model selection and Stage 4 consolidation.
