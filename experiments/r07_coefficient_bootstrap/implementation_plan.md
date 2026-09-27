# R07 implementation plan

1. Verify Amendment 10 and every pinned preregistration, C03, fold, R01, R02, R06, feature-reference, and shared-code hash before fitting.
2. Reconstruct the same 680 C03-calibrated cases and 583 similarity groups. Confirm exactly 22,891 changed hard-mask pixels across 679 images at threshold 0.50.
3. Recompute the locked ten R02 inference-time features and binary reliability target. Enforce the feature contract before any fit.
4. Refit the unchanged standardized R02 logistic pipeline in the same five similarity-aware outer folds. Reproduce every committed fold coefficient within `1e-12`; preserve these vectors for fold sign agreement.
5. Fit one descriptive full-population standardized R02 pipeline. Use its coefficient sign only as the registered reference direction; do not report its in-sample performance as generalization evidence.
6. Generate exactly 2,000 similarity-group bootstrap resamples with seed 42. Refit the scaler and logistic estimator on every duplicated group resample. Stop if any draw lacks both outcomes, raises a fit error, or produces a non-finite coefficient.
7. For each feature, compute bootstrap mean, sample SD, percentile 95% interval, five-fold sign agreement, and 2,000-resample sign agreement. Rank by absolute bootstrap mean with feature-name tie-breaking.
8. Create a ten-panel coefficient-distribution figure with zero, mean, and percentile-limit markers. Save the complete long-form 20,000-row bootstrap coefficient table and compact summary CSV.
9. Compare the ten shared R02 coefficient summaries with the committed R01 fourteen-feature fold summaries, reporting signed and magnitude changes and sign changes without causal claims.
10. Write `audit/reliability_coefficients_with_ci.md`, experiment `result.json`, `decision.md`, supporting CSV/figure artifacts, and a hash manifest. Mark R07 `DESCRIPTIVE_COMPLETE` only when every completion condition passes.
11. Update the experiment registry and review log, run the full test suite and Ruff, independently verify all coefficient summaries and artifact hashes, then commit and push.
12. Pause after R07. Do not register or run R08 until the user approves.

No R07 estimator fitting or bootstrap execution occurs until the user approves this plan.
