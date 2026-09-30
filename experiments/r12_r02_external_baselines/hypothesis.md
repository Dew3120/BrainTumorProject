# R12 — external trivial reliability baselines

## Registered hypothesis H11

On X02 OpenNeuro and X03 UPENN-GBM separately, frozen R02 failure-detection AUROC exceeds the AUROC of both a predicted-area-only comparator and a maximum-calibrated-foreground-probability comparator by at least `0.03`.

## Gate

All four cohort/comparator margins must satisfy `AUROC_R02 - AUROC_comparator >= 0.03`. Equality passes. Cohorts may not be pooled to rescue a failed comparison.

## Leakage prohibition

Both comparators are fitted only on the 680 BRISC development cases using the fixed similarity-aware folds. A full-BRISC comparator is then frozen and applied unchanged externally. X02/X03 targets, Dice, groups, and performance values cannot influence fitting, preprocessing, feature direction, coefficients, thresholds, or missing-value rules.

## Interpretation

This is an exploratory post-review baseline comparison. A pass would show that R02's external ranking contains more information than either registered one-feature baseline. It would not show external calibration, a transferable threshold, patient safety, or clinical utility.
