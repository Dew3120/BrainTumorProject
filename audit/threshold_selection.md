# Reliability threshold selection

## Scope

This internal-development analysis uses the frozen C03-calibrated E08 predictions and R02 out-of-fold reliability scores for 680 BRISC validation cases. It changes no model or segmentation threshold and does not establish clinical utility or external generalization.

## Binary cost-optimal threshold

Review is required when reliability is strictly below `0.94902501732358346`. This threshold minimizes the fixed P01 cost `(TP + FP + 10×FN) / 680` over `0`, all distinct observed scores, and `1`.

- Flagged: **195 / 680** (28.7%)
- Failures caught: **81 / 82** (98.8%)
- Failures missed: **1 / 82** (1.2%)
- Acceptable masks reviewed: **114 / 598**
- Total cost: **205**
- Mean cost: **0.301471**

## Operating points

| Operating point | Threshold | Flagged | Failures caught | Failures missed | False alarms | Total cost | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| flag_nothing | 0 | 0 | 0 | 82 | 0 | 820 | 1.205882 |
| flag_everything | 1 | 680 | 82 | 0 | 598 | 680 | 1.000000 |
| conventional_0.50_reference | 0.5 | 66 | 48 | 34 | 18 | 406 | 0.597059 |
| cost_optimal | 0.949025017324 | 195 | 81 | 1 | 114 | 205 | 0.301471 |

`0.50` is a conventional comparison, not a previously deployed reliability threshold.

## Three-state optimum

- `theta_low`: **0.94902501732358346**
- `theta_high`: **0.94902501732358346**
- Review required: **195** cases
- Warning: **0** cases
- Standard: **485** cases
- Bad masks in Standard: **1**
- Bad masks outside mandatory review: **1**
- Total cost: **205.0**
- Mean cost: **0.301471**

The thresholds were obtained by exhaustively evaluating all `230860` valid observed-score pairs under the preregistered three-state costs.

## Descriptive category optima

| Category | Cases | Failures | Category optimum | Global cost | Category-optimal cost | Decisions changed | Material difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| glioma | 204 | 63 | 0.950106439124 | 148 | 148 | 0 | False |
| meningioma | 227 | 7 | 0.555686904198 | 18 | 11 | 7 | True |
| pituitary | 249 | 12 | 0.947061850066 | 39 | 38 | 1 | True |

The glioma optimum does not produce an operationally material difference under the declared definition: a different assignment plus lower subgroup cost than applying the global threshold. True-category optima are descriptive and are not deployment routing rules. The global threshold remains the operating candidate.

## Calibration and ranking context

- Failure-detection ROC AUC: **0.955502**
- Reliability ECE, 15 equal-frequency bins: **0.017133**
- Reliability MCE: **0.061806**

The reliability diagram is a calibration summary and does not become threshold-dependent merely because the selected operating point is marked.

## Verification and limits

All ten upstream hashes matched before results were written. The analysis used 680 unique OOF cases, 598 acceptable cases, 82 failures, and five original folds. Both cost vectors were unchanged. The neural segmentation threshold remained `0.50`; no sealed-test artifact was accessed; no classifier, segmenter, calibrator, or reliability estimator was retrained or refit.

Thresholds are optimized on internal development validation and may be optimistic. BRISC patient independence is unverified, category failure counts are small outside glioma, and real workflow costs or prevalence can change the preferred operating point.
