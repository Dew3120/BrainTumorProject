# Classifier binary tumor/no-tumor calibration

One temperature was fitted to the collapsed binary tumor margin using binary tumor/no-tumor NLL. Four-class NLL was not used. Reported calibrated metrics come only from held-out folds.

## Fitted temperatures

| Fold | Temperature | Fit cases | Evaluation cases | Evaluation groups |
|---:|---:|---:|---:|---:|
| 1 | 1.34843865 | 714 | 179 | 157 |
| 2 | 1.06053554 | 714 | 179 | 157 |
| 3 | 1.34314119 | 714 | 179 | 157 |
| 4 | 1.25667864 | 715 | 178 | 156 |
| 5 | 1.35108131 | 715 | 178 | 157 |

Proposed all-validation operational temperature: `1.28596640`. This value was fitted on all 893 validation cases and was **not** used for any reported metric.

## Binary calibration metrics

| Metric | Raw | Cross-fitted calibrated | Change |
|---|---:|---:|---:|
| Binary NLL | 0.01924441 | 0.01961144 | +0.00036704 |
| Binary Brier | 0.00224595 | 0.00229208 | +0.00004613 |
| Binary ECE | 0.00206821 | 0.00231656 | +0.00024835 |
| Binary MCE | 0.01109602 | 0.01097680 | -0.00011921 |

## Paired similarity-group bootstrap

Calibrated-minus-raw changes from 2,000 paired resamples:

| Metric | Point change | Bootstrap mean | 95% interval |
|---|---:|---:|---:|
| Binary NLL | +0.00036704 | +0.00031475 | [-0.00283907, +0.00241481] |
| Binary Brier | +0.00004613 | +0.00004515 | [-0.00001605, +0.00009188] |
| Binary ECE | +0.00024835 | +0.00073515 | [-0.00018007, +0.00183112] |
| Binary MCE | -0.00011921 | -0.00061668 | [-0.00831802, +0.00811913] |

## Gate result

**FAIL**

- `binary_nll_improved`: **FAIL**
- `binary_brier_improved`: **FAIL**
- `binary_ece_at_most_0_05`: **PASS**
- `brier_delta_bootstrap_ci95_upper_below_zero`: **FAIL**
- `binary_threshold_decisions_identical`: **PASS**

The binary temperature is rejected. The classifier remains closed with raw probabilities for both four-class and tumor/no-tumor outputs.

Reliability diagram: `E:\BrainTumorProject\runs\classifier\20260921-170607-243133-baseline\classifier_binary_reliability_diagram.png`

## Limits

This is post-hoc evaluation on BRISC internal validation. The checkpoint was previously selected with this partition, the grouping is similarity-aware rather than patient-aware, and patient independence and cross-dataset calibration remain unverified.
