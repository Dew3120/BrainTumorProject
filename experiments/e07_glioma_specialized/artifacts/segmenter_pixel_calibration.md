# Segmenter pixel confidence calibration

A single positive temperature was fitted through five-fold similarity-aware cross-fitting on 680 frozen BRISC segmentation-validation images. Reported calibrated metrics use held-out fold temperatures only.

## Temperatures

| Fold | Temperature | Fit images | Evaluation images |
|---:|---:|---:|---:|
| 1 | 2.13949179 | 544 | 136 |
| 2 | 2.11010583 | 544 | 136 |
| 3 | 2.14499522 | 543 | 137 |
| 4 | 2.12688099 | 545 | 135 |
| 5 | 2.12999465 | 544 | 136 |

Proposed all-validation temperature: `2.13039560`. It was not used for reported metrics.

## Region metrics

| Region | Metric | Raw | OOF calibrated | Change |
|---|---|---:|---:|---:|
| Lesion-centered foreground | NLL | 0.53734486 | 0.29798632 | -0.23935854 |
| Lesion-centered foreground | Brier | 0.08789618 | 0.07891661 | -0.00897957 |
| Lesion-centered foreground | ECE | 0.08203548 | 0.06040467 | -0.02163080 |
| Lesion-centered foreground | MCE | 0.46295093 | 0.30699356 | -0.15595737 |
| Background | NLL | 0.00647377 | 0.00680341 | +0.00032964 |
| Background | Brier | 0.00127504 | 0.00120461 | -0.00007043 |
| Background | ECE | 0.00166038 | 0.00470241 | +0.00304203 |
| Background | MCE | 0.98692273 | 0.97163921 | -0.01528352 |
| Whole image | NLL | 0.04010045 | 0.02511040 | -0.01499005 |
| Whole image | Brier | 0.00659138 | 0.00599931 | -0.00059207 |
| Whole image | ECE | 0.00615045 | 0.00476181 | -0.00138864 |
| Whole image | MCE | 0.48425383 | 0.32514757 | -0.15910625 |

## Brier paired bootstrap

| Region | Point change | 95% interval |
|---|---:|---:|
| Lesion-centered foreground | -0.00897957 | [-0.00976748, -0.00822403] |
| Background | -0.00007043 | [-0.00010558, -0.00003634] |
| Whole image | -0.00059207 | [-0.00065972, -0.00052175] |

## Foreground-region ECE by tumor type

| Type | Raw | OOF calibrated |
|---|---:|---:|
| glioma | 0.15128576 | 0.10961301 |
| meningioma | 0.04491173 | 0.03377957 |
| pituitary | 0.05949439 | 0.04627764 |

## Gate

**FAIL**

- `lesion_centered_nll_improved`: **PASS**
- `lesion_centered_brier_improved`: **PASS**
- `lesion_centered_ece_at_most_0.05`: **FAIL**
- `lesion_centered_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `background_nll_improved`: **FAIL**
- `background_brier_improved`: **PASS**
- `background_ece_at_most_0.02`: **PASS**
- `background_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `whole_image_nll_improved`: **PASS**
- `whole_image_brier_improved`: **PASS**
- `whole_image_ece_at_most_0.02`: **PASS**
- `whole_image_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `oof_hard_masks_bitwise_identical`: **PASS**
- `final_temperature_hard_masks_bitwise_identical`: **PASS**

Foreground diagram: `E:\BrainTumorProject\runs\segmentation_v2_glioma_specialized\20260924-105229-241365-specialist\segmenter_foreground_reliability_diagram.png`

Whole-image diagram: `E:\BrainTumorProject\runs\segmentation_v2_glioma_specialized\20260924-105229-241365-specialist\segmenter_whole_image_reliability_diagram.png`

No model was retrained, no test image or test prediction was accessed, the 0.50 threshold was unchanged, and no application integration was performed.
