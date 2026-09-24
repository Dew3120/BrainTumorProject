# Segmenter pixel confidence calibration

A single positive temperature was fitted through five-fold similarity-aware cross-fitting on 680 frozen BRISC segmentation-validation images. Reported calibrated metrics use held-out fold temperatures only.

## Temperatures

| Fold | Temperature | Fit images | Evaluation images |
|---:|---:|---:|---:|
| 1 | 2.30281177 | 544 | 136 |
| 2 | 2.24148613 | 544 | 136 |
| 3 | 2.30718357 | 543 | 137 |
| 4 | 2.28450097 | 545 | 135 |
| 5 | 2.29202000 | 544 | 136 |

Proposed all-validation temperature: `2.28595990`. It was not used for reported metrics.

## Region metrics

| Region | Metric | Raw | OOF calibrated | Change |
|---|---|---:|---:|---:|
| Lesion-centered foreground | NLL | 0.55823989 | 0.30015480 | -0.25808509 |
| Lesion-centered foreground | Brier | 0.09051087 | 0.08063955 | -0.00987132 |
| Lesion-centered foreground | ECE | 0.08425939 | 0.05796845 | -0.02629094 |
| Lesion-centered foreground | MCE | 0.46843697 | 0.29385220 | -0.17458477 |
| Background | NLL | 0.00779462 | 0.00731693 | -0.00047769 |
| Background | Brier | 0.00150771 | 0.00143504 | -0.00007267 |
| Background | ECE | 0.00199766 | 0.00491095 | +0.00291329 |
| Background | MCE | 0.98610542 | 0.97319325 | -0.01291217 |
| Whole image | NLL | 0.04235902 | 0.02551622 | -0.01684280 |
| Whole image | Brier | 0.00694270 | 0.00627743 | -0.00066527 |
| Whole image | ECE | 0.00649801 | 0.00525690 | -0.00124112 |
| Whole image | MCE | 0.51953118 | 0.33970848 | -0.17982270 |

## Brier paired bootstrap

| Region | Point change | 95% interval |
|---|---:|---:|
| Lesion-centered foreground | -0.00987132 | [-0.01076112, -0.00903000] |
| Background | -0.00007267 | [-0.00011786, -0.00003247] |
| Whole image | -0.00066527 | [-0.00076334, -0.00057631] |

## Foreground-region ECE by tumor type

| Type | Raw | OOF calibrated |
|---|---:|---:|
| glioma | 0.14251791 | 0.10145481 |
| meningioma | 0.05243319 | 0.03779289 |
| pituitary | 0.06676271 | 0.05013171 |

## Gate

**FAIL**

- `lesion_centered_nll_improved`: **PASS**
- `lesion_centered_brier_improved`: **PASS**
- `lesion_centered_ece_at_most_0.05`: **FAIL**
- `lesion_centered_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `background_nll_improved`: **PASS**
- `background_brier_improved`: **PASS**
- `background_ece_at_most_0.02`: **PASS**
- `background_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `whole_image_nll_improved`: **PASS**
- `whole_image_brier_improved`: **PASS**
- `whole_image_ece_at_most_0.02`: **PASS**
- `whole_image_brier_bootstrap_ci95_upper_below_zero`: **PASS**
- `oof_hard_masks_bitwise_identical`: **PASS**
- `final_temperature_hard_masks_bitwise_identical`: **PASS**

Foreground diagram: `E:\BrainTumorProject\runs\segmentation_v2_weighted\20260924-055049-892543-weighted\segmenter_foreground_reliability_diagram.png`

Whole-image diagram: `E:\BrainTumorProject\runs\segmentation_v2_weighted\20260924-055049-892543-weighted\segmenter_whole_image_reliability_diagram.png`

No model was retrained, no test image or test prediction was accessed, the 0.50 threshold was unchanged, and no application integration was performed.
