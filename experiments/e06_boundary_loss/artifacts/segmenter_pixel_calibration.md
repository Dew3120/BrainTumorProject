# Segmenter pixel confidence calibration

A single positive temperature was fitted through five-fold similarity-aware cross-fitting on 680 frozen BRISC segmentation-validation images. Reported calibrated metrics use held-out fold temperatures only.

## Temperatures

| Fold | Temperature | Fit images | Evaluation images |
|---:|---:|---:|---:|
| 1 | 2.15200877 | 544 | 136 |
| 2 | 2.13763468 | 544 | 136 |
| 3 | 2.15582137 | 543 | 137 |
| 4 | 2.15822032 | 545 | 135 |
| 5 | 2.16621393 | 544 | 136 |

Proposed all-validation temperature: `2.15404716`. It was not used for reported metrics.

## Region metrics

| Region | Metric | Raw | OOF calibrated | Change |
|---|---|---:|---:|---:|
| Lesion-centered foreground | NLL | 0.52950421 | 0.29713313 | -0.23237108 |
| Lesion-centered foreground | Brier | 0.08901737 | 0.07987908 | -0.00913829 |
| Lesion-centered foreground | ECE | 0.08246893 | 0.05836315 | -0.02410578 |
| Lesion-centered foreground | MCE | 0.46516141 | 0.31681692 | -0.14834449 |
| Background | NLL | 0.00666236 | 0.00627272 | -0.00038964 |
| Background | Brier | 0.00125187 | 0.00116206 | -0.00008981 |
| Background | ECE | 0.00157683 | 0.00411500 | +0.00253817 |
| Background | MCE | 0.98771633 | 0.97280546 | -0.01491087 |
| Whole image | NLL | 0.03950798 | 0.02435964 | -0.01514835 |
| Whole image | Brier | 0.00662420 | 0.00599246 | -0.00063173 |
| Whole image | ECE | 0.00613947 | 0.00437701 | -0.00176246 |
| Whole image | MCE | 0.48194140 | 0.32635637 | -0.15558503 |

## Brier paired bootstrap

| Region | Point change | 95% interval |
|---|---:|---:|
| Lesion-centered foreground | -0.00913829 | [-0.00990965, -0.00841135] |
| Background | -0.00008981 | [-0.00012803, -0.00005268] |
| Whole image | -0.00063173 | [-0.00070746, -0.00055870] |

## Foreground-region ECE by tumor type

| Type | Raw | OOF calibrated |
|---|---:|---:|
| glioma | 0.14644553 | 0.10842051 |
| meningioma | 0.04850184 | 0.03734609 |
| pituitary | 0.06305878 | 0.04692901 |

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

Foreground diagram: `E:\BrainTumorProject\runs\segmentation_v2_boundary_loss\20260924-063051-647842-boundary\segmenter_foreground_reliability_diagram.png`

Whole-image diagram: `E:\BrainTumorProject\runs\segmentation_v2_boundary_loss\20260924-063051-647842-boundary\segmenter_whole_image_reliability_diagram.png`

No model was retrained, no test image or test prediction was accessed, the 0.50 threshold was unchanged, and no application integration was performed.
