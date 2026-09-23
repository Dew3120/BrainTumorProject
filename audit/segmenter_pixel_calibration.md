# Segmenter pixel confidence calibration

A single positive temperature was fitted through five-fold similarity-aware cross-fitting on 680 frozen BRISC segmentation-validation images. Reported calibrated metrics use held-out fold temperatures only.

## Temperatures

| Fold | Temperature | Fit images | Evaluation images |
|---:|---:|---:|---:|
| 1 | 2.54956676 | 544 | 136 |
| 2 | 2.52113661 | 544 | 136 |
| 3 | 2.55248649 | 543 | 137 |
| 4 | 2.53493376 | 545 | 135 |
| 5 | 2.53518852 | 544 | 136 |

Proposed all-validation temperature: `2.53873405`. It was not used for reported metrics.

## Region metrics

| Region | Metric | Raw | OOF calibrated | Change |
|---|---|---:|---:|---:|
| Lesion-centered foreground | NLL | 0.59544472 | 0.28034549 | -0.31509923 |
| Lesion-centered foreground | Brier | 0.08040554 | 0.07155001 | -0.00885554 |
| Lesion-centered foreground | ECE | 0.07551321 | 0.05141609 | -0.02409712 |
| Lesion-centered foreground | MCE | 0.43750226 | 0.25688457 | -0.18061769 |
| Background | NLL | 0.00485816 | 0.00515855 | +0.00030038 |
| Background | Brier | 0.00082405 | 0.00077882 | -0.00004523 |
| Background | ECE | 0.00102952 | 0.00377620 | +0.00274668 |
| Background | MCE | 0.98867913 | 0.97429194 | -0.01438720 |
| Whole image | NLL | 0.04472920 | 0.02340222 | -0.02132698 |
| Whole image | Brier | 0.00588690 | 0.00532238 | -0.00056452 |
| Whole image | ECE | 0.00552318 | 0.00294740 | -0.00257578 |
| Whole image | MCE | 0.44439589 | 0.26107497 | -0.18332093 |

## Brier paired bootstrap

| Region | Point change | 95% interval |
|---|---:|---:|
| Lesion-centered foreground | -0.00885554 | [-0.00951966, -0.00820665] |
| Background | -0.00004523 | [-0.00007525, -0.00001894] |
| Whole image | -0.00056452 | [-0.00062441, -0.00050625] |

## Foreground-region ECE by tumor type

| Type | Raw | OOF calibrated |
|---|---:|---:|
| glioma | 0.13110488 | 0.09463909 |
| meningioma | 0.04434785 | 0.03009554 |
| pituitary | 0.05949439 | 0.04198426 |

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

Foreground diagram: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\segmenter_foreground_reliability_diagram.png`

Whole-image diagram: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\segmenter_whole_image_reliability_diagram.png`

No model was retrained, no test image or test prediction was accessed, the 0.50 threshold was unchanged, and no application integration was performed.
