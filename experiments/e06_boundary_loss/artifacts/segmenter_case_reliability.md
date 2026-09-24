# Segmenter case-level reliability

An L2-regularized logistic regression was evaluated through five-fold similarity-aware cross-fitting. Features use only the raw segmenter probability map and unchanged 0.50 hard mask. Reference masks were used only to create and evaluate the Dice-based target.

## Performance

- Failure-detection AUROC: `0.915253` (95% group-bootstrap interval `[0.891177, 0.938399]`)
- Zero-Dice cases captured in top 20% risk: `11/17`
- All Dice-below-0.50 failures captured in top 20% risk: `64/88`
- Reliability ECE: `0.029545`
- Reliability Brier: `0.074597`
- Fold-prevalence baseline Brier: `0.112693`
- Brier skill: `0.338053` (95% group-bootstrap interval `[0.233022, 0.432959]`)

## Standardized coefficients

Positive coefficients increase predicted reliability; negative coefficients increase predicted failure risk.

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | SD | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty_prediction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |
| predicted_area_fraction | +0.0921 | +0.2360 | +0.0285 | +0.1263 | +0.1798 | +0.1325 | 0.0798 | positive 100% |
| whole_probability_mean | -0.1328 | +0.0372 | -0.2874 | +0.0445 | +0.0209 | -0.0635 | 0.1448 | positive 60% |
| whole_probability_sd | +0.3554 | +0.4238 | +0.7070 | +0.1253 | +0.1093 | +0.3442 | 0.2456 | positive 100% |
| predicted_region_probability_mean | -0.3372 | -1.3093 | -1.1651 | -1.4029 | -1.2686 | -1.0966 | 0.4330 | negative 100% |
| predicted_region_entropy_mean | -1.0001 | -0.9137 | -1.1650 | -1.0017 | -1.2941 | -1.0749 | 0.1526 | negative 100% |
| boundary_entropy_mean | -1.2688 | -1.5704 | -1.5005 | -1.3561 | -1.4008 | -1.4193 | 0.1187 | negative 100% |
| high_confidence_foreground_fraction | -0.1603 | +0.6268 | +0.2129 | +0.6798 | -0.0163 | +0.2686 | 0.3760 | positive 60% |
| uncertain_pixel_fraction | +0.7522 | +0.2508 | +0.5074 | +0.2989 | +0.2996 | +0.4218 | 0.2096 | positive 100% |
| log1p_component_count | -0.4173 | -0.1230 | -0.2550 | -0.3479 | -0.1902 | -0.2667 | 0.1182 | negative 100% |
| largest_component_fraction | -0.1755 | +0.0363 | -0.0256 | -0.0366 | +0.1264 | -0.0150 | 0.1106 | negative 60% |
| bounding_box_fill_ratio | +0.2455 | +0.1561 | -0.0448 | +0.2156 | +0.3153 | +0.1775 | 0.1369 | positive 80% |
| normalized_perimeter | -0.5177 | -0.1052 | -0.4391 | -0.0798 | -0.2208 | -0.2725 | 0.1973 | negative 100% |
| border_touch_fraction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |

## Gate

**PASS**

- `failure_detection_auroc_at_least_0_75`: **PASS**
- `top20_risk_captures_more_than_half_of_zero_dice`: **PASS**
- `reliability_ece_at_most_0_05`: **PASS**
- `brier_skill_positive`: **PASS**
- `brier_skill_bootstrap_ci95_lower_above_zero`: **PASS**

Reliability diagram: `E:\BrainTumorProject\runs\segmentation_v2_boundary_loss\20260924-063051-647842-boundary\segmenter_case_reliability_diagram.png`

No base model was changed during reliability fitting, no test image or test prediction was accessed, the segmentation threshold remained 0.50, and no application integration was performed.
