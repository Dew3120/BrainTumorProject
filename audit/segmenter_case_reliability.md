# Segmenter case-level reliability

An L2-regularized logistic regression was evaluated through five-fold similarity-aware cross-fitting. Features use only the raw segmenter probability map and unchanged 0.50 hard mask. Reference masks were used only to create and evaluate the Dice-based target.

## Performance

- Failure-detection AUROC: `0.940990` (95% group-bootstrap interval `[0.918352, 0.961197]`)
- Zero-Dice cases captured in top 20% risk: `14/17`
- All Dice-below-0.50 failures captured in top 20% risk: `68/83`
- Reliability ECE: `0.021171`
- Reliability Brier: `0.056205`
- Fold-prevalence baseline Brier: `0.107273`
- Brier skill: `0.476052` (95% group-bootstrap interval `[0.362997, 0.578839]`)

## Standardized coefficients

Positive coefficients increase predicted reliability; negative coefficients increase predicted failure risk.

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | SD | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty_prediction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |
| predicted_area_fraction | +0.4041 | +0.2874 | +0.5620 | +0.3335 | +0.3025 | +0.3779 | 0.1123 | positive 100% |
| whole_probability_mean | +0.3974 | +0.1195 | +0.4632 | +0.1288 | +0.2319 | +0.2682 | 0.1562 | positive 100% |
| whole_probability_sd | +0.0240 | +0.6593 | +0.3580 | +0.3211 | +0.3708 | +0.3467 | 0.2254 | positive 100% |
| predicted_region_probability_mean | +0.0890 | -0.1225 | +0.0933 | +0.0886 | -0.0613 | +0.0174 | 0.1021 | positive 60% |
| predicted_region_entropy_mean | -0.5149 | -0.3243 | -0.3502 | -0.8727 | -0.4081 | -0.4940 | 0.2240 | negative 100% |
| boundary_entropy_mean | -0.8621 | -1.3737 | -1.0464 | -0.9397 | -0.9670 | -1.0378 | 0.1990 | negative 100% |
| high_confidence_foreground_fraction | -0.0488 | -0.1038 | -0.0980 | -0.1070 | -0.0736 | -0.0863 | 0.0247 | negative 100% |
| uncertain_pixel_fraction | +0.0618 | +0.0951 | -0.2600 | +0.3622 | +0.0457 | +0.0610 | 0.2209 | positive 80% |
| log1p_component_count | +0.5041 | +0.2526 | +0.2890 | +0.2088 | +0.0548 | +0.2618 | 0.1622 | positive 100% |
| largest_component_fraction | +0.3570 | +0.3562 | +0.0786 | +0.0710 | +0.2764 | +0.2278 | 0.1435 | positive 100% |
| bounding_box_fill_ratio | +0.6001 | +0.3928 | +0.4887 | +0.5648 | +0.4158 | +0.4924 | 0.0903 | positive 100% |
| normalized_perimeter | -0.5166 | -0.4238 | -0.6257 | -0.4574 | -0.5376 | -0.5122 | 0.0780 | negative 100% |
| border_touch_fraction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |

## Gate

**PASS**

- `failure_detection_auroc_at_least_0_75`: **PASS**
- `top20_risk_captures_at_least_9_of_17_zero_dice`: **PASS**
- `reliability_ece_at_most_0_05`: **PASS**
- `brier_skill_positive`: **PASS**
- `brier_skill_bootstrap_ci95_lower_above_zero`: **PASS**

Reliability diagram: `E:\BrainTumorProject\runs\segmentation\20260923-072439-397263-full\segmenter_case_reliability_diagram.png`

No base model was retrained, no test image or test prediction was accessed, the segmentation threshold remained 0.50, and no application integration was performed.
