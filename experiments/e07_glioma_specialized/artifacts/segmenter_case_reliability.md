# Segmenter case-level reliability

An L2-regularized logistic regression was evaluated through five-fold similarity-aware cross-fitting. Features use only the raw segmenter probability map and unchanged 0.50 hard mask. Reference masks were used only to create and evaluate the Dice-based target.

## Performance

- Failure-detection AUROC: `0.941025` (95% group-bootstrap interval `[0.918622, 0.958996]`)
- Zero-Dice cases captured in top 20% risk: `10/13`
- All Dice-below-0.50 failures captured in top 20% risk: `70/89`
- Reliability ECE: `0.017103`
- Reliability Brier: `0.062780`
- Fold-prevalence baseline Brier: `0.113945`
- Brier skill: `0.449034` (95% group-bootstrap interval `[0.341619, 0.542478]`)

## Standardized coefficients

Positive coefficients increase predicted reliability; negative coefficients increase predicted failure risk.

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | SD | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty_prediction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |
| predicted_area_fraction | -0.0492 | +0.3517 | +0.0241 | -0.1057 | +0.1689 | +0.0780 | 0.1844 | positive 60% |
| whole_probability_mean | -0.0605 | +0.2226 | -0.0855 | -0.1612 | +0.0439 | -0.0081 | 0.1484 | negative 60% |
| whole_probability_sd | +0.3243 | -0.0841 | +0.3320 | +0.1927 | -0.0299 | +0.1470 | 0.1952 | positive 60% |
| predicted_region_probability_mean | -0.2344 | -0.4643 | -0.2611 | -0.2034 | -0.4611 | -0.3249 | 0.1275 | negative 100% |
| predicted_region_entropy_mean | -1.3721 | -1.1290 | -1.1045 | -1.0708 | -0.9059 | -1.1164 | 0.1674 | negative 100% |
| boundary_entropy_mean | -1.0385 | -1.4812 | -1.4403 | -1.0749 | -1.2353 | -1.2540 | 0.2032 | negative 100% |
| high_confidence_foreground_fraction | -0.2141 | +0.3263 | -0.1418 | +0.3257 | +0.3260 | +0.1244 | 0.2772 | positive 60% |
| uncertain_pixel_fraction | +0.3145 | +0.5059 | +0.5862 | +0.6680 | +0.4781 | +0.5105 | 0.1323 | positive 100% |
| log1p_component_count | +0.0195 | -0.1415 | -0.2532 | -0.2967 | -0.0326 | -0.1409 | 0.1363 | negative 80% |
| largest_component_fraction | -0.1911 | -0.1573 | -0.1453 | -0.0907 | +0.0344 | -0.1100 | 0.0884 | negative 80% |
| bounding_box_fill_ratio | +0.4098 | +0.3878 | +0.1868 | +0.2267 | +0.4314 | +0.3285 | 0.1131 | positive 100% |
| normalized_perimeter | -0.7457 | -0.6197 | -0.7228 | -0.5374 | -0.5289 | -0.6309 | 0.1011 | negative 100% |
| border_touch_fraction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |

## Gate

**PASS**

- `failure_detection_auroc_at_least_0_75`: **PASS**
- `top20_risk_captures_more_than_half_of_zero_dice`: **PASS**
- `reliability_ece_at_most_0_05`: **PASS**
- `brier_skill_positive`: **PASS**
- `brier_skill_bootstrap_ci95_lower_above_zero`: **PASS**

Reliability diagram: `E:\BrainTumorProject\runs\segmentation_v2_glioma_specialized\20260924-105229-241365-specialist\segmenter_case_reliability_diagram.png`

No base model was changed during reliability fitting, no test image or test prediction was accessed, the segmentation threshold remained 0.50, and no application integration was performed.
