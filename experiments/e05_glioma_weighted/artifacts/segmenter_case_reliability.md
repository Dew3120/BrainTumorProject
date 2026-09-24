# Segmenter case-level reliability

An L2-regularized logistic regression was evaluated through five-fold similarity-aware cross-fitting. Features use only the raw segmenter probability map and unchanged 0.50 hard mask. Reference masks were used only to create and evaluate the Dice-based target.

## Performance

- Failure-detection AUROC: `0.942269` (95% group-bootstrap interval `[0.916004, 0.963386]`)
- Zero-Dice cases captured in top 20% risk: `13/17`
- All Dice-below-0.50 failures captured in top 20% risk: `82/103`
- Reliability ECE: `0.015471`
- Reliability Brier: `0.063367`
- Fold-prevalence baseline Brier: `0.128613`
- Brier skill: `0.507309` (95% group-bootstrap interval `[0.409992, 0.598345]`)

## Standardized coefficients

Positive coefficients increase predicted reliability; negative coefficients increase predicted failure risk.

| Feature | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean | SD | Sign agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| empty_prediction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |
| predicted_area_fraction | -0.0278 | +0.3181 | +0.0344 | +0.1906 | +0.2016 | +0.1434 | 0.1390 | positive 80% |
| whole_probability_mean | -0.0886 | +0.1898 | +0.0083 | +0.0505 | +0.0591 | +0.0438 | 0.1005 | positive 80% |
| whole_probability_sd | +0.5649 | +0.5536 | +0.8371 | +0.6632 | +0.7781 | +0.6794 | 0.1263 | positive 100% |
| predicted_region_probability_mean | -0.4610 | -0.9519 | -0.5022 | -0.8481 | -0.7284 | -0.6983 | 0.2136 | negative 100% |
| predicted_region_entropy_mean | -0.7008 | -0.6608 | -0.4104 | -1.0066 | -0.0973 | -0.5752 | 0.3409 | negative 100% |
| boundary_entropy_mean | -0.6939 | -1.6164 | -1.0382 | -1.4442 | -1.3593 | -1.2304 | 0.3661 | negative 100% |
| high_confidence_foreground_fraction | +0.1842 | -0.1351 | +0.1673 | -0.3833 | +0.0254 | -0.0283 | 0.2365 | positive 60% |
| uncertain_pixel_fraction | +0.1046 | +0.1894 | -0.2859 | +0.0530 | -0.3330 | -0.0544 | 0.2385 | positive 60% |
| log1p_component_count | +0.5851 | +0.2964 | +0.4834 | +0.4140 | +0.4213 | +0.4401 | 0.1056 | positive 100% |
| largest_component_fraction | +0.1254 | +0.0612 | +0.1373 | +0.1534 | +0.2547 | +0.1464 | 0.0699 | positive 100% |
| bounding_box_fill_ratio | +0.8783 | +0.5418 | +0.6610 | +0.6691 | +0.6775 | +0.6855 | 0.1212 | positive 100% |
| normalized_perimeter | -0.9873 | -1.1911 | -0.8290 | -0.9499 | -1.0601 | -1.0035 | 0.1342 | negative 100% |
| border_touch_fraction | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | 0.0000 | zero 100% |

## Gate

**PASS**

- `failure_detection_auroc_at_least_0_75`: **PASS**
- `top20_risk_captures_more_than_half_of_zero_dice`: **PASS**
- `reliability_ece_at_most_0_05`: **PASS**
- `brier_skill_positive`: **PASS**
- `brier_skill_bootstrap_ci95_lower_above_zero`: **PASS**

Reliability diagram: `E:\BrainTumorProject\runs\segmentation_v2_weighted\20260924-055049-892543-weighted\segmenter_case_reliability_diagram.png`

No base model was changed during reliability fitting, no test image or test prediction was accessed, the segmentation threshold remained 0.50, and no application integration was performed.
