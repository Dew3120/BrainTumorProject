# Glioma subgroup analysis

This analysis covers all 204 frozen BRISC glioma validation cases. Failure is Dice below 0.50. Confidence intervals use 2,000 similarity-group bootstrap resamples with seed 42. The success/failure mean-Dice columns are descriptive because those labels are defined by Dice itself. Association is judged by the failure-rate risk-difference CI against the complementary group.

## Reference size

| Group | n | Mean Dice (95% CI) | Failures | Failure mean | Success mean | Failure rate | Risk difference vs others (95% CI) | Odds ratio | Associated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| small | 32 | 0.393 (0.297, 0.483) | 21/32 | 0.223 | 0.718 | 0.656 | +0.430 (+0.258, +0.592) | 6.32 | yes |
| medium | 135 | 0.655 (0.601, 0.702) | 27/135 | 0.182 | 0.774 | 0.200 | -0.278 (-0.407, -0.149) | 0.28 | yes |
| large | 37 | 0.626 (0.540, 0.711) | 12/37 | 0.299 | 0.784 | 0.324 | +0.037 (-0.131, +0.205) | 1.21 | no |

## Boundary class

| Group | n | Mean Dice (95% CI) | Failures | Failure mean | Success mean | Failure rate | Risk difference vs others (95% CI) | Odds ratio | Associated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sharp | 31 | 0.751 (0.646, 0.836) | 5/31 | 0.238 | 0.850 | 0.161 | -0.157 (-0.293, +0.012) | 0.44 | no |
| diffuse | 173 | 0.583 (0.536, 0.629) | 55/173 | 0.218 | 0.754 | 0.318 | +0.157 (-0.012, +0.293) | 2.26 | no |

## Reference focality

| Group | n | Mean Dice (95% CI) | Failures | Failure mean | Success mean | Failure rate | Risk difference vs others (95% CI) | Odds ratio | Associated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| single_focal | 194 | 0.624 (0.579, 0.666) | 54/194 | 0.227 | 0.777 | 0.278 | -0.322 (-0.619, +0.005) | 0.27 | no |
| multi_focal | 10 | 0.325 (0.176, 0.473) | 6/10 | 0.155 | 0.580 | 0.600 | +0.322 (-0.005, +0.619) | 3.72 | no |

## Reference location

| Group | n | Mean Dice (95% CI) | Failures | Failure mean | Success mean | Failure rate | Risk difference vs others (95% CI) | Odds ratio | Associated? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| central | 185 | 0.612 (0.566, 0.657) | 53/185 | 0.222 | 0.769 | 0.286 | -0.082 (-0.313, +0.137) | 0.67 | no |
| peripheral | 19 | 0.575 (0.427, 0.716) | 7/19 | 0.202 | 0.793 | 0.368 | +0.082 (-0.137, +0.313) | 1.49 | no |

## Evidence statement

Characteristics with group-bootstrap risk-difference intervals excluding zero: reference size=small (higher failure risk); reference size=medium (lower failure risk).

These are internal associations on a reused development validation set. They do not prove biological causes, clinical subgroup performance, or patient-independent generalization.