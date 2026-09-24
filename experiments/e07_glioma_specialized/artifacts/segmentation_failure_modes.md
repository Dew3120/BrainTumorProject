# Segmentation failure-mode analysis

This analysis covers the 89 held-out BRISC validation cases with hard-mask Dice below 0.50 at the unchanged probability threshold of 0.50. The selected epoch-31 checkpoint was reloaded, all 680 validation predictions were recomputed exactly, and no test image was accessed.

Foreground pixel counts, overlap, area ratios, edge checks, and connected components are computed on the 128x128 hard masks used by the model. `original_width` and `original_height` preserve each source image's dimensions.

## Verification

- Checkpoint SHA256: `bb68ce4f0a9203d74a5ec9dea2e83468d3d833ac285a13ebbdb1d793a0d3bbe0`
- Split snapshot SHA256: `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28`
- Recomputed validation cases: 680
- Maximum stored-versus-recomputed metric difference: 0.0
- Threshold: 0.50
- Test images accessed: 0

## Definitions

- **Empty prediction:** predicted foreground contains zero pixels.
- **Non-empty zero overlap:** predicted foreground exists but has no intersection with the reference.
- **Complete miss:** empty prediction or non-empty zero overlap.
- **Severe under-prediction:** overlap exists and prediction/reference area ratio is below 0.5.
- **Severe over-prediction:** overlap exists and prediction/reference area ratio is above 2.0.
- **Fragmented:** prediction has more than one 8-connected component.
- **Disconnected false positive:** prediction has multiple components and at least one component has no reference overlap.

Except for the mutually exclusive empty and non-empty-zero-overlap flags, diagnostic flags may overlap and therefore do not sum to 100%.

## Breakdown

| Group | n | Empty | Non-empty zero overlap | Complete miss | Severe under | Severe over | Fragmented | Disconnected false positive |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 89 | 0 (0.0%) | 13 (14.6%) | 13 (14.6%) | 9 (10.1%) | 45 (50.6%) | 68 (76.4%) | 68 (76.4%) |
| glioma | 65 | 0 (0.0%) | 8 (12.3%) | 8 (12.3%) | 6 (9.2%) | 35 (53.8%) | 47 (72.3%) | 47 (72.3%) |
| meningioma | 9 | 0 (0.0%) | 2 (22.2%) | 2 (22.2%) | 0 (0.0%) | 6 (66.7%) | 9 (100.0%) | 9 (100.0%) |
| pituitary | 15 | 0 (0.0%) | 3 (20.0%) | 3 (20.0%) | 3 (20.0%) | 4 (26.7%) | 12 (80.0%) | 12 (80.0%) |

The flags above describe independent properties and can overlap. The following table assigns exactly one primary mode to every case, using this precedence: empty, non-empty zero overlap, severe under-prediction, severe over-prediction, fragmented prediction, then other partial overlap.

| Group | Empty | Non-empty zero overlap | Severe under | Severe over | Fragmented | Other partial overlap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0 (0.0%) | 13 (14.6%) | 9 (10.1%) | 45 (50.6%) | 20 (22.5%) | 2 (2.2%) |
| glioma | 0 (0.0%) | 8 (12.3%) | 6 (9.2%) | 35 (53.8%) | 14 (21.5%) | 2 (3.1%) |
| meningioma | 0 (0.0%) | 2 (22.2%) | 0 (0.0%) | 6 (66.7%) | 1 (11.1%) | 0 (0.0%) |
| pituitary | 0 (0.0%) | 3 (20.0%) | 3 (20.0%) | 4 (26.7%) | 5 (33.3%) | 0 (0.0%) |

## Tumor-type hypothesis

Glioma failures are heterogeneous: only 12.3% are complete misses. The visual panel overstated a single dominant failure mode.

Severe over-prediction accounts for 66.7% of meningioma failures and 26.7% of pituitary failures.

## Technical cause versus task difficulty

Only 0.0% of failed references touch a model-space edge. Together with the passed visual audit alignment review, the failure distribution does not indicate a common flip, shift, pairing, or edge-clipping defect.

The dominant cross-category topology signal is fragmentation: 76.4% of all weak cases contain multiple predicted components, and 76.4% contain at least one predicted component with no reference overlap. This is a model-output failure pattern, not evidence of a shared image/mask mapping defect.

These findings quantify model behavior on BRISC's internal validation partition. They do not establish clinical validity, cross-dataset generalization, or verified patient-independent performance.
