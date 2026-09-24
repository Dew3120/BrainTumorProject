# Segmentation failure-mode analysis

This analysis covers the 103 held-out BRISC validation cases with hard-mask Dice below 0.50 at the unchanged probability threshold of 0.50. The selected epoch-45 checkpoint was reloaded, all 680 validation predictions were recomputed exactly, and no test image was accessed.

Foreground pixel counts, overlap, area ratios, edge checks, and connected components are computed on the 128x128 hard masks used by the model. `original_width` and `original_height` preserve each source image's dimensions.

## Verification

- Checkpoint SHA256: `e17e7f5afec1c56e9faf12e12db263eff4a3ebc184cfcd7bdfcfff0c1a977c6a`
- Split snapshot SHA256: `d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28`
- Recomputed validation cases: 680
- Maximum stored-versus-recomputed metric difference: 1.4199930120639692e-08
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
| all | 103 | 0 (0.0%) | 17 (16.5%) | 17 (16.5%) | 16 (15.5%) | 45 (43.7%) | 87 (84.5%) | 81 (78.6%) |
| glioma | 69 | 0 (0.0%) | 9 (13.0%) | 9 (13.0%) | 13 (18.8%) | 26 (37.7%) | 62 (89.9%) | 56 (81.2%) |
| meningioma | 14 | 0 (0.0%) | 4 (28.6%) | 4 (28.6%) | 2 (14.3%) | 7 (50.0%) | 13 (92.9%) | 13 (92.9%) |
| pituitary | 20 | 0 (0.0%) | 4 (20.0%) | 4 (20.0%) | 1 (5.0%) | 12 (60.0%) | 12 (60.0%) | 12 (60.0%) |

The flags above describe independent properties and can overlap. The following table assigns exactly one primary mode to every case, using this precedence: empty, non-empty zero overlap, severe under-prediction, severe over-prediction, fragmented prediction, then other partial overlap.

| Group | Empty | Non-empty zero overlap | Severe under | Severe over | Fragmented | Other partial overlap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0 (0.0%) | 17 (16.5%) | 16 (15.5%) | 45 (43.7%) | 22 (21.4%) | 3 (2.9%) |
| glioma | 0 (0.0%) | 9 (13.0%) | 13 (18.8%) | 26 (37.7%) | 20 (29.0%) | 1 (1.4%) |
| meningioma | 0 (0.0%) | 4 (28.6%) | 2 (14.3%) | 7 (50.0%) | 1 (7.1%) | 0 (0.0%) |
| pituitary | 0 (0.0%) | 4 (20.0%) | 1 (5.0%) | 12 (60.0%) | 1 (5.0%) | 2 (10.0%) |

## Tumor-type hypothesis

Glioma failures are heterogeneous: only 13.0% are complete misses. The visual panel overstated a single dominant failure mode.

Severe over-prediction accounts for 50.0% of meningioma failures and 60.0% of pituitary failures.

## Technical cause versus task difficulty

Only 0.0% of failed references touch a model-space edge. Together with the passed visual audit alignment review, the failure distribution does not indicate a common flip, shift, pairing, or edge-clipping defect.

The dominant cross-category topology signal is fragmentation: 84.5% of all weak cases contain multiple predicted components, and 78.6% contain at least one predicted component with no reference overlap. This is a model-output failure pattern, not evidence of a shared image/mask mapping defect.

These findings quantify model behavior on BRISC's internal validation partition. They do not establish clinical validity, cross-dataset generalization, or verified patient-independent performance.
