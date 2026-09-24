# Segmentation failure-mode analysis

This analysis covers the 88 held-out BRISC validation cases with hard-mask Dice below 0.50 at the unchanged probability threshold of 0.50. The selected epoch-40 checkpoint was reloaded, all 680 validation predictions were recomputed exactly, and no test image was accessed.

Foreground pixel counts, overlap, area ratios, edge checks, and connected components are computed on the 128x128 hard masks used by the model. `original_width` and `original_height` preserve each source image's dimensions.

## Verification

- Checkpoint SHA256: `b98afabc8118471e01f560ff6a727ce25c08001574bf12005d77d1be773f43ab`
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
| all | 88 | 0 (0.0%) | 17 (19.3%) | 17 (19.3%) | 12 (13.6%) | 32 (36.4%) | 66 (75.0%) | 62 (70.5%) |
| glioma | 69 | 0 (0.0%) | 15 (21.7%) | 15 (21.7%) | 11 (15.9%) | 21 (30.4%) | 52 (75.4%) | 48 (69.6%) |
| meningioma | 5 | 0 (0.0%) | 1 (20.0%) | 1 (20.0%) | 0 (0.0%) | 3 (60.0%) | 5 (100.0%) | 5 (100.0%) |
| pituitary | 14 | 0 (0.0%) | 1 (7.1%) | 1 (7.1%) | 1 (7.1%) | 8 (57.1%) | 9 (64.3%) | 9 (64.3%) |

The flags above describe independent properties and can overlap. The following table assigns exactly one primary mode to every case, using this precedence: empty, non-empty zero overlap, severe under-prediction, severe over-prediction, fragmented prediction, then other partial overlap.

| Group | Empty | Non-empty zero overlap | Severe under | Severe over | Fragmented | Other partial overlap |
|---|---:|---:|---:|---:|---:|---:|
| all | 0 (0.0%) | 17 (19.3%) | 12 (13.6%) | 32 (36.4%) | 23 (26.1%) | 4 (4.5%) |
| glioma | 0 (0.0%) | 15 (21.7%) | 11 (15.9%) | 21 (30.4%) | 20 (29.0%) | 2 (2.9%) |
| meningioma | 0 (0.0%) | 1 (20.0%) | 0 (0.0%) | 3 (60.0%) | 1 (20.0%) | 0 (0.0%) |
| pituitary | 0 (0.0%) | 1 (7.1%) | 1 (7.1%) | 8 (57.1%) | 2 (14.3%) | 2 (14.3%) |

## Tumor-type hypothesis

Glioma failures are heterogeneous: only 21.7% are complete misses. The visual panel overstated a single dominant failure mode.

Severe over-prediction accounts for 60.0% of meningioma failures and 57.1% of pituitary failures.

## Technical cause versus task difficulty

Only 0.0% of failed references touch a model-space edge. Together with the passed visual audit alignment review, the failure distribution does not indicate a common flip, shift, pairing, or edge-clipping defect.

The dominant cross-category topology signal is fragmentation: 75.0% of all weak cases contain multiple predicted components, and 70.5% contain at least one predicted component with no reference overlap. This is a model-output failure pattern, not evidence of a shared image/mask mapping defect.

These findings quantify model behavior on BRISC's internal validation partition. They do not establish clinical validity, cross-dataset generalization, or verified patient-independent performance.
