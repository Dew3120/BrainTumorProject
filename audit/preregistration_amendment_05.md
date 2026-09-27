# Pre-registration amendment 05 — R03 extended reliability model

**Status:** REGISTERED BEFORE R03 FEATURE EXTRACTION OR FITTING

**Registered:** 2026-09-27

**Parent registration:** `audit/preregistration.md`, SHA256 `7293d48c4eaa683a25a060681e44ec6838203703e1e588da439b7470ecf88fda`

**Stage 4 parent amendment:** `audit/preregistration_amendment_03.md`, SHA256 `0235d8553a3a4c67a1e1ee293679f53c4984c72a102f7ed84fd8acef561f0a40`

**R01 source result:** `experiments/r01_ablation/result.json`, SHA256 `d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d`

**R01 experiment manifest:** `experiments/r01_ablation/artifacts/experiment_manifest.json`, SHA256 `7e7162d4d1b33349c7944a8d8045898dbe37966d2c407b5feec556dbcfa51a7a`

## Prospective R03 hypothesis

Adding six prespecified inference-time morphology, geometry, texture-complexity, and image-boundary features to the locked fourteen-feature reliability model will improve five-fold out-of-fold failure-detection AUROC by at least `0.005` on the same 680 C03-calibrated internal-validation cases.

The locked baseline is `0.9545028142589118`. R03 passes only at extended-model AUROC `0.9595028142589118` or higher. Equality passes. The hypothesis is falsified below this value.

## Locked feature set

The first fourteen inputs are the unchanged R01 features, in their committed order. R03 appends exactly these six features:

1. `mask_circularity`
2. `mask_solidity`
3. `boundary_fractal_dimension`
4. `predicted_boundary_image_gradient_mean`
5. `predicted_centroid_to_image_center_distance`
6. `predicted_vertical_symmetry`

No feature may be added, removed, renamed, substituted, or redefined after R03 results are observed.

## Exact feature definitions

Let `P` be the C03 out-of-fold probability map, let `M = P >= 0.50`, let `A` be the number of foreground pixels in `M`, and let `B = M AND NOT binary_erosion(M)` using one iteration, a full `3×3` structure, and zero-valued image borders. All arrays are `128×128`.

1. **Mask circularity:** `4πA / perimeter(M)^2`. Perimeter is `skimage.measure.perimeter(M, neighborhood=8)`. Return `0` when `A=0` or perimeter is zero. Do not clip the value.
2. **Solidity:** `A / convex_hull_area`, where the hull is `skimage.morphology.convex_hull_image(M, offset_coordinates=True, tolerance=1e-10, include_borders=True)`. Return `0` when `A=0` or hull area is zero.
3. **Boundary fractal dimension:** box-count `B` at box widths `2, 4, 8, 16, 32`. For each width, count non-overlapping boxes containing at least one boundary pixel. Fit ordinary least squares to `log(count)` versus `log(128 / box_width)` and use the slope. Return `0` when fewer than two scales have positive counts or `M` is empty.
4. **Local intensity gradient at predicted boundary:** transform the uploaded source image to the exact model-space image described below. Compute `numpy.gradient` with unit spacing and default edge handling, then gradient magnitude `sqrt(gx^2 + gy^2)`. Report its arithmetic mean over `B`. Return `0` when `B` is empty. This feature uses the source image and prediction, never the reference mask.
5. **Centroid distance to image centre:** compute the predicted-mask centroid `(mean_y, mean_x)`, its Euclidean distance to `(63.5, 63.5)`, and divide by `sqrt(2) × 63.5`. Return `0` when `M` is empty.
6. **Vertical symmetry:** reflect `M` left-to-right and compute `|M intersection flip(M)| / A`, equivalent to Dice between equal-area `M` and its vertical-axis reflection. Return `0` when `M` is empty.

## Locked source-image transform

Only feature 4 reads the input image. The image is opened from the case path after SHA256 verification against the frozen split snapshot. Convert with PIL to mode `L`; preserve aspect ratio; scale by `min(128 / width, 128 / height)`; round each resized dimension with Python `round`; resize using `PIL.Image.Resampling.BILINEAR`; centre the result on a black `128×128` canvas using integer floor offsets; convert to `float64`; divide by `255`. This reproduces the segmenter's inference-time image geometry without loading or consulting a reference mask.

The other five new features use `M` only. All six are available from an uploaded image and model prediction at inference. True label, reference mask, Dice, patient/group identity, path text, dataset metadata, and classifier probabilities are prohibited inputs.

## Locked protocol

- Population: the same 680 BRISC internal segmentation-validation cases used by R01 and R02.
- Calibration: each case's locked C03 five-fold out-of-fold isotonic mapping.
- Threshold: exactly `0.50`.
- Folds: the same five similarity-aware folds, SHA256 `48802b704465c2807ae0752e0a62b26cc95f4c40dcdbf2db69c1194ac2f31bf4`.
- Models: reproduce the fourteen-feature baseline and fit the twenty-feature extended model with `StandardScaler` plus L2 logistic regression (`C=1`, `lbfgs`, no class weighting, maximum 5,000 iterations, seed 42) independently inside each outer training fold.
- Reported predictions: out-of-fold only.
- Bootstrap: 2,000 paired similarity-group resamples, seed 42, percentile 95% intervals.
- Calibration: 15 equal-frequency bins with stable score ordering.
- Catastrophic capture: zero-Dice cases in the top 20% highest failure risk, with normalized path as deterministic tie-breaker.
- Subgroups: true-category and classifier-predicted-category AUROCs are descriptive only.

## New-feature attribution

Each of the six new features receives a leave-one-new-feature-out refit of the twenty-feature model. Ablation importance is `extended AUROC - ablated AUROC`.

Permutation importance uses the already fitted twenty-feature fold models. Within each held-out fold, permute one new feature at a time for 30 deterministic repetitions. Seed is `42 + 1000 × new_feature_index + repetition_index`. Importance is `extended AUROC - mean permuted AUROC`. Its 95% interval uses the same 2,000 paired similarity-group bootstrap samples, averaging all 30 repetitions inside each sample.

## Gate and fail-closed decision

R03 passes only when:

`extended OOF AUROC - 0.9545028142589118 >= 0.005`

which is equivalent to:

`extended OOF AUROC >= 0.9595028142589118`

R03 stops without an accepted result if any source hash, raw-image hash, population, feature name/order/formula, C03 transformation, fold, group, baseline OOF vector, baseline AUROC, finiteness, threshold, sealed-test, or retraining verification fails. The R01 baseline must reproduce within `1e-12`, including per-case OOF scores. Reference masks create targets and evaluation metrics only. Reliability-estimator fitting is authorized; classifier or neural segmenter retraining is prohibited. Gates will not be relaxed after results are seen.
