# Stage 6 shared external adapter contract

**Status:** COMPLETE — SYNTHETIC VERIFICATION ONLY

This contract fixes the shared data boundary before any external image is opened. Dataset-specific audits may reject a dataset or add stricter checks, but they may not change these transforms after outcomes are seen.

## Input boundary

Every source adapter must produce a two-dimensional grayscale MRI slice and an aligned binary reference mask in the same source coordinates. NIfTI volumes are converted to closest canonical RAS orientation while retaining the canonical affine and voxel sizes. DICOM slices are ordered from `ImageOrientationPatient` and `ImagePositionPatient`, require one frame of reference, and reject duplicate geometry. RTSTRUCT rasterization remains dataset-specific and must prove frame-of-reference and contour-to-slice alignment before inference.

Native 8-bit slices are preserved. Non-8-bit MRI intensities use a fixed, label-free nonzero 0.5th/99.5th percentile window, clipping, and uint8 conversion. This is an external-format bridge fixed before data access; it does not use masks, outcomes, tumor type, or model predictions. The audit records the numeric window for every slice.

## Frozen preprocessing

- **Segmenter:** grayscale, aspect-preserving resize, Python `round`, bilinear interpolation, centered black `128×128` canvas, divide by 255, output shape `1×128×128`.
- **Classifier:** the same display-ready grayscale slice, aspect-preserving bilinear resize and centered black `224×224` canvas, replicate to RGB, divide by 255, then ImageNet mean/std normalization, output shape `3×224×224`.
- **Reference masks:** binary validation, identical aspect-preserving geometry, nearest-neighbor resize, centered black `128×128` canvas. Tumor-bearing view rejects an empty transformed mask; whole-volume secondary analysis permits registered reference-empty slices.
- **Restoration:** crop only the recorded content rectangle, then restore masks with nearest-neighbor and probabilities with bilinear interpolation. No mask is stretched independently from its image.

## Frozen postprocessing and reliability

Raw E08 probabilities receive C03's committed `final_all_validation_mapping`. The hard mask remains `probability >= 0.50`. The R02 ten features are computed only from the calibrated probability map and its hard prediction. The frozen scaler and logistic artifacts generate `P(Dice >= 0.50)`. Policy status is `STANDARD` only at or above the exact threshold `0.9490250173235835`; otherwise it is `REVIEW_REQUIRED`.

The external model path must route E07 to its glioma specialist from the frozen classifier's predicted category. Historical BRISC path-based routing assignments are development evidence and must never route external cases.

## Geometry and measurement rules

Source affine, orientation, voxel spacing, source shape, canonical shape, slice plane, slice index, letterbox dimensions, offsets, and intensity-window parameters are retained per case. Millimetres, areas, volumes, Hausdorff distance, and depth-like measurements are forbidden when valid spatial metadata is absent. JPEG/PNG-only cohorts remain pixel-measurement cohorts.

## Fail-closed conditions

Stop the active dataset on a frozen-artifact hash mismatch, unknown label semantics, non-finite image, singular affine, image/mask geometry mismatch, ambiguous orientation, invalid or disappearing positive mask, inconsistent DICOM frame of reference, RTSTRUCT alignment failure, threshold change, or output outside its declared range. A failed dataset audit produces no performance panels or metrics.

## Verification completed

Synthetic tests cover landscape and portrait letterboxing, exact uint8 preservation, deterministic non-8-bit windowing, ImageNet classifier normalization, nearest-neighbor masks, empty-mask policy, source-coordinate restoration, RAS NIfTI conversion with voxel spacing, DICOM geometry sorting, C03 mapping, frozen R02 inference, exact artifact hashes, and both locked thresholds. No external image or BRISC sealed-test pixel was accessed.
