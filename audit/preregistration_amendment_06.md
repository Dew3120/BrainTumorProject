# Pre-registration amendment 06 — exact R03 image-transform dtype

**Status:** REGISTERED BEFORE R03 FEATURE EXTRACTION OR FITTING

**Registered:** 2026-09-27

**Parent R03 registration:** `audit/preregistration_amendment_05.md`, SHA256 `a5c8ff14dea6349c7b3be334e27edf6323f1e0e23a772f530223e89f6bfc03a3`

## Prospective correction

Amendment 05 correctly registered the segmenter's grayscale conversion, aspect-preserving resize, bilinear interpolation, centred black canvas, and division by 255, but incorrectly stated that the canvas is converted to `float64` before division. The frozen segmenter implementation converts the canvas to `float32` before dividing by 255.

R03 therefore uses this exact sequence:

1. Create the registered 128×128 PIL grayscale letterbox canvas.
2. Execute `numpy.asarray(canvas, dtype=numpy.float32) / 255.0`, matching `segmentation_smoke_test.transform_pair` pixel for pixel.
3. After the exact model-space image exists, the inference-only R03 feature function may promote it to `float64` for gradient and summary arithmetic.

The segmenter preprocessing source is `scripts/segmentation_smoke_test.py`, SHA256 `defa28dc8f94dc6fd3c41f5928fba1dbb6959d685acc32972711526ce3341dca`. The R03 feature contract implementation is `scripts/r03_extended_feature_contract.py`, SHA256 `da3b1666adb644dead7f1588b9c8434de2b18732755f470eaf8b3be00943e3b9`.

## Reason and timing

A pre-execution regression test compared the proposed image-only transform against the segmenter's existing `transform_pair` image branch. The Amendment 05 float64 ordering differed in 9,216 of 16,384 synthetic pixels, with maximum absolute difference about `2.97e-8`. No R03 feature had been extracted, no R03 estimator had been fitted, and no R03 outcome had been observed. This amendment corrects the implementation to the user's explicit requirement that preprocessing match exactly; it is not informed by model performance.

The regression test must show exact pixel equality before R03 may run. Every other Amendment 05 feature definition, fold, metric, bootstrap, gate, and fail-closed rule remains unchanged. The AUROC gate remains `0.9595028142589118` and will not be relaxed.
