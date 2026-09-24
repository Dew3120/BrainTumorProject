# Classifier-routed per-category pixel calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C06

Three positive scalar temperatures, fitted and routed by the frozen classifier's predicted tumor category, will achieve ECE at most `0.05` in each of the lesion-centred, background, and whole-image regions within every predicted tumor category. These nine checks are conjunctive.

The router uses classifier prediction only, never true tumor type. The single segmentation-validation case predicted as `no_tumor` uses a preregistered global fallback temperature because a one-case route cannot support a stable category fit or gate; its result is reported separately. Every transform is fitted within the four calibration training folds and evaluated on the fifth. The final all-validation parameters are operational proposals only.

NLL, Brier, MCE, hard-mask changes, true-label breakdowns, and the no-tumor fallback are descriptive. The threshold remains `0.50`. No neural model is retrained and no sealed-test image is accessed.
