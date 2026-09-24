# Separate-region temperature calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C04

Five-fold similarity-aware scalar temperature fits, estimated separately inside the reference-defined lesion-centred region and its background complement, will achieve lesion-centred ECE at most `0.05` and background ECE at most `0.02` without worsening Brier score in either region. These four checks are conjunctive.

The experiment reports lesion-centred, background, whole-image, and per-category metrics. Because selecting the foreground/background temperature for each pixel requires the reference mask, C04 is an oracle diagnostic. It is not an operational inference method and cannot satisfy H7. Reference masks are used only for fitting and evaluation; no claim is made that they are available at inference.

The hard threshold remains `0.50`. Hard-mask changes are reported descriptively. No neural model is retrained and no sealed-test image is accessed.
