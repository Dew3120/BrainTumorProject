# Isotonic pixel calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C03

A deterministic five-fold similarity-aware isotonic mapping fitted to E08 training-fold pixel probabilities will keep ECE at or below `0.04` and will not worsen Brier score in each of the lesion-centred, background, and whole-image regions. All six checks are conjunctive.

The monotone piecewise-constant mapping is fitted by the pool-adjacent-violators algorithm on all training-fold pixels. Identical stored float32 probabilities are deterministically aggregated by their target mean and count before fitting; this is an exact sufficient-statistic compression, not sampling. Held-out probabilities outside the fitted range are clipped to the nearest endpoint. Reported metrics use only out-of-fold mappings.

Reference masks provide supervised calibration targets and define evaluation regions during development. At inference, the mapping receives only the raw segmenter probability. Hard masks are thresholded at the unchanged `0.50`, compared bitwise with raw masks, and all changes are reported; hard-mask identity is descriptive for C03 rather than a gate condition. No neural model is retrained and no sealed-test image is accessed.
