# Vector scaling calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C01

Five-fold similarity-aware out-of-fold vector scaling will reduce E08 pixel NLL in the lesion-centered, background, and whole-image regions while keeping ECE at or below `0.05` in all three regions. The gate is conjunctive.

For the segmenter, vector scaling is the identifiable binary affine-logit map `sigmoid(a × logit(p) + b)`, with positive `a`; it is fit by whole-image pixel NLL with equal-size 128×128 cases. For the classifier, four-class diagonal scale plus centered bias and binary affine-logit scaling are evaluated descriptively under the same frozen five-fold assignments. Classifier results do not control the segmenter gate and do not reverse the previously accepted raw-classifier decision.

Reported pixel NLL and Brier are image-balanced. ECE/MCE use 15 equal-frequency pooled-pixel bins, recorded now so the later unified-ECE audit can reproduce the same definition. No threshold, model weight, route, or test image is changed.
