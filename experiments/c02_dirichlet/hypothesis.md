# Two-class Dirichlet pixel calibration

**Status:** REGISTERED — NOT STARTED

**Hypothesis:** Calibration experiment C02

Five-fold similarity-aware out-of-fold binary Dirichlet calibration will reduce E08 pixel NLL in every fixed evaluation region, keep ECE at or below `0.05` in every region, and leave every hard mask bitwise identical when both raw and calibrated probabilities are thresholded at `0.50`. The gate is conjunctive.

The identifiable binary Dirichlet map is `sigmoid(a × log(p_fg) + b × log(p_bg) + c)`. It is the two-class form of linear Dirichlet calibration on log class probabilities. Parameters are fitted by image-balanced whole-image pixel NLL inside four folds and applied only to the held-out fifth fold. The five held-out outputs provide all reported metrics. A final all-validation fit is operationally proposed only and is not used for evaluation.

Evaluation uses the registered lesion-centered, background, and whole-image regions. NLL and Brier are image-balanced; ECE and MCE use 15 equal-frequency pooled-pixel bins. The fixed threshold remains `0.50`. Any changed hard-mask pixel rejects C02 even if probability metrics improve. No model weight, route, or sealed-test image is used or changed.
