# Category-aware calibrated-E08 reliability decision

**Decision:** FAIL

The registered gate required classifier-predicted-glioma failure-detection AUROC at least `0.88`. R05 achieved `0.871096` with 95% similarity-group bootstrap interval `[0.817977, 0.920612]`; therefore R05 **failed** its unchanged gate.

Overall AUROC on the 679 included cases was `0.954808`. Predicted-pituitary AUROC was `0.965014`. Full true-category, predicted-category, calibration, Brier, coefficient, and paired comparator results are stored in `result.json` and the artifact tables.

The model used the ten R02 prediction-derived features plus three frozen-classifier-predicted-category indicators. True category, classifier probabilities, and reference-mask values did not enter the feature matrix. The registered predicted-no-tumor case was excluded and received no R05 score. No sealed-test image was accessed, no threshold changed, and no classifier or neural segmenter was retrained.
