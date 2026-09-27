# Calibrated-E08 reliability feature ablation decision

**Decision:** DESCRIPTIVE COMPLETE

R01 has no performance gate and cannot be selected. The full fourteen-feature calibrated-E08 baseline achieved failure-detection AUROC `0.954503` with 95% similarity-group bootstrap CI `[0.937618, 0.969855]`.

Provisional R02 candidates under the user-specified screen (ablation decrease at least `0.005` or identical non-zero coefficient sign across all five folds) are: empty_prediction, predicted_area_fraction, whole_probability_mean, whole_probability_sd, predicted_region_entropy_mean, boundary_entropy_mean, high_confidence_foreground_fraction, uncertain_pixel_fraction, bounding_box_fill_ratio, normalized_perimeter. This screen becomes binding only when R02 is separately registered before fitting.

All features receive only a C03-calibrated probability map and the unchanged `0.50` threshold. Reference masks created targets and evaluation metrics only. No classifier probability, true category, path, metadata, or raw image enters a feature. No neural model was retrained and no sealed-test image was accessed.
