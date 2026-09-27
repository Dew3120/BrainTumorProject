import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r01_ablation import auc_many
from stage4_reliability_common import verify_feature_contract


def test_vectorized_auc_matches_sklearn_with_ties():
    failure = np.array([False, True, False, True, True, False])
    scores = np.array(
        [
            [0.1, 0.8, 0.2, 0.8, 0.9, 0.1],
            [0.2, 0.7, 0.2, 0.6, 0.6, 0.1],
        ]
    )
    expected = np.array([roc_auc_score(failure, row) for row in scores])
    np.testing.assert_allclose(auc_many(scores, failure), expected)


def test_feature_contract_accepts_only_probability_and_threshold():
    features = (
        "empty_prediction",
        "predicted_area_fraction",
        "whole_probability_mean",
        "whole_probability_sd",
        "predicted_region_probability_mean",
        "predicted_region_entropy_mean",
        "boundary_entropy_mean",
        "high_confidence_foreground_fraction",
        "uncertain_pixel_fraction",
        "log1p_component_count",
        "largest_component_fraction",
        "bounding_box_fill_ratio",
        "normalized_perimeter",
        "border_touch_fraction",
    )
    contract = verify_feature_contract(features)
    assert contract["function_parameters"] == ["probability", "threshold"]
    assert contract["reference_mask_feature_input"] is False
    assert contract["classifier_probability_feature_input"] is False
