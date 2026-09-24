import advanced_calibration_common as common
import numpy as np


def test_equal_frequency_binary_is_zero_for_perfect_two_bin_predictions():
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])
    targets = np.array([0, 0, 1, 1])
    ece, mce, bins = common.equal_frequency_binary(probabilities, targets, bins=2)
    assert ece == 0.0
    assert mce == 0.0
    assert len(bins) == 2


def test_group_bootstrap_zero_change_has_zero_interval():
    values = np.array([0.1, 0.2, 0.3])
    result = common.paired_group_bootstrap(values, values, ["a", "b", "c"], 50)
    assert result == {"mean_change": 0.0, "ci95_lower": 0.0, "ci95_upper": 0.0}
