import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c05_reliability_weighted import fit_weighted_temperature


def test_weighted_temperature_is_positive_and_bounded():
    logits = np.array([[[2.0, 1.0], [-1.0, -2.0]], [[1.0, 0.5], [-0.5, -1.0]]])
    targets = np.array([[[True, True], [False, False]], [[True, True], [False, False]]])
    reliability = np.array([0.9, 0.2])
    fitted = fit_weighted_temperature(
        logits, targets, np.array([0, 1]), reliability
    )
    assert 0.05 <= fitted["temperature"] <= 20.0
    assert fitted["minimum_reliability_weight"] == 0.2
    assert fitted["maximum_reliability_weight"] == 0.9
