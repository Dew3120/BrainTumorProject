import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c04_split_temperature import apply_split_temperature, fit_temperature


def test_split_temperature_routes_regions_independently():
    logits = np.array([[[2.0, 2.0], [-2.0, -2.0]]])
    regions = np.array([[[True, False], [True, False]]])
    transformed, _probability = apply_split_temperature(
        logits,
        regions,
        np.array([0]),
        {"temperature": 2.0},
        {"temperature": 0.5},
    )
    np.testing.assert_allclose(transformed, [[[1.0, 4.0], [-1.0, -4.0]]])


def test_temperature_fit_returns_positive_bounded_value():
    logits = np.array([[[2.0, 1.0], [-1.0, -2.0]]])
    targets = np.array([[[True, True], [False, False]]])
    regions = np.ones_like(targets, dtype=bool)
    fitted = fit_temperature(logits, targets, regions, np.array([0]), "lesion_centered")
    assert 0.05 <= fitted["temperature"] <= 20.0
