import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c03_isotonic import apply_isotonic, fit_isotonic


def test_isotonic_mapping_is_monotone_and_clips_outside_fit_range():
    probability = np.array([0.2, 0.2, 0.4, 0.6, 0.8, 0.8])
    target = np.array([0, 0, 0, 1, 1, 1])
    parameters = fit_isotonic(probability, target)
    _logit, calibrated = apply_isotonic(
        np.array([0.0, 0.2, 0.5, 0.8, 1.0]), parameters
    )
    assert np.all(np.diff(calibrated) >= 0)
    assert calibrated[0] == calibrated[1]
    assert calibrated[-1] == calibrated[-2]


def test_fitted_isotonic_map_reduces_shifted_synthetic_brier():
    probability = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 20)
    target = np.array([0, 0, 0, 1, 1, 1] * 20)
    parameters = fit_isotonic(probability, target)
    _logit, calibrated = apply_isotonic(probability, parameters)
    assert np.mean(np.square(calibrated - target)) < np.mean(
        np.square(probability - target)
    )
