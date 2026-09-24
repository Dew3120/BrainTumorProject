import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c02_dirichlet import apply_binary_dirichlet, fit_binary_dirichlet


def test_identity_dirichlet_map_reproduces_probabilities():
    probability = np.array([0.01, 0.2, 0.5, 0.8, 0.99], dtype=np.float64)
    parameters = {
        "foreground_log_coefficient": 1.0,
        "background_log_coefficient": -1.0,
        "intercept": 0.0,
    }
    _calibrated_logit, calibrated = apply_binary_dirichlet(probability, parameters)
    np.testing.assert_allclose(calibrated, probability, atol=1e-12, rtol=1e-12)


def test_fitted_dirichlet_map_reduces_shifted_synthetic_nll():
    probability = np.array([0.05, 0.10, 0.20, 0.75, 0.85, 0.95] * 20)
    target = np.array([0, 0, 0, 1, 1, 1] * 20)
    parameters = fit_binary_dirichlet(probability, target)
    calibrated_logit, _calibrated = apply_binary_dirichlet(probability, parameters)
    raw_nll = -np.mean(target * np.log(probability) + (1 - target) * np.log1p(-probability))
    calibrated_nll = np.mean(
        np.logaddexp(0.0, calibrated_logit) - target * calibrated_logit
    )
    assert calibrated_nll < raw_nll
