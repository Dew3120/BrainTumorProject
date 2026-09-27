import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c03_isotonic import apply_isotonic


def test_c03_mapping_used_by_diagrams_is_probability_only():
    mapping = {
        "x_thresholds": [0.0, 0.5, 1.0],
        "y_thresholds": [0.1, 0.4, 0.9],
    }
    _logits, calibrated = apply_isotonic(np.array([0.25, 0.75]), mapping)
    np.testing.assert_allclose(calibrated, [0.25, 0.65])
