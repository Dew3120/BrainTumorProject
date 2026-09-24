import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_c06_per_category import fit_temperature


def test_per_category_temperature_is_positive_and_bounded():
    logits = np.array([[[2.0, 1.0], [-1.0, -2.0]], [[1.0, 0.5], [-0.5, -1.0]]])
    targets = np.array([[[True, True], [False, False]], [[True, True], [False, False]]])
    fitted = fit_temperature(logits, targets, np.array([0, 1]))
    assert 0.05 <= fitted["temperature"] <= 20.0
