import evaluate_morphology as morphology
import numpy as np


def test_closing_then_opening_fills_hole_and_removes_isolated_noise():
    mask = np.zeros((11, 11), dtype=bool)
    mask[3:8, 3:8] = True
    mask[5, 5] = False
    mask[1, 1] = True
    transformed = morphology.closing_then_opening(mask, 3)
    assert transformed[5, 5]
    assert not transformed[1, 1]
    assert transformed.dtype == bool
