import finalize_e06_boundary_loss as finalize
import numpy as np


def test_boundary_f1_is_one_for_identical_nonempty_masks():
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:10, 5:12] = True
    assert finalize.boundary_f1(mask, mask) == 1.0


def test_boundary_f1_tolerates_two_pixel_translation():
    reference = np.zeros((16, 16), dtype=bool)
    prediction = np.zeros((16, 16), dtype=bool)
    reference[4:9, 4:9] = True
    prediction[4:9, 6:11] = True
    assert finalize.boundary_f1(prediction, reference, tolerance=2) == 1.0


def test_boundary_f1_is_zero_for_disjoint_masks():
    reference = np.zeros((20, 20), dtype=bool)
    prediction = np.zeros((20, 20), dtype=bool)
    reference[2:5, 2:5] = True
    prediction[14:17, 14:17] = True
    assert finalize.boundary_f1(prediction, reference, tolerance=2) == 0.0
