import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r05_category_aware import (
    CATEGORIES,
    gate_passes,
    one_hot_categories,
)


def test_r05_gate_includes_exact_boundary():
    assert gate_passes(0.88, 0.88)
    assert not gate_passes(np.nextafter(0.88, 0.0), 0.88)


def test_one_hot_categories_uses_registered_order():
    labels = np.asarray(["glioma", "meningioma", "pituitary"])
    matrix, included = one_hot_categories(labels)
    np.testing.assert_array_equal(matrix, np.eye(3))
    np.testing.assert_array_equal(included, [True, True, True])


def test_one_hot_categories_excludes_no_tumor_as_all_zero():
    labels = np.asarray(["glioma", "no_tumor", "pituitary"])
    matrix, included = one_hot_categories(labels)
    np.testing.assert_array_equal(matrix[1], [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(included, [True, False, True])


def test_registered_category_order_is_fixed():
    assert CATEGORIES == ("glioma", "meningioma", "pituitary")
