import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r04_per_category import CATEGORIES, category_cross_fit, gate_passes


def test_r04_gate_requires_all_six_and_includes_boundary():
    values = {
        f"{version}:{category}": 0.8
        for version in ("A", "B")
        for category in CATEGORIES
    }
    assert gate_passes(values, 0.8)
    values["B:pituitary"] = np.nextafter(0.8, 0.0)
    assert not gate_passes(values, 0.8)


def test_r04_gate_fails_closed_on_missing_category():
    with pytest.raises(ValueError, match="exactly six"):
        gate_passes({"A:glioma": 0.9}, 0.8)


def test_category_cross_fit_rejects_unregistered_category():
    features = np.zeros((10, 2), dtype=np.float64)
    success = np.asarray([False, True] * 5)
    assignment = np.repeat(np.arange(5), 2)
    categories = np.asarray(["glioma"] * 9 + ["no_tumor"])
    with pytest.raises(ValueError, match="unregistered"):
        category_cross_fit(features, success, assignment, categories, ("glioma",))


def test_category_cross_fit_requires_both_training_outcomes():
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    success = np.asarray([False, False, True, True, True, True, True, True, True, True])
    assignment = np.repeat(np.arange(5), 2)
    categories = np.asarray(["glioma"] * 10)
    with pytest.raises(ValueError, match="training partition lacks both outcomes"):
        category_cross_fit(features, success, assignment, categories, ("glioma",))
