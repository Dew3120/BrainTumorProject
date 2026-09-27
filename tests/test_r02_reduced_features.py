import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_r02_reduced_features import candidate_indices, gate_passes


def test_r02_gate_includes_exact_boundary():
    assert gate_passes(0.9495028142589118, 0.9495028142589118)
    assert not gate_passes(0.9495028142589117, 0.9495028142589118)


def test_candidate_indices_preserve_registered_order():
    all_features = tuple("abcdefghijklmn")
    candidates = ("j", "b", "m", "d", "a", "i", "f", "c", "n", "h")
    np.testing.assert_array_equal(
        candidate_indices(all_features, candidates, candidates),
        [9, 1, 12, 3, 0, 8, 5, 2, 13, 7],
    )


def test_candidate_indices_fail_closed_on_r01_mismatch():
    with pytest.raises(ValueError, match="committed R01"):
        candidate_indices(("a",) * 10, ("a",) * 10, ("b",) * 10)
