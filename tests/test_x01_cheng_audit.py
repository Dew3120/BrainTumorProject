from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_x01_cheng import (
    boundary_alignment_fraction,
    minmax_uint8,
    phash_distance,
)


def test_minmax_uint8_spans_full_display_range():
    value = np.asarray([[2, 3], [4, 6]], dtype=np.int16)
    result = minmax_uint8(value)
    assert result.dtype == np.uint8
    assert int(result.min()) == 0
    assert int(result.max()) == 255


def test_phash_distance_is_hamming_distance():
    assert phash_distance("0000000000000000", "0000000000000000") == 0
    assert phash_distance("0000000000000000", "ffffffffffffffff") == 64
    assert phash_distance("0000000000000001", "0000000000000003") == 1


def test_boundary_alignment_uses_matlab_xy_coordinates():
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:8, 5:10] = 1
    # One-based MATLAB (x, y) pairs around the mask boundary.
    border = np.asarray([[6, 4], [10, 4], [10, 8], [6, 8]], dtype=float)
    assert boundary_alignment_fraction(mask, border) == 1.0
