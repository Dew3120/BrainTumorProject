from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from finalize_x01_cheng import assert_close


def test_assert_close_accepts_registered_tolerance():
    assert assert_close(1.0, 1.0 + 1e-13, "metric") < 1e-12


def test_assert_close_fails_closed_outside_tolerance():
    with pytest.raises(ValueError, match="Independent recomputation mismatch"):
        assert_close(1.0, 1.0 + 1e-9, "metric")
