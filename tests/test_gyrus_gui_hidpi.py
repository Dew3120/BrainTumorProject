"""Exercise the Qt shell at every display scale promised by the design."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("scale", (1.0, 1.25, 1.5, 2.0))
def test_shell_uses_logical_geometry_at_display_scale(scale: float) -> None:
    program = """
from gyrus.gui.app import create_application
from gyrus.gui.main_window import GyrusMainWindow
app = create_application(['Gyrus'])
window = GyrusMainWindow()
window.show()
app.processEvents()
assert abs(window.devicePixelRatioF() - EXPECTED_SCALE) < 1e-6
assert (window.width(), window.height()) == (1440, 900)
assert (window.minimumWidth(), window.minimumHeight()) == (1180, 720)
assert (window.rail.width(), window.header.height(), window.footer.height()) == (72, 64, 32)
window.close()
""".replace("EXPECTED_SCALE", repr(scale))
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR=str(scale))
    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
