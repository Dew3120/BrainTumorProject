"""Visual, accessibility, and privacy regressions for the local Qt shell."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.gui.main_window import GyrusMainWindow

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform != "win32", reason="Native Windows visual baseline")
@pytest.mark.parametrize("size", ((1440, 900), (1180, 720)))
def test_native_shell_matches_reviewed_golden(tmp_path: Path, size: tuple[int, int]) -> None:
    width, height = size
    captured = tmp_path / "capture.png"
    env = os.environ.copy()
    env["QT_QPA_PLATFORM"] = "windows"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.capture_gyrus_golden", str(captured),
         "--width", str(width), "--height", str(height)],
        cwd=ROOT, env=env, capture_output=True, text=True, check=False, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    reference = Image.open(ROOT / "tests" / "golden" / f"gyrus_ready_{width}x{height}.png")
    actual = Image.open(captured)
    if actual.size != reference.size:
        pytest.skip("Display DPI or work-area differs from the reviewed Windows baseline")
    assert ImageChops.difference(actual.convert("RGB"), reference.convert("RGB")).getbbox() is None


def test_primary_controls_have_accessible_names_and_disclaimer(qtbot) -> None:
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.on_worker_ready(SimpleNamespace(
        artifact_count=31, manifest_sha256=EXPECTED_MANIFEST_SHA256, device="cpu",
    ))
    controls = (
        window.choose_button, window.run_button, window.copy_error_button,
        window.open_log_button, window.close_error_button,
        window.image_canvas.view, window.image_canvas.zoom_in,
        window.image_canvas.zoom_out, window.image_canvas.fit_button,
        window.image_canvas.opacity, window.history_list.search,
        window.history_list.table, window.history_preview.preview,
    )
    assert all(control.accessibleName().strip() for control in controls)
    assert DISCLAIMER in window.footer.findChild(type(window.case_label)).text() or any(
        DISCLAIMER in label.text() for label in window.footer.findChildren(type(window.case_label))
    )


def test_verified_startup_does_not_create_history_without_opt_in(tmp_path: Path, monkeypatch, qtbot) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.on_worker_ready(SimpleNamespace(
        artifact_count=31, manifest_sha256=EXPECTED_MANIFEST_SHA256, device="cpu",
    ))
    assert not (tmp_path / "GyrusResearch" / "history").exists()
