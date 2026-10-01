import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QMovie

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.gui.app import create_application
from gyrus.gui.main_window import GyrusMainWindow
from gyrus.gui.splash import GyrusSplash
from gyrus.gui.state import AppState
from gyrus.gui.theme import COLORS

ROOT = Path(__file__).resolve().parents[1]


def _ready(device="cuda", sha=EXPECTED_MANIFEST_SHA256):
    return SimpleNamespace(
        artifact_count=31,
        manifest_sha256=sha,
        device=device,
    )


def test_shell_geometry_and_required_information(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(40)
    window.on_worker_ready(_ready())
    assert window.size().width() == 1440
    assert window.size().height() == 900
    assert window.minimumWidth() == 1180
    assert window.minimumHeight() == 720
    assert window.rail.width() == 72
    assert window.header.height() == 64
    assert window.footer.height() == 32
    assert window.inspector.width() == 480
    assert window.machine.state == AppState.READY_EMPTY
    assert "VERIFIED" in window.integrity_chip.text()
    assert "CUDA:0" in window.device_chip.text()
    assert any(
        DISCLAIMER in label.text() for label in window.footer.findChildren(type(window.case_label))
    )
    assert window.history_button.isEnabled() is False
    assert window.choose_button.isEnabled() is True
    assert window.privacy_chip.text().endswith("LOCAL ONLY")
    assert window.notice.isVisible() is False


def test_minimum_window_keeps_inspector_docked_and_workspace_visible(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.resize(1180, 720)
    window.show()
    qtbot.wait(40)
    window.on_worker_ready(_ready(device="cpu"))
    assert window.rail.width() == 72
    assert window.inspector.width() == 412
    assert window.workspace.width() >= 620
    assert window.device_chip.text() == "CPU MODE"
    assert window.drop_zone.isVisible()


def test_invalid_ready_metadata_locks_result_display(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.on_worker_ready(_ready(sha="0" * 64)) is False
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert not window.drop_zone.isVisible()
    assert not window.retry_button.isVisible()
    assert "VERIFICATION STOPPED" in window.integrity_chip.text()


def test_splash_uses_approved_animation_without_early_checkmarks(qtbot):
    splash = GyrusSplash()
    qtbot.addWidget(splash)
    splash.show()
    assert splash.size().width() == 800
    assert splash.size().height() == 500
    assert splash.movie.isValid()
    assert splash.movie.state() == QMovie.MovieState.Running
    assert "pending" in splash.manifest_line.text()
    assert "verified" not in splash.manifest_line.text().lower()
    splash.mark_verified()
    assert "✓" in splash.manifest_line.text()
    splash.close()


def test_palette_and_fractional_dpi_policy(qapp):
    create_application(["Gyrus"])
    assert COLORS["background"] == "#0B1016"
    assert COLORS["primary"] == "#4EA8DE"
    assert QGuiApplication.highDpiScaleFactorRoundingPolicy() == (
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
