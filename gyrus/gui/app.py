"""Executable Qt bootstrap for the local Gyrus application shell."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

from gyrus.domain.constants import APP_NAME, APP_VERSION

from .main_window import GyrusMainWindow
from .splash import GyrusSplash
from .theme import apply_theme
from .work_controller import WorkerBridge

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_application(arguments: list[str] | None = None) -> QApplication:
    """Create the one Qt application with fractional display scaling preserved."""

    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(arguments if arguments is not None else sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("GyrusResearch")
    apply_theme(app)
    return app


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gyrus local research viewer")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--stack-root", type=Path, default=PROJECT_ROOT / "runs" / "final_stack"
    )
    args = parser.parse_args(arguments)
    app = create_application(["Gyrus"])
    splash = GyrusSplash()
    window = GyrusMainWindow()
    bridge = WorkerBridge(args.stack_root, args.device)
    window.retry_requested.connect(bridge.retry_verification)

    def submit_analysis(path: str) -> None:
        try:
            bridge.submit_image(path)
        except Exception as exc:  # noqa: BLE001 - keep invalid jobs out of the UI
            window.on_job_failed(f"WORKER_SUBMISSION_FAILED: {type(exc).__name__}")

    window.analysis_requested.connect(submit_analysis)
    bridge.job_finished.connect(window.on_job_finished)
    bridge.job_failed.connect(window.on_job_failed)

    def reveal() -> None:
        splash.close()
        window.show()

    def on_verified(snapshot: object) -> None:
        if window.on_worker_ready(snapshot):
            splash.mark_verified()
        else:
            splash.mark_blocked("WORKER_STARTUP_PROTOCOL")
        QTimer.singleShot(350, reveal)

    def on_failed(code: str, description: str) -> None:
        window.on_worker_boot_failed(code, description)
        splash.mark_blocked(code)
        QTimer.singleShot(350, reveal)

    bridge.verified.connect(on_verified)
    bridge.boot_failed.connect(on_failed)
    app.aboutToQuit.connect(bridge.close)
    splash.show()
    bridge.start()
    return app.exec()
