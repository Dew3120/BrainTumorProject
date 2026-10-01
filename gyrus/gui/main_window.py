"""Gyrus's 1440×900 Qt shell and Stage 3 empty state."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256

from .splash import ASSET_ROOT
from .state import AppState, ShellStateMachine


class GyrusMainWindow(QMainWindow):
    """Show a verified empty shell; result panels arrive in Stage 4."""

    retry_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.machine = ShellStateMachine()
        self._manifest_sha256: str | None = None
        self._artifact_count: int | None = None
        self._device: str | None = None
        self.setObjectName("GyrusMainWindow")
        self.setWindowTitle("Gyrus — Brain MRI Research Viewer")
        self.setWindowIcon(QIcon(str(ASSET_ROOT / "gyrus.ico")))
        self.resize(1440, 900)
        self.setMinimumSize(1180, 720)
        self._build_shell()
        self._render_state()

    def _build_shell(self) -> None:
        root = QWidget(self)
        root.setObjectName("AppRoot")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.rail = QFrame(root)
        self.rail.setObjectName("NavigationRail")
        self.rail.setFixedWidth(72)
        rail_layout = QVBoxLayout(self.rail)
        rail_layout.setContentsMargins(0, 16, 0, 12)
        rail_layout.setSpacing(8)
        brand = QLabel("G", self.rail)
        brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
        brand.setObjectName("WindowHeading")
        brand.setFixedHeight(44)
        rail_layout.addWidget(brand)
        self.analyze_button = self._nav_button("◉\nAnalyze", "Analyze", True)
        self.history_button = self._nav_button("◷\nHistory", "History", False)
        self.about_button = self._nav_button("ⓘ\nAbout", "About and integrity", True)
        self.analyze_button.setProperty("active", "true")
        self.history_button.setToolTip("History becomes available in Stage 6")
        rail_layout.addWidget(self.analyze_button)
        rail_layout.addWidget(self.history_button)
        rail_layout.addWidget(self.about_button)
        rail_layout.addStretch(1)
        self.about_button.clicked.connect(self._show_integrity_dialog)
        layout.addWidget(self.rail)

        right = QWidget(root)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.header = self._make_header(right)
        right_layout.addWidget(self.header)

        self.notice = QLabel("", right)
        self.notice.setObjectName("Notice")
        self.notice.setAccessibleName("Application notice")
        self.notice.setVisible(False)
        right_layout.addWidget(self.notice)

        self.content_host = QWidget(right)
        self.content_layout = QHBoxLayout(self.content_host)
        self.content_layout.setContentsMargins(24, 24, 24, 32)
        self.content_layout.setSpacing(20)
        self.workspace = self._make_workspace(self.content_host)
        self.inspector = self._make_inspector(self.content_host)
        self.content_layout.addWidget(self.workspace, 1)
        self.content_layout.addWidget(self.inspector)
        right_layout.addWidget(self.content_host, 1)
        self.footer = self._make_footer(right)
        right_layout.addWidget(self.footer)
        layout.addWidget(right, 1)

    def _nav_button(self, text: str, accessible_name: str, enabled: bool) -> QPushButton:
        button = QPushButton(text, self.rail)
        button.setObjectName("NavigationButton")
        button.setAccessibleName(accessible_name)
        button.setEnabled(enabled)
        return button

    def _make_header(self, parent: QWidget) -> QFrame:
        header = QFrame(parent)
        header.setObjectName("Header")
        header.setFixedHeight(64)
        row = QHBoxLayout(header)
        row.setContentsMargins(24, 0, 24, 0)
        row.setSpacing(10)
        self.case_label = QLabel("New analysis", header)
        self.case_label.setObjectName("SectionHeading")
        row.addWidget(self.case_label)
        row.addStretch(1)
        self.privacy_chip = QLabel("◉  LOCAL ONLY", header)
        self.privacy_chip.setObjectName("PrivacyChip")
        self.integrity_chip = QLabel("○  VERIFYING STACK", header)
        self.integrity_chip.setObjectName("IntegrityChip")
        self.device_chip = QLabel("DEVICE PENDING", header)
        self.device_chip.setObjectName("DeviceChip")
        for chip in (self.privacy_chip, self.integrity_chip, self.device_chip):
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            row.addWidget(chip)
        return header

    def _make_workspace(self, parent: QWidget) -> QFrame:
        workspace = QFrame(parent)
        workspace.setObjectName("ImageWorkspace")
        column = QVBoxLayout(workspace)
        column.setContentsMargins(24, 24, 24, 24)
        column.addStretch(1)
        self.drop_zone = QFrame(workspace)
        self.drop_zone.setObjectName("EmptyDropZone")
        self.drop_zone.setFixedSize(440, 260)
        drop_layout = QVBoxLayout(self.drop_zone)
        drop_layout.setContentsMargins(24, 24, 24, 24)
        drop_layout.setSpacing(14)
        drop_layout.addStretch(1)
        drop_title = QLabel("DROP ONE IMAGE", self.drop_zone)
        drop_title.setObjectName("SectionHeading")
        drop_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(drop_title)
        supported = QLabel("One 2D PNG or JPEG from this computer", self.drop_zone)
        supported.setObjectName("Secondary")
        supported.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(supported)
        self.choose_button = QPushButton("Choose image", self.drop_zone)
        self.choose_button.setObjectName("PrimaryAction")
        self.choose_button.setAccessibleName("Choose one local PNG or JPEG image")
        self.choose_button.setEnabled(False)
        self.choose_button.setToolTip("Image selection is added with core panels in Stage 4")
        drop_layout.addWidget(self.choose_button, alignment=Qt.AlignmentFlag.AlignCenter)
        detail = QLabel("Processed on this computer", self.drop_zone)
        detail.setObjectName("Metadata")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(detail)
        drop_layout.addStretch(1)
        column.addWidget(self.drop_zone, alignment=Qt.AlignmentFlag.AlignCenter)
        column.addStretch(1)
        self.empty_label = QLabel("No file selected", workspace)
        self.empty_label.setObjectName("Metadata")
        column.addWidget(self.empty_label)
        self.workspace_message = QLabel("", workspace)
        self.workspace_message.setObjectName("Secondary")
        self.workspace_message.setWordWrap(True)
        self.workspace_message.setVisible(False)
        column.addWidget(self.workspace_message)
        self.retry_button = QPushButton("Retry verification", workspace)
        self.retry_button.setAccessibleName("Retry frozen stack verification")
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self.retry_boot)
        column.addWidget(self.retry_button, alignment=Qt.AlignmentFlag.AlignLeft)
        return workspace

    def _make_inspector(self, parent: QWidget) -> QFrame:
        inspector = QFrame(parent)
        inspector.setObjectName("Inspector")
        inspector.setFixedWidth(480)
        column = QVBoxLayout(inspector)
        column.setContentsMargins(20, 20, 20, 20)
        column.setSpacing(15)
        heading = QLabel("Before you begin", inspector)
        heading.setObjectName("SectionHeading")
        column.addWidget(heading)
        for line in (
            "• One 2D PNG or JPEG",
            "• BRISC-style model categories only",
            "• Mask area reported in pixels",
            "• External-looking inputs require review",
            "• Local history is opt-in",
        ):
            label = QLabel(line, inspector)
            label.setObjectName("Secondary")
            label.setWordWrap(True)
            column.addWidget(label)
        column.addStretch(1)
        self.manifest_label = QLabel("Frozen manifest\nVerification pending", inspector)
        self.manifest_label.setObjectName("Metadata")
        self.manifest_label.setWordWrap(True)
        column.addWidget(self.manifest_label)
        return inspector

    def _make_footer(self, parent: QWidget) -> QFrame:
        footer = QFrame(parent)
        footer.setObjectName("Footer")
        footer.setFixedHeight(32)
        row = QHBoxLayout(footer)
        row.setContentsMargins(24, 0, 24, 0)
        disclaimer = QLabel(DISCLAIMER, footer)
        disclaimer.setObjectName("Metadata")
        row.addWidget(disclaimer)
        row.addStretch(1)
        local = QLabel("No network · Local only", footer)
        local.setObjectName("Metadata")
        row.addWidget(local)
        return footer

    def on_worker_ready(self, snapshot: Any) -> bool:
        if self.machine.state not in {AppState.BOOT_VERIFYING, AppState.BOOT_BLOCKED}:
            raise RuntimeError("worker readiness arrived in an invalid UI state")
        if self.machine.state == AppState.BOOT_BLOCKED:
            self.machine.enter(AppState.BOOT_VERIFYING, reason="explicit verification retry")
        if (
            snapshot.artifact_count != 31
            or snapshot.manifest_sha256 != EXPECTED_MANIFEST_SHA256
            or snapshot.device not in {"cpu", "cuda"}
        ):
            self.machine.enter(AppState.INTEGRITY_ERROR_LOCKED, reason="invalid ready metadata")
            self._render_state()
            return False
        self._manifest_sha256 = snapshot.manifest_sha256
        self._artifact_count = snapshot.artifact_count
        self._device = snapshot.device
        self.machine.enter(AppState.READY_EMPTY, reason="frozen worker verified and loaded")
        self._render_state()
        return True

    def on_worker_boot_failed(self, code: str, _description: str) -> None:
        integrity_prefixes = ("MANIFEST_", "FROZEN_STACK_", "WORKER_STARTUP_PROTOCOL")
        target = (
            AppState.INTEGRITY_ERROR_LOCKED
            if code.startswith(integrity_prefixes)
            else AppState.BOOT_BLOCKED
        )
        self.machine.enter(target, reason=code)
        self._render_state()

    def retry_boot(self) -> None:
        self.machine.enter(AppState.BOOT_VERIFYING, reason="user requested verification retry")
        self._render_state()
        self.retry_requested.emit()

    def show_notice(self, message: str, *, duration_ms: int = 5000) -> None:
        self.notice.setText(message)
        self.notice.setVisible(True)
        QTimer.singleShot(duration_ms, lambda: self.notice.setVisible(False))

    def _render_state(self) -> None:
        state = self.machine.state
        if state == AppState.READY_EMPTY:
            self.integrity_chip.setText("✓  VERIFIED · 31")
            self.integrity_chip.setProperty("state", "verified")
            self.device_chip.setText(
                "GPU · CUDA:0" if self._device == "cuda" else "CPU MODE"
            )
            self.manifest_label.setText(
                f"Frozen manifest\n{self._manifest_sha256[:16]}…"  # type: ignore[index]
            )
            self.drop_zone.setVisible(True)
            self.workspace_message.setVisible(False)
            self.retry_button.setVisible(False)
        elif state in {AppState.BOOT_BLOCKED, AppState.INTEGRITY_ERROR_LOCKED}:
            self.integrity_chip.setText("⚠  VERIFICATION STOPPED")
            self.integrity_chip.setProperty("state", "blocked")
            self.device_chip.setText("DEVICE UNAVAILABLE")
            self.drop_zone.setVisible(False)
            self.workspace_message.setText(
                "Frozen stack integrity verification failed. Analysis is locked."
                if state == AppState.INTEGRITY_ERROR_LOCKED
                else "The worker could not start. Review the local device setting and retry."
            )
            self.workspace_message.setVisible(True)
            self.retry_button.setVisible(True)
            self.manifest_label.setText("Frozen manifest\nNot verified")
        else:
            self.integrity_chip.setText("○  VERIFYING STACK")
            self.integrity_chip.setProperty("state", "verifying")
            self.drop_zone.setVisible(False)
            self.workspace_message.setVisible(False)
            self.retry_button.setVisible(False)
        self.integrity_chip.style().unpolish(self.integrity_chip)
        self.integrity_chip.style().polish(self.integrity_chip)

    def _show_integrity_dialog(self) -> None:
        text = (
            f"Status: {self.machine.state.value}\n"
            f"Artifacts: {self._artifact_count or 'not verified'}\n"
            f"Manifest SHA256: {self._manifest_sha256 or 'not verified'}\n"
            f"Device: {self._device or 'not loaded'}\n"
            f"{DISCLAIMER}"
        )
        QMessageBox.information(self, "Gyrus integrity", text)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.width() < 1280:
            self.inspector.setFixedWidth(412)
            self.content_layout.setContentsMargins(18, 24, 18, 24)
        else:
            self.inspector.setFixedWidth(480)
            self.content_layout.setContentsMargins(24, 24, 24, 32)
