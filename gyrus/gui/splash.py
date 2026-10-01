"""Animated 800×500 startup view with truthful verification progress."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from gyrus.domain.constants import APP_VERSION

from .theme import COLORS

ASSET_ROOT = Path(__file__).resolve().parents[2] / "assets" / "logo"


class GyrusSplash(QWidget):
    """Keep all check marks inactive until a verified worker says they passed."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.Dialog)
        self.setObjectName("GyrusSplash")
        self.setFixedSize(800, 500)
        self.setWindowTitle("Gyrus — loading")
        self.setStyleSheet(f"background: {COLORS['background']};")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 26)
        card = QFrame(self)
        card.setObjectName("SplashCard")
        outer.addWidget(card)

        row = QHBoxLayout(card)
        row.setContentsMargins(42, 42, 42, 42)
        row.setSpacing(26)
        self.logo = QLabel(card)
        self.logo.setObjectName("SplashAnimation")
        self.logo.setFixedSize(290, 290)
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo.setAccessibleName("Animated six-panel Gyrus mark")
        self.movie = QMovie(str(ASSET_ROOT / "gyrus_loading_6panel_256.gif"))
        if self.movie.isValid():
            self.movie.setScaledSize(QSize(256, 256))
            self.logo.setMovie(self.movie)
            self.movie.start()
        else:
            self.logo.setPixmap(QPixmap(str(ASSET_ROOT / "gyrus_mark_256.png")))
        row.addWidget(self.logo, 0)

        content = QVBoxLayout()
        content.setSpacing(10)
        content.addStretch(1)
        title = QLabel("GYRUS", card)
        title.setObjectName("WindowHeading")
        content.addWidget(title)
        subtitle = QLabel("Brain MRI Research Viewer", card)
        subtitle.setObjectName("Secondary")
        content.addWidget(subtitle)
        version = QLabel(f"v{APP_VERSION}", card)
        version.setObjectName("Metadata")
        content.addWidget(version)
        content.addSpacing(28)
        self.manifest_line = QLabel("○  Frozen manifest verification pending", card)
        self.artifact_line = QLabel("○  31 allowlisted artifacts pending", card)
        self.threshold_line = QLabel("○  Registered thresholds pending", card)
        self.loading_line = QLabel("◌  Verifying package and loading frozen models", card)
        for label in (
            self.manifest_line,
            self.artifact_line,
            self.threshold_line,
            self.loading_line,
        ):
            label.setObjectName("Secondary")
            content.addWidget(label)
        content.addStretch(1)
        row.addLayout(content, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        screen = self.screen()
        if screen is not None:
            bounds = screen.availableGeometry()
            self.move(bounds.center() - self.rect().center())

    def mark_verified(self) -> None:
        self.manifest_line.setText("✓  Frozen manifest verified")
        self.artifact_line.setText("✓  31 allowlisted artifacts verified")
        self.threshold_line.setText("✓  Registered thresholds verified")
        self.loading_line.setText("✓  Frozen models loaded")

    def mark_blocked(self, code: str) -> None:
        self.loading_line.setText(f"⚠  Startup stopped: {code}")

    def closeEvent(self, event) -> None:
        self.movie.stop()
        super().closeEvent(event)
