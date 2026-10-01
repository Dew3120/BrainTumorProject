"""Approved Gyrus palette, typography, and Qt widget styling."""

from __future__ import annotations

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

COLORS = {
    "background": "#0B1016",
    "navigation": "#0E151D",
    "surface": "#151D27",
    "elevated": "#1B2633",
    "border": "#2A3848",
    "text": "#F3F7FB",
    "secondary": "#A7B3C2",
    "muted": "#748296",
    "primary": "#4EA8DE",
    "focus": "#7CC7F5",
    "standard": "#2FC38C",
    "review": "#F2B84B",
    "error": "#F06464",
    "predicted_mask": "#35D0E0",
    "manual_add": "#A78BFA",
    "manual_remove": "#FF6B8A",
    "image_canvas": "#05070A",
    "pdf_ink": "#17202A",
}


def _available_font(*families: str) -> str:
    available = set(QFontDatabase.families())
    return next((name for name in families if name in available), families[-1])


def apply_theme(app: QApplication) -> None:
    """Apply only logical-pixel sizes so Qt can scale them with display DPI."""

    body = _available_font("Segoe UI Variable Text", "Segoe UI", "Arial")
    app.setFont(QFont(body, 10))
    app.setStyleSheet(
        f"""
        QWidget {{ background: {COLORS['background']}; color: {COLORS['text']};
                  font-family: '{body}'; font-size: 13px; }}
        QLabel {{ background: transparent; }}
        QMainWindow {{ background: {COLORS['background']}; }}
        QFrame#NavigationRail {{ background: {COLORS['navigation']};
                                  border-right: 1px solid {COLORS['border']}; }}
        QFrame#Header {{ background: {COLORS['background']};
                          border-bottom: 1px solid {COLORS['border']}; }}
        QFrame#Footer {{ background: {COLORS['background']};
                          border-top: 1px solid {COLORS['border']}; }}
        QFrame#ImageWorkspace, QFrame#Inspector, QFrame#EmptyDropZone,
        QFrame#SplashCard {{ background: {COLORS['surface']};
                            border: 1px solid {COLORS['border']}; border-radius: 8px; }}
        QLabel#WindowHeading {{ font-size: 24px; font-weight: 600; }}
        QLabel#SectionHeading {{ font-size: 16px; font-weight: 600; }}
        QLabel#CardHeading {{ font-size: 13px; font-weight: 600; }}
        QLabel#Secondary {{ color: {COLORS['secondary']}; }}
        QLabel#Metadata {{ color: {COLORS['muted']}; font-size: 11px; }}
        QLabel#PrivacyChip, QLabel#IntegrityChip, QLabel#DeviceChip {{
            background: {COLORS['elevated']}; border: 1px solid {COLORS['border']};
            border-radius: 4px; padding: 5px 9px; font-size: 11px; font-weight: 600;
        }}
        QLabel#IntegrityChip[state="verified"] {{ color: {COLORS['standard']}; }}
        QLabel#IntegrityChip[state="blocked"] {{ color: {COLORS['error']}; }}
        QLabel#Notice {{ background: {COLORS['elevated']};
                         border: 1px solid {COLORS['primary']};
                         border-radius: 6px; padding: 8px; }}
        QPushButton {{ min-height: 40px; padding: 0 16px; border-radius: 6px;
                       border: 1px solid {COLORS['border']};
                       background: {COLORS['elevated']}; font-weight: 600; }}
        QPushButton:hover:enabled {{ border-color: {COLORS['primary']}; }}
        QPushButton:focus {{ border: 2px solid {COLORS['focus']}; }}
        QPushButton:disabled {{ color: {COLORS['muted']}; }}
        QPushButton#PrimaryAction {{ background: {COLORS['primary']};
                                     color: {COLORS['background']}; border: 0; }}
        QPushButton#NavigationButton {{ min-height: 70px; padding: 2px;
                                        background: transparent; border: 0;
                                        border-radius: 0; font-size: 11px; }}
        QPushButton#NavigationButton[active="true"] {{
            color: {COLORS['primary']}; background: {COLORS['elevated']};
            border-left: 3px solid {COLORS['primary']};
        }}
        """
    )
