"""Read-only aligned image views with zoom, pan, fit, and overlay opacity."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from .input_controller import ValidatedSelection
from .presentation import ResultPresentation

VIEW_NAMES = ("Original", "Probability", "Overlay", "Contour", "Side-by-side")


def _pixmap_from_rgb(rgb: np.ndarray) -> QPixmap:
    value = np.ascontiguousarray(rgb, dtype=np.uint8)
    height, width, _ = value.shape
    image = QImage(value.data, width, height, width * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _heatmap(probability: np.ndarray) -> np.ndarray:
    blue = np.array((42, 111, 219), dtype=np.float64)
    cyan = np.array((53, 208, 224), dtype=np.float64)
    amber = np.array((242, 184, 75), dtype=np.float64)
    low = (blue + (cyan - blue) * (probability * 2)[..., None])
    high = cyan + (amber - cyan) * ((probability - 0.5) * 2)[..., None]
    return np.where((probability < 0.5)[..., None], low, high).clip(0, 255).astype(np.uint8)


def _contour(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(bool)
    inner = value.copy()
    inner[1:, :] &= value[:-1, :]
    inner[:-1, :] &= value[1:, :]
    inner[:, 1:] &= value[:, :-1]
    inner[:, :-1] &= value[:, 1:]
    return value & ~inner


class ImageCanvas(QWidget):
    """All spatial views share the verified 128×128 model coordinate frame."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        header = QFrame(self)
        header.setFixedHeight(48)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(12, 0, 12, 0)
        self.tabs = QTabBar(header)
        for name in VIEW_NAMES:
            self.tabs.addTab(name)
        self.tabs.currentChanged.connect(self._render)
        header_row.addWidget(self.tabs)
        header_row.addStretch(1)
        self.coordinate_label = QLabel("SOURCE PREVIEW", header)
        self.coordinate_label.setObjectName("Metadata")
        header_row.addWidget(self.coordinate_label)
        column.addWidget(header)

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene, self)
        self.view.setObjectName("ModelImageCanvas")
        self.view.setAccessibleName("Image and model mask viewer")
        self.view.setBackgroundBrush(Qt.GlobalColor.black)
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.viewport().installEventFilter(self)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        column.addWidget(self.view, 1)

        strip = QFrame(self)
        strip.setFixedHeight(64)
        strip_row = QHBoxLayout(strip)
        strip_row.setContentsMargins(12, 8, 12, 8)
        self.zoom_out = QPushButton("−", strip)
        self.zoom_out.setAccessibleName("Zoom out image")
        self.zoom_in = QPushButton("+", strip)
        self.zoom_in.setAccessibleName("Zoom in image")
        self.fit_button = QPushButton("Fit", strip)
        self.fit_button.setAccessibleName("Fit image to viewer")
        for button in (self.zoom_out, self.zoom_in):
            button.setFixedWidth(42)
        self.zoom_out.clicked.connect(lambda: self._zoom(1 / 1.25))
        self.zoom_in.clicked.connect(lambda: self._zoom(1.25))
        self.fit_button.clicked.connect(self.fit_image)
        strip_row.addWidget(self.zoom_out)
        strip_row.addWidget(self.zoom_in)
        strip_row.addWidget(self.fit_button)
        strip_row.addStretch(1)
        strip_row.addWidget(QLabel("Overlay opacity", strip))
        self.opacity = QSlider(Qt.Orientation.Horizontal, strip)
        self.opacity.setAccessibleName("Model mask overlay opacity")
        self.opacity.setRange(0, 100)
        self.opacity.setValue(35)
        self.opacity.setFixedWidth(124)
        self.opacity.valueChanged.connect(self._render)
        strip_row.addWidget(self.opacity)
        column.addWidget(strip)
        self._selection: ValidatedSelection | None = None
        self._presentation: ResultPresentation | None = None
        self._fit = True
        self._zoom_factor = 1.0
        self._set_spatial_tabs(False)

    def _set_spatial_tabs(self, enabled: bool) -> None:
        for index in range(1, len(VIEW_NAMES)):
            self.tabs.setTabEnabled(index, enabled)
        if not enabled:
            self.tabs.setCurrentIndex(0)

    def set_selection(self, selection: ValidatedSelection) -> None:
        self._selection = selection
        self._presentation = None
        self._set_spatial_tabs(False)
        self._render()

    def set_result(self, presentation: ResultPresentation) -> None:
        self._selection = presentation.selection
        self._presentation = presentation
        self._set_spatial_tabs(presentation.mask is not None)
        if presentation.mask is not None:
            self.tabs.setCurrentIndex(2)
        else:
            self._render()

    def clear(self) -> None:
        self._selection = None
        self._presentation = None
        self._set_spatial_tabs(False)
        self.image_item.setPixmap(QPixmap())
        self.scene.setSceneRect(0, 0, 0, 0)

    def _render(self, *_args: object) -> None:
        selection = self._selection
        if selection is None:
            return
        mode = self.tabs.currentIndex()
        if mode == 0 or self._presentation is None or self._presentation.mask is None:
            pixmap = QPixmap()
            if not pixmap.loadFromData(selection.preview_png, "PNG"):
                raise ValueError("validated preview could not be rendered")
            self.coordinate_label.setText("SOURCE PREVIEW")
        else:
            base = np.frombuffer(selection.model_gray_128, dtype=np.uint8).reshape(128, 128)
            gray = np.repeat(base[..., None], 3, axis=2)
            probability = self._presentation.probability
            mask = self._presentation.mask
            assert probability is not None and mask is not None
            alpha = self.opacity.value() / 100.0
            if mode == 1:
                rgb = _heatmap(probability)
            elif mode in (2, 4):
                rgb = gray.astype(np.float64)
                rgb[mask.astype(bool)] = (
                    (1 - alpha) * rgb[mask.astype(bool)]
                    + alpha * np.array((53, 208, 224))
                )
                rgb = rgb.astype(np.uint8)
                if mode == 4:
                    rgb = np.concatenate((gray, rgb), axis=1)
            else:
                rgb = gray.copy()
                rgb[_contour(mask)] = np.array((53, 208, 224), dtype=np.uint8)
            pixmap = _pixmap_from_rgb(rgb)
            self.coordinate_label.setText("MODEL SPACE · 128×128")
        self.image_item.setPixmap(pixmap)
        self.scene.setSceneRect(self.image_item.boundingRect())
        self.fit_image()
        # The result can arrive before Qt finishes laying out a newly shown
        # viewer. A second fit after the layout pass prevents a tiny image.
        QTimer.singleShot(0, self._fit_after_layout)

    def _fit_after_layout(self) -> None:
        if self._fit and self.isVisible():
            self.fit_image()

    def fit_image(self) -> None:
        if self.image_item.pixmap().isNull():
            return
        self._fit = True
        self.view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = 1.0

    def _zoom(self, factor: float) -> None:
        if self.image_item.pixmap().isNull():
            return
        next_zoom = self._zoom_factor * factor
        if not 0.25 <= next_zoom <= 16:
            return
        self._fit = False
        self._zoom_factor = next_zoom
        self.view.scale(factor, factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            QTimer.singleShot(0, self._fit_after_layout)

    def eventFilter(self, watched, event) -> bool:
        if watched is self.view.viewport() and event.type() == QEvent.Type.Resize and self._fit:
            QTimer.singleShot(0, self._fit_after_layout)
        return super().eventFilter(watched, event)
