"""In-panel, pixel-exact mask editor for a separate human layer."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
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

from gyrus.domain.mask_editor import MaskEditIntegrityError, MaskEditSession

from .image_canvas import _pixmap_from_rgb
from .input_controller import ValidatedSelection


class BrushView(QGraphicsView):
    """Map viewport gestures into model pixels; middle button pans."""

    stroke_finished = Signal(object)

    def __init__(self, scene: QGraphicsScene, parent: QWidget):
        super().__init__(scene, parent)
        self._points: list[tuple[int, int]] = []
        self._panning = False
        self._last_pan = None
        self.setObjectName("ModelImageCanvas")
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)

    def _model_point(self, position) -> tuple[int, int] | None:
        point = self.mapToScene(position.toPoint())
        x, y = int(point.x()), int(point.y())
        if 0 <= x < 128 and 0 <= y < 128:
            return x, y
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._last_pan = event.position().toPoint()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._model_point(event.position())
            self._points = [point] if point is not None else []
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning and self._last_pan is not None:
            delta = event.position().toPoint() - self._last_pan
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self._last_pan = event.position().toPoint()
            event.accept()
            return
        if self._points and event.buttons() & Qt.MouseButton.LeftButton:
            point = self._model_point(event.position())
            if point is not None:
                last_x, last_y = self._points[-1]
                steps = max(abs(point[0] - last_x), abs(point[1] - last_y), 1)
                for step in range(1, steps + 1):
                    self._points.append((round(last_x + (point[0] - last_x) * step / steps),
                                         round(last_y + (point[1] - last_y) * step / steps)))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._last_pan = None
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._points:
                self.stroke_finished.emit(self._points.copy())
            self._points = []
            event.accept()
            return
        super().mouseReleaseEvent(event)


def render_edit_view(gray: np.ndarray, model: np.ndarray, edited: np.ndarray, mode: int) -> np.ndarray:
    """Render overlays without altering either mask; removals use diagonal hatch."""
    base = np.repeat(gray[..., None], 3, axis=2)
    model_color = np.array((53, 208, 224), dtype=np.uint8)
    add_color = np.array((167, 139, 250), dtype=np.uint8)
    remove_color = np.array((255, 107, 138), dtype=np.uint8)

    def overlay(mask: np.ndarray) -> np.ndarray:
        rgb = base.copy()
        rgb[mask] = (0.55 * rgb[mask] + 0.45 * model_color).astype(np.uint8)
        return rgb

    if mode == 0:
        return overlay(model)
    if mode == 1:
        rgb = overlay(edited)
        rgb[edited & ~model] = add_color
        return rgb
    if mode == 3:
        left = overlay(model)
        right = overlay(edited)
        right[edited & ~model] = add_color
        return np.concatenate((left, right), axis=1)
    rgb = overlay(model & edited)
    rgb[edited & ~model] = add_color
    removed = model & ~edited
    yy, xx = np.indices(removed.shape)
    rgb[removed & ((xx + yy) % 6 < 3)] = remove_color
    return rgb


class MaskEditor(QWidget):
    """An editor mode within the existing image workspace."""

    save_requested = Signal()
    close_requested = Signal()
    changed = Signal()
    integrity_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.session: MaskEditSession | None = None
        self.gray: np.ndarray | None = None
        self._fit = True
        self._zoom = 1.0
        self._tool_add = True
        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(0)
        header = QFrame(self)
        header.setFixedHeight(48)
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(12, 0, 12, 0)
        self.tabs = QTabBar(header)
        for title in ("Model", "Edited", "Difference", "Split"):
            self.tabs.addTab(title)
        self.tabs.currentChanged.connect(self._render)
        header_row.addWidget(self.tabs)
        header_row.addStretch(1)
        self.coordinate_label = QLabel("MODEL SPACE · 128×128", header)
        self.coordinate_label.setObjectName("Metadata")
        header_row.addWidget(self.coordinate_label)
        column.addWidget(header)

        canvas_row = QHBoxLayout()
        canvas_row.setContentsMargins(0, 0, 0, 0)
        canvas_row.setSpacing(0)
        tools = QFrame(self)
        tools.setObjectName("EditorToolStrip")
        tools.setFixedWidth(56)
        tool_column = QVBoxLayout(tools)
        tool_column.setContentsMargins(4, 8, 4, 8)
        tool_column.setSpacing(6)
        self.add_button = QPushButton("+", tools)
        self.erase_button = QPushButton("−", tools)
        self.undo_button = QPushButton("↶", tools)
        self.redo_button = QPushButton("↷", tools)
        for button, name in ((self.add_button, "Add mask pixels"),
                             (self.erase_button, "Erase mask pixels"),
                             (self.undo_button, "Undo mask stroke"),
                             (self.redo_button, "Redo mask stroke")):
            button.setAccessibleName(name)
            button.setFixedSize(48, 44)
            tool_column.addWidget(button)
        tool_column.addStretch(1)
        self.add_button.clicked.connect(lambda: self._set_tool(True))
        self.erase_button.clicked.connect(lambda: self._set_tool(False))
        self.undo_button.clicked.connect(self._undo_edit)
        self.redo_button.clicked.connect(self._redo_edit)
        canvas_row.addWidget(tools)
        self.scene = QGraphicsScene(self)
        self.view = BrushView(self.scene, self)
        self.view.stroke_finished.connect(self._stroke)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        canvas_row.addWidget(self.view, 1)
        column.addLayout(canvas_row, 1)

        actions = QFrame(self)
        actions.setFixedHeight(64)
        row = QHBoxLayout(actions)
        self.brush_label = QLabel("Brush 5 px", actions)
        self.brush_size = QSlider(Qt.Orientation.Horizontal, actions)
        self.brush_size.setRange(1, 32)
        self.brush_size.setValue(5)
        self.brush_size.setFixedWidth(110)
        self.brush_size.valueChanged.connect(lambda value: self.brush_label.setText(f"Brush {value} px"))
        self.reset_button = QPushButton("Reset to model", actions)
        self.save_button = QPushButton("Save edited layer", actions)
        self.save_button.setObjectName("PrimaryAction")
        self.close_button = QPushButton("Back to result", actions)
        self.zoom_out = QPushButton("−", actions)
        self.zoom_in = QPushButton("+", actions)
        self.fit_button = QPushButton("Fit", actions)
        for button in (self.zoom_out, self.zoom_in):
            button.setFixedWidth(36)
        self.reset_button.clicked.connect(self._reset_edit)
        self.save_button.clicked.connect(self.save_requested)
        self.close_button.clicked.connect(self.close_requested)
        self.zoom_out.clicked.connect(lambda: self.zoom(1 / 1.25))
        self.zoom_in.clicked.connect(lambda: self.zoom(1.25))
        self.fit_button.clicked.connect(self.fit_image)
        for widget in (self.brush_label, self.brush_size, self.undo_button,
                       self.redo_button, self.reset_button, self.zoom_out,
                       self.zoom_in, self.fit_button, self.save_button, self.close_button):
            if widget not in (self.undo_button, self.redo_button):
                row.addWidget(widget)
        column.addWidget(actions)
        self._set_tool(True)

    def set_case(self, session: MaskEditSession, selection: ValidatedSelection) -> None:
        self.session = session
        self.gray = np.frombuffer(selection.model_gray_128, dtype=np.uint8).reshape(128, 128).copy()
        self.tabs.setCurrentIndex(1)
        self._render()

    def clear(self) -> None:
        self.session = None
        self.gray = None
        self.image_item.setPixmap(QPixmap())

    def _set_tool(self, add: bool) -> None:
        self._tool_add = add
        self.add_button.setProperty("selected", add)
        self.erase_button.setProperty("selected", not add)
        for button in (self.add_button, self.erase_button):
            button.style().unpolish(button)
            button.style().polish(button)

    def _stroke(self, points: list[tuple[int, int]]) -> None:
        if self.session is None or self.tabs.currentIndex() in (0, 3):
            return
        try:
            if self.session.brush(points, self.brush_size.value(), add=self._tool_add):
                self._render()
                self.changed.emit()
        except MaskEditIntegrityError as exc:
            self.integrity_failed.emit(str(exc))

    def _undo_edit(self) -> None:
        try:
            if self.session is not None and self.session.undo():
                self._render()
                self.changed.emit()
        except MaskEditIntegrityError as exc:
            self.integrity_failed.emit(str(exc))

    def _redo_edit(self) -> None:
        try:
            if self.session is not None and self.session.redo():
                self._render()
                self.changed.emit()
        except MaskEditIntegrityError as exc:
            self.integrity_failed.emit(str(exc))

    def _reset_edit(self) -> None:
        try:
            if self.session is not None and self.session.reset():
                self._render()
                self.changed.emit()
        except MaskEditIntegrityError as exc:
            self.integrity_failed.emit(str(exc))

    def _render(self, *_args: object) -> None:
        if self.session is None or self.gray is None:
            return
        rgb = render_edit_view(self.gray, self.session.model,
                               self.session.current, self.tabs.currentIndex())
        self.image_item.setPixmap(_pixmap_from_rgb(rgb))
        self.scene.setSceneRect(self.image_item.boundingRect())
        added = int((self.session.current & ~self.session.model).sum())
        removed = int((self.session.model & ~self.session.current).sum())
        self.coordinate_label.setText(
            f"{int(self.session.current.sum())} px · +{added} / −{removed}"
        )
        if self._fit:
            self.fit_image()
        self.undo_button.setEnabled(bool(self.session._undo))
        self.redo_button.setEnabled(bool(self.session._redo))
        self.save_button.setEnabled(self.session.provenance.edited and self.session.dirty)

    def fit_image(self) -> None:
        if not self.image_item.pixmap().isNull():
            self.view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._fit = True
            self._zoom = 1.0

    def zoom(self, factor: float) -> None:
        value = self._zoom * factor
        if 0.25 <= value <= 16:
            self.view.scale(factor, factor)
            self._zoom = value
            self._fit = False

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fit:
            QTimer.singleShot(0, self.fit_image)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._fit:
            QTimer.singleShot(0, self.fit_image)
