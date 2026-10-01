"""Opt-in local history list, filters, and verified case preview."""

from __future__ import annotations

from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gyrus.domain.history import HistoryRecord
from gyrus.domain.history_bundle import LoadedHistoryCase


class HistoryListPanel(QWidget):
    """Newest-first case rows; filters are applied by the repository query."""

    filters_changed = Signal()
    case_selected = Signal(str)
    back_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(18, 18, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Local history", self)
        title.setObjectName("SectionHeading")
        heading.addWidget(title)
        heading.addStretch(1)
        self.back_button = QPushButton("Back to analysis", self)
        self.back_button.clicked.connect(self.back_requested)
        heading.addWidget(self.back_button)
        column.addLayout(heading)

        filters = QGridLayout()
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search filename or case ID")
        self.status = QComboBox(self)
        self.status.addItems(("Any status", "STANDARD", "REVIEW_REQUIRED"))
        self.category = QComboBox(self)
        self.category.addItems(("Any category", "glioma", "meningioma", "pituitary", "no_tumor"))
        self.ood = QComboBox(self)
        self.ood.addItems(("Any OOD", "ACCEPT", "REVIEW_REQUIRED"))
        self.edited = QComboBox(self)
        self.edited.addItems(("Any edit state", "Edited", "Model only"))
        filters.addWidget(self.search, 0, 0, 1, 2)
        filters.addWidget(self.status, 0, 2)
        filters.addWidget(self.category, 0, 3)
        filters.addWidget(self.ood, 1, 0)
        filters.addWidget(self.edited, 1, 1)
        self.from_enabled = QCheckBox("From", self)
        self.from_date = QDateEdit(QDate.currentDate(), self)
        self.from_date.setCalendarPopup(True)
        self.to_enabled = QCheckBox("To", self)
        self.to_date = QDateEdit(QDate.currentDate(), self)
        self.to_date.setCalendarPopup(True)
        date_row = QHBoxLayout()
        for widget in (self.from_enabled, self.from_date, self.to_enabled, self.to_date):
            date_row.addWidget(widget)
        filters.addLayout(date_row, 1, 2, 1, 2)
        column.addLayout(filters)
        for widget in (self.search,):
            widget.textChanged.connect(self.filters_changed)
        for widget in (self.status, self.category, self.ood, self.edited):
            widget.currentIndexChanged.connect(self.filters_changed)
        for widget in (self.from_enabled, self.to_enabled):
            widget.toggled.connect(self.filters_changed)
        for widget in (self.from_date, self.to_date):
            widget.dateChanged.connect(self.filters_changed)

        self.table = QTableWidget(0, 6, self)
        self.table.setHorizontalHeaderLabels(("Saved (UTC)", "Filename", "Category", "Status", "Edit", "Source"))
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.currentCellChanged.connect(self._on_selected)
        column.addWidget(self.table, 1)
        self.empty = QLabel(
            "No cases saved. Retention is off by default. Choose “Save to local history” "
            "on a result to keep that case on this computer.", self,
        )
        self.empty.setObjectName("Secondary")
        self.empty.setWordWrap(True)
        column.addWidget(self.empty)
        self._case_ids: list[str] = []

    def query(self) -> dict:
        return {
            "query": self.search.text().strip() or None,
            "status": self.status.currentText() if self.status.currentIndex() else None,
            "category": self.category.currentText() if self.category.currentIndex() else None,
            "ood": self.ood.currentText() if self.ood.currentIndex() else None,
            "edited": (None if self.edited.currentIndex() == 0 else self.edited.currentIndex() == 1),
            "date_from": self.from_date.date().toString("yyyy-MM-dd") if self.from_enabled.isChecked() else None,
            "date_to": self.to_date.date().toString("yyyy-MM-dd") if self.to_enabled.isChecked() else None,
        }

    def set_records(self, records: list[HistoryRecord]) -> None:
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self._case_ids = [record.case_id for record in records]
        for row, record in enumerate(records):
            self.table.insertRow(row)
            values = (
                record.saved_at[:16].replace("T", " "), record.filename,
                record.predicted_category, record.final_status,
                "USER EDITED" if record.edited else "—",
                "SAVED" if record.source_saved else "NOT SAVED",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, record.case_id)
                self.table.setItem(row, column, item)
        self.table.blockSignals(False)
        self.empty.setVisible(not records)
        if records:
            self.table.selectRow(0)
            self.case_selected.emit(records[0].case_id)
        else:
            self.case_selected.emit("")

    def _on_selected(self, row: int, _column: int, _old_row: int, _old_column: int) -> None:
        if 0 <= row < len(self._case_ids):
            self.case_selected.emit(self._case_ids[row])


class HistoryPreviewPanel(QWidget):
    """A selected case preview with explicit open and delete actions."""

    open_requested = Signal()
    delete_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        column = QVBoxLayout(self)
        column.setContentsMargins(18, 18, 18, 18)
        heading = QLabel("CASE PREVIEW", self)
        heading.setObjectName("SectionHeading")
        column.addWidget(heading)
        self.preview = QLabel("Select a saved case", self)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setFixedHeight(320)
        self.preview.setObjectName("HistoryPreview")
        column.addWidget(self.preview)
        self.details = QLabel("", self)
        self.details.setWordWrap(True)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        column.addWidget(self.details)
        column.addStretch(1)
        self.open_button = QPushButton("Open saved result", self)
        self.open_button.setObjectName("PrimaryAction")
        self.delete_button = QPushButton("Delete saved case", self)
        self.export_button = QPushButton("Export · Stage 7", self)
        self.export_button.setEnabled(False)
        self.export_button.setToolTip("PDF export arrives in Stage 7")
        self.open_button.clicked.connect(self.open_requested)
        self.delete_button.clicked.connect(self.delete_requested)
        for button in (self.open_button, self.export_button, self.delete_button):
            column.addWidget(button)
        self.clear()

    def clear(self) -> None:
        self.preview.setPixmap(QPixmap())
        self.preview.setText("Select a saved case")
        self.details.setText("History stores only cases you explicitly saved.")
        self.open_button.setEnabled(False)
        self.delete_button.setEnabled(False)

    def set_case(self, loaded: LoadedHistoryCase) -> None:
        pixmap = QPixmap()
        if not pixmap.loadFromData(loaded.overlay_preview_png, "PNG"):
            raise ValueError("Saved overlay preview could not be rendered")
        self.preview.setText("")
        self.preview.setPixmap(pixmap.scaled(
            300, 300, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        ))
        record = loaded.record
        self.details.setText(
            f"{record.filename}\nSaved {record.saved_at}\n"
            f"Model-predicted: {record.predicted_category}\n"
            f"Final status: {record.final_status}\nOOD: {record.ood_status}\n"
            f"Human edited: {'yes' if record.edited else 'no'}\n"
            f"Source retained: {'yes' if record.source_saved else 'no'}\n"
            "Research prototype — not for clinical use"
        )
        self.open_button.setEnabled(True)
        self.delete_button.setEnabled(True)
