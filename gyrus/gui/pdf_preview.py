"""Local, read-only three-page PDF preview and atomic export dialog."""

from __future__ import annotations

import ctypes
import io
import os
import tempfile
from pathlib import Path

from pypdf import PdfReader
from PySide6.QtCore import QBuffer, QByteArray, QEventLoop, QIODevice, Qt, QTimer
from PySide6.QtPdf import QPdfDocument
from PySide6.QtPdfWidgets import QPdfView
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.error_reporting import report_error


class PdfSaveError(ValueError):
    """The destination did not meet the local, atomic-save contract."""


def _local_target(path: str | Path) -> Path:
    selected = Path(path).expanduser().absolute()
    if selected.suffix.lower() != ".pdf":
        raise PdfSaveError("PDF filename must end in .pdf")
    if selected.is_symlink() or any(parent.is_symlink() for parent in selected.parents):
        raise PdfSaveError("PDF destination contains a linked path")
    target = selected.resolve(strict=False)
    if not target.parent.is_dir():
        raise PdfSaveError("PDF destination is unavailable or linked")
    if os.name == "nt" and (
        str(target).startswith("\\\\")
        or ctypes.windll.kernel32.GetDriveTypeW(target.anchor) not in (2, 3)
    ):
        raise PdfSaveError("PDF destination must be on a local drive")
    for key in ("OneDrive", "OneDriveCommercial", "OneDriveConsumer"):
        cloud = os.environ.get(key)
        if cloud and target.is_relative_to(Path(cloud).resolve()):
            raise PdfSaveError("PDF destination is a synchronized cloud folder")
    if any(part.lower().startswith(("onedrive", "dropbox", "google drive", "box sync"))
           for part in target.parts):
        raise PdfSaveError("PDF destination appears to be a synchronized cloud folder")
    return target


def save_pdf_atomic(pdf_bytes: bytes, path: str | Path, *, overwrite: bool = False) -> Path:
    """Commit the already-previewed bytes to one user-chosen local PDF path."""
    if not pdf_bytes.startswith(b"%PDF-"):
        raise PdfSaveError("PDF bytes are invalid")
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        if len(reader.pages) != 3 or reader.metadata.creator != "Gyrus 0.1.0":
            raise PdfSaveError("PDF page count or creator is invalid")
        for page in reader.pages:
            text = page.extract_text()
            if "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE" not in text or DISCLAIMER not in text:
                raise PdfSaveError("PDF page disclaimer is missing")
    except PdfSaveError:
        raise
    except Exception as exc:
        raise PdfSaveError("PDF structure could not be verified") from exc
    target = _local_target(path)
    if target.exists() and not overwrite:
        raise PdfSaveError("PDF already exists; explicit overwrite approval is required")
    staging: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".gyrus-pdf-", suffix=".tmp",
                                         dir=target.parent, delete=False) as stream:
            staging = Path(stream.name)
            stream.write(pdf_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        if staging.read_bytes() != pdf_bytes:
            raise PdfSaveError("Staged PDF does not match previewed bytes")
        if overwrite:
            os.replace(staging, target)
        else:
            os.link(staging, target)  # atomic creation; refuses a destination that appeared meanwhile
            staging.unlink()
    finally:
        if staging is not None and staging.exists():
            staging.unlink()
    return target


class PdfPreviewDialog(QDialog):
    """Use Qt's PDF viewer to preview exact bytes before any local save."""

    def __init__(self, pdf_bytes: bytes, default_filename: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export research summary")
        self.setModal(True)
        self.resize(1160, 780)
        self.setMinimumSize(900, 650)
        self._pdf_bytes = pdf_bytes
        self.default_filename = default_filename
        self.saved_path: Path | None = None
        self._buffer = QBuffer(self)
        self._buffer.setData(QByteArray(pdf_bytes))
        if not self._buffer.open(QIODevice.OpenModeFlag.ReadOnly):
            raise PdfSaveError("Could not open in-memory PDF preview")
        self.document = QPdfDocument(self)
        self.document.load(self._buffer)
        if self.document.status() == QPdfDocument.Status.Loading:
            wait = QEventLoop(self)
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(wait.quit)
            self.document.statusChanged.connect(wait.quit)
            timer.start(5000)
            wait.exec()
            timer.stop()
        if self.document.status() != QPdfDocument.Status.Ready or self.document.pageCount() != 3:
            raise PdfSaveError(f"PDF preview validation failed: {self.document.status()}")
        self._build()

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.setSpacing(14)
        heading = QLabel("Export research summary", self)
        heading.setObjectName("SectionHeading")
        outer.addWidget(heading)
        row = QHBoxLayout()
        row.setSpacing(16)
        self.viewer = QPdfView(self)
        self.viewer.setDocument(self.document)
        self.viewer.setPageMode(QPdfView.PageMode.MultiPage)
        self.viewer.setZoomMode(QPdfView.ZoomMode.FitToWidth)
        self.viewer.setMinimumWidth(600)
        row.addWidget(self.viewer, 1)
        sidebar = QWidget(self)
        sidebar.setFixedWidth(336)
        details = QVBoxLayout(sidebar)
        details.setContentsMargins(10, 0, 0, 0)
        details.setSpacing(14)
        for line in (
            "3 fixed A4 pages",
            "Source image and frozen overlay",
            "Probability, model and saved human layer",
            "OOD, reliability, hashes and limitations",
            "PDF creator: Gyrus 0.1.0",
        ):
            label = QLabel(line, sidebar)
            label.setWordWrap(True)
            details.addWidget(label)
        details.addStretch(1)
        self.destination_label = QLabel(f"Local filename:\n{self.default_filename}", sidebar)
        self.destination_label.setWordWrap(True)
        self.destination_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details.addWidget(self.destination_label)
        self.save_button = QPushButton("Save PDF locally", sidebar)
        self.save_button.setObjectName("PrimaryAction")
        self.save_button.clicked.connect(self._choose_and_save)
        details.addWidget(self.save_button)
        cancel = QPushButton("Cancel", sidebar)
        cancel.clicked.connect(self.reject)
        details.addWidget(cancel)
        row.addWidget(sidebar)
        outer.addLayout(row, 1)
        footer = QLabel(DISCLAIMER, self)
        footer.setObjectName("Metadata")
        outer.addWidget(footer)

    def _choose_and_save(self) -> None:
        local = os.environ.get("LOCALAPPDATA")
        if not local:
            QMessageBox.warning(self, "Local export unavailable", "Windows local application data is unavailable")
            return
        directory = Path(local) / "GyrusResearch" / "exports"
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            notice = report_error("PDF_EXPORT_DIRECTORY_FAILED", "The local export folder is unavailable.",
                                  exception=exc)
            QMessageBox.warning(self, "PDF was not saved", notice.safe_message)
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Gyrus research summary", str(directory / self.default_filename),
            "PDF documents (*.pdf)",
        )
        if not filename:
            return
        target = Path(filename)
        overwrite = False
        if target.exists():
            answer = QMessageBox.question(
                self, "Replace existing PDF?", f"Replace {target.name} with this research summary?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            overwrite = True
        try:
            self.saved_path = save_pdf_atomic(self._pdf_bytes, target, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001 - export failure is visible, not partial success
            notice = report_error("PDF_SAVE_FAILED", "PDF was not saved. Check the destination and local log.",
                                  exception=exc)
            QMessageBox.warning(self, "PDF was not saved", notice.safe_message)
            return
        self.accept()
