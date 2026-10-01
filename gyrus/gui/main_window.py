"""Gyrus's verified Qt shell, input workflow, and read-only result states."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.domain.mask_editor import MaskEditIntegrityError, MaskEditSession
from gyrus.worker.protocol import AnalyzeSuccess, WorkerFailure

from .image_canvas import ImageCanvas
from .input_controller import InputValidationThread, ValidatedSelection
from .mask_editor import MaskEditor
from .presentation import ResultPresentation, build_presentation
from .result_panel import LoadingPanel, OodReviewDialog, ResultInspector
from .splash import ASSET_ROOT
from .state import AppState, ShellStateMachine


class ImageDropZone(QFrame):
    """Accept exactly one local file; the input controller validates its contents."""

    file_dropped = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if len(urls) == 1 and urls[0].isLocalFile():
            self.file_dropped.emit(urls[0].toLocalFile())
            event.acceptProposedAction()
        else:
            event.ignore()


class GyrusMainWindow(QMainWindow):
    """Show only validated input and final worker output in the active case."""

    retry_requested = Signal()
    analysis_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.machine = ShellStateMachine()
        self._manifest_sha256: str | None = None
        self._artifact_count: int | None = None
        self._device: str | None = None
        self._selection: ValidatedSelection | None = None
        self._validator: InputValidationThread | None = None
        self._pending_result: ResultPresentation | None = None
        self._edit_session: MaskEditSession | None = None
        self._gate: OodReviewDialog | None = None
        self._error_message = ""
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
        self.notice.setTextFormat(Qt.TextFormat.PlainText)
        self.notice.setAccessibleName("Application notice")
        self.notice.setVisible(False)
        right_layout.addWidget(self.notice)
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self.notice.hide)

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
        self.case_label.setTextFormat(Qt.TextFormat.PlainText)
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
        frame_column = QVBoxLayout(workspace)
        frame_column.setContentsMargins(0, 0, 0, 0)
        self.workspace_stack = QStackedWidget(workspace)
        frame_column.addWidget(self.workspace_stack)
        empty_page = QWidget(self.workspace_stack)
        column = QVBoxLayout(empty_page)
        column.setContentsMargins(24, 24, 24, 24)
        column.addStretch(1)
        self.drop_zone = ImageDropZone(empty_page)
        self.drop_zone.setObjectName("EmptyDropZone")
        self.drop_zone.setFixedSize(440, 260)
        self.drop_zone.file_dropped.connect(self.select_image)
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
        self.choose_button.clicked.connect(self.choose_image)
        drop_layout.addWidget(self.choose_button, alignment=Qt.AlignmentFlag.AlignCenter)
        detail = QLabel("Processed on this computer", self.drop_zone)
        detail.setObjectName("Metadata")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        drop_layout.addWidget(detail)
        drop_layout.addStretch(1)
        column.addWidget(self.drop_zone, alignment=Qt.AlignmentFlag.AlignCenter)
        column.addStretch(1)
        self.empty_label = QLabel("No file selected", empty_page)
        self.empty_label.setObjectName("Metadata")
        column.addWidget(self.empty_label)
        self.workspace_message = QLabel("", empty_page)
        self.workspace_message.setObjectName("Secondary")
        self.workspace_message.setTextFormat(Qt.TextFormat.PlainText)
        self.workspace_message.setWordWrap(True)
        self.workspace_message.setVisible(False)
        column.addWidget(self.workspace_message)
        self.retry_button = QPushButton("Retry verification", empty_page)
        self.retry_button.setAccessibleName("Retry frozen stack verification")
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self.retry_boot)
        column.addWidget(self.retry_button, alignment=Qt.AlignmentFlag.AlignLeft)
        self.workspace_stack.addWidget(empty_page)

        self.image_canvas = ImageCanvas(self.workspace_stack)
        self.workspace_stack.addWidget(self.image_canvas)
        self.mask_editor = MaskEditor(self.workspace_stack)
        self.mask_editor.save_requested.connect(self.save_edited_layer)
        self.mask_editor.close_requested.connect(self.close_mask_editor)
        self.mask_editor.changed.connect(self._update_edit_status)
        self.mask_editor.integrity_failed.connect(
            lambda reason: self._lock_result(f"MASK_EDIT_INTEGRITY_FAILED: {reason}")
        )
        self.workspace_stack.addWidget(self.mask_editor)
        return workspace

    def _make_inspector(self, parent: QWidget) -> QFrame:
        inspector = QFrame(parent)
        inspector.setObjectName("Inspector")
        inspector.setFixedWidth(480)
        outer = QVBoxLayout(inspector)
        outer.setContentsMargins(0, 0, 0, 0)
        self.inspector_stack = QStackedWidget(inspector)
        outer.addWidget(self.inspector_stack)
        empty_page = QWidget(self.inspector_stack)
        column = QVBoxLayout(empty_page)
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
        self.inspector_stack.addWidget(empty_page)

        input_page = QWidget(self.inspector_stack)
        input_column = QVBoxLayout(input_page)
        input_column.setContentsMargins(20, 20, 20, 20)
        input_column.setSpacing(15)
        heading = QLabel("VALIDATED LOCAL IMAGE", input_page)
        heading.setObjectName("SectionHeading")
        input_column.addWidget(heading)
        self.input_metadata = QLabel("", input_page)
        self.input_metadata.setTextFormat(Qt.TextFormat.PlainText)
        self.input_metadata.setWordWrap(True)
        self.input_metadata.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        input_column.addWidget(self.input_metadata)
        input_column.addStretch(1)
        self.run_button = QPushButton("Run frozen analysis", input_page)
        self.run_button.setObjectName("PrimaryAction")
        self.run_button.clicked.connect(self.run_analysis)
        input_column.addWidget(self.run_button)
        self.clear_button = QPushButton("Choose another image", input_page)
        self.clear_button.clicked.connect(self.clear_case)
        input_column.addWidget(self.clear_button)
        self.inspector_stack.addWidget(input_page)

        self.loading_panel = LoadingPanel(self.inspector_stack)
        self.inspector_stack.addWidget(self.loading_panel)
        self.result_inspector = ResultInspector(self.inspector_stack)
        self.result_inspector.new_case_button.clicked.connect(self.clear_case)
        self.result_inspector.edit_mask_button.clicked.connect(self.open_mask_editor)
        self.inspector_stack.addWidget(self.result_inspector)
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

    def choose_image(self) -> None:
        path, _selected_filter = QFileDialog.getOpenFileName(
            self,
            "Choose one local image",
            "",
            "Images (*.png *.jpg *.jpeg)",
        )
        if path:
            self.select_image(path)

    def select_image(self, path: str) -> None:
        """Selection is explicit and never starts frozen inference."""

        if self.machine.state in {
            AppState.READY_TO_ANALYZE,
            AppState.RESULT_STANDARD,
            AppState.RESULT_REVIEW_REQUIRED,
            AppState.RECOVERABLE_ERROR,
        }:
            self.clear_case()
        if self.machine.state != AppState.READY_EMPTY or self._validator is not None:
            raise RuntimeError("image selection is unavailable in this state")
        self.machine.enter(AppState.INPUT_VALIDATING, reason="local image selected")
        self._render_state()
        self._validator = InputValidationThread(path)
        self._validator.validated.connect(self._on_input_validated)
        self._validator.rejected.connect(self._on_input_rejected)
        self._validator.finished.connect(self._on_validation_finished)
        self._validator.start()

    def _on_validation_finished(self) -> None:
        if self._validator is not None:
            self._validator.deleteLater()
            self._validator = None

    def _on_input_validated(self, selection: ValidatedSelection) -> None:
        if self.machine.state != AppState.INPUT_VALIDATING:
            return
        self._selection = selection
        self.image_canvas.set_selection(selection)
        descriptor = selection.descriptor
        self.case_label.setText(descriptor.filename)
        self.input_metadata.setText(
            f"Filename: {descriptor.filename}\n"
            f"Dimensions: {descriptor.width} × {descriptor.height} px\n"
            f"Mode: {descriptor.source_mode} · {descriptor.image_format}\n"
            f"SHA256: {descriptor.source_sha256}\n\n"
            "Inspect the preview, then start analysis explicitly. "
            "The worker will validate the source again before inference."
        )
        self.machine.enter(AppState.READY_TO_ANALYZE, reason="input contract validated")
        self._render_state()

    def _on_input_rejected(self, code: str, message: str) -> None:
        if self.machine.state != AppState.INPUT_VALIDATING:
            return
        self._error_message = f"{code}: {message}"
        self.machine.enter(AppState.RECOVERABLE_ERROR, reason=code)
        self._render_state()

    def run_analysis(self) -> None:
        if self.machine.state != AppState.READY_TO_ANALYZE or self._selection is None:
            raise RuntimeError("analysis requires one validated image")
        self._pending_result = None
        self.machine.enter(AppState.ANALYZING, reason="user started frozen analysis")
        self.loading_panel.start()
        self._render_state()
        self.analysis_requested.emit(self._selection.descriptor.source_path)

    def on_job_finished(self, response: AnalyzeSuccess) -> None:
        if self.machine.state != AppState.ANALYZING or self._selection is None:
            self._lock_result("RESULT_ARRIVED_OUT_OF_STATE")
            return
        try:
            presentation = build_presentation(response, self._selection)
        except Exception as exc:  # noqa: BLE001 - no unverified result can reach a widget
            self._lock_result(f"RESULT_VERIFICATION_FAILED: {type(exc).__name__}: {exc}")
            return
        self.loading_panel.complete()
        self._pending_result = presentation
        ood = presentation.result["ood"]
        if ood["status"] == "REVIEW_REQUIRED":
            self.machine.enter(AppState.OOD_REVIEW_GATE, reason="OOD review disclosure gate")
            self._render_state()
            self._gate = OodReviewDialog(ood["score"], self)
            self._gate.finished.connect(self._on_gate_closed)
            self._gate.open()
        else:
            self._show_final_result()

    def _on_gate_closed(self, code: int) -> None:
        self._gate = None
        if self.machine.state != AppState.OOD_REVIEW_GATE:
            return
        if code == QDialog.DialogCode.Accepted:
            self._show_final_result()
        else:
            self.clear_case()

    def _show_final_result(self) -> None:
        presentation = self._pending_result
        if presentation is None:
            self._lock_result("RESULT_NOT_AVAILABLE")
            return
        status = presentation.result["final_policy"]["status"]
        target = (
            AppState.RESULT_STANDARD
            if status == "STANDARD"
            else AppState.RESULT_REVIEW_REQUIRED
        )
        self.machine.enter(target, reason="complete validated result disclosed")
        self.image_canvas.set_result(presentation)
        self.result_inspector.set_result(presentation)
        self._edit_session = None
        self._render_state()

    def open_mask_editor(self) -> None:
        if self.machine.state not in {AppState.RESULT_STANDARD, AppState.RESULT_REVIEW_REQUIRED}:
            raise RuntimeError("Mask editing requires a disclosed result")
        presentation = self._pending_result
        if presentation is None or presentation.mask is None:
            raise RuntimeError("No model mask is available for editing")
        if self._edit_session is None:
            self._edit_session = MaskEditSession(
                presentation.result["case_id"], presentation.mask,
                presentation.result["reliability"]["score"],
            )
        self.mask_editor.set_case(self._edit_session, presentation.selection)
        self.machine.enter(AppState.EDITING_MASK, reason="user opened manual mask editor")
        self._update_edit_status()
        self._render_state()

    def _update_edit_status(self) -> None:
        if self._edit_session is not None:
            self.result_inspector.set_edit_status(self._edit_session)

    def save_edited_layer(self) -> None:
        if self.machine.state != AppState.EDITING_MASK or self._edit_session is None:
            raise RuntimeError("No active edit to save")
        try:
            local_appdata = os.environ.get("LOCALAPPDATA")
            if not local_appdata:
                raise RuntimeError("Windows local application-data folder is unavailable")
            directory = Path(local_appdata) / "GyrusResearch" / "edited_layers"
            directory.mkdir(parents=True, exist_ok=True)
            target = self._edit_session.export(directory)
        except MaskEditIntegrityError as exc:
            self._lock_result(f"MASK_EDIT_INTEGRITY_FAILED: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - I/O failure is visible, never success
            self.show_notice(f"Edited layer was not saved: {type(exc).__name__}: {exc}")
            return
        self._update_edit_status()
        self.mask_editor._render()
        self.show_notice(f"Edited layer saved locally: {target}")

    def close_mask_editor(self) -> None:
        if self.machine.state != AppState.EDITING_MASK or self._pending_result is None:
            raise RuntimeError("Mask editor is not active")
        if self._edit_session is not None and self._edit_session.dirty:
            answer = QMessageBox.question(
                self, "Discard unsaved edits?", "Unsaved manual edits will be discarded. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            if self._edit_session.saved_mask is None:
                self._edit_session = None
            else:
                self._edit_session.discard_since_save()
            self._update_edit_status()
        target = (AppState.RESULT_STANDARD if self._pending_result.result["final_policy"]["status"] == "STANDARD"
                  else AppState.RESULT_REVIEW_REQUIRED)
        self.machine.enter(target, reason="user returned from manual mask editor")
        self._render_state()

    def on_job_failed(self, response: object) -> None:
        if self.machine.state != AppState.ANALYZING:
            return
        if isinstance(response, WorkerFailure):
            self._error_message = f"{response.error_code}: {response.safe_message}"
            fatal = response.fatal
        else:
            self._error_message = str(response)
            fatal = False
        self._pending_result = None
        self.machine.enter(
            AppState.INTEGRITY_ERROR_LOCKED if fatal else AppState.RECOVERABLE_ERROR,
            reason="worker analysis failed without result disclosure",
        )
        self._render_state()

    def _lock_result(self, reason: str) -> None:
        self._pending_result = None
        self._error_message = reason
        if self.machine.can_enter(AppState.INTEGRITY_ERROR_LOCKED):
            self.machine.enter(AppState.INTEGRITY_ERROR_LOCKED, reason=reason)
        self._render_state()

    def clear_case(self) -> None:
        if self._gate is not None:
            self._gate.close()
            self._gate = None
        if self.machine.state != AppState.READY_EMPTY:
            self.machine.enter(AppState.READY_EMPTY, reason="user cleared current case")
        self._selection = None
        self._pending_result = None
        self._edit_session = None
        self.mask_editor.clear()
        self._error_message = ""
        self.case_label.setText("New analysis")
        self.image_canvas.clear()
        self._render_state()

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
        self._notice_timer.start(duration_ms)

    def _render_state(self) -> None:
        state = self.machine.state
        verified_states = {
            AppState.READY_EMPTY,
            AppState.INPUT_VALIDATING,
            AppState.READY_TO_ANALYZE,
            AppState.ANALYZING,
            AppState.OOD_REVIEW_GATE,
            AppState.RESULT_STANDARD,
            AppState.RESULT_REVIEW_REQUIRED,
            AppState.EDITING_MASK,
            AppState.RECOVERABLE_ERROR,
        }
        if state in verified_states:
            self.integrity_chip.setText("✓  VERIFIED · 31")
            self.integrity_chip.setProperty("state", "verified")
            self.device_chip.setText(
                "GPU · CUDA:0" if self._device == "cuda" else "CPU MODE"
            )
            if self._manifest_sha256 is not None:
                self.manifest_label.setText(
                    f"Frozen manifest\n{self._manifest_sha256[:16]}…"
                )
        elif state in {AppState.BOOT_BLOCKED, AppState.INTEGRITY_ERROR_LOCKED}:
            self.integrity_chip.setText("⚠  VERIFICATION STOPPED")
            self.integrity_chip.setProperty("state", "blocked")
            self.device_chip.setText("DEVICE UNAVAILABLE")
            self.manifest_label.setText("Frozen manifest\nNot verified")
        else:
            self.integrity_chip.setText("○  VERIFYING STACK")
            self.integrity_chip.setProperty("state", "verifying")

        show_image = state in {
            AppState.READY_TO_ANALYZE,
            AppState.ANALYZING,
            AppState.OOD_REVIEW_GATE,
            AppState.RESULT_STANDARD,
            AppState.RESULT_REVIEW_REQUIRED,
        }
        self.workspace_stack.setCurrentIndex(2 if state == AppState.EDITING_MASK else (1 if show_image else 0))
        if state == AppState.READY_TO_ANALYZE:
            self.inspector_stack.setCurrentIndex(1)
        elif state in {AppState.ANALYZING, AppState.OOD_REVIEW_GATE}:
            self.inspector_stack.setCurrentIndex(2)
        elif state in {AppState.RESULT_STANDARD, AppState.RESULT_REVIEW_REQUIRED, AppState.EDITING_MASK}:
            self.inspector_stack.setCurrentIndex(3)
        else:
            self.inspector_stack.setCurrentIndex(0)

        self.drop_zone.setVisible(state in {AppState.READY_EMPTY, AppState.RECOVERABLE_ERROR})
        self.choose_button.setEnabled(state in {AppState.READY_EMPTY, AppState.RECOVERABLE_ERROR})
        self.run_button.setEnabled(state == AppState.READY_TO_ANALYZE)
        self.result_inspector.edit_mask_button.setEnabled(
            state in {AppState.RESULT_STANDARD, AppState.RESULT_REVIEW_REQUIRED}
            and self._pending_result is not None
            and self._pending_result.mask is not None
        )
        self.result_inspector.new_case_button.setEnabled(state != AppState.EDITING_MASK)
        self.empty_label.setVisible(state == AppState.READY_EMPTY)
        self.workspace_message.setVisible(
            state in {
                AppState.INPUT_VALIDATING,
                AppState.BOOT_BLOCKED,
                AppState.INTEGRITY_ERROR_LOCKED,
                AppState.RECOVERABLE_ERROR,
            }
        )
        self.retry_button.setVisible(state == AppState.BOOT_BLOCKED)
        if state == AppState.INPUT_VALIDATING:
            self.workspace_message.setText("Validating the selected local image…")
        elif state == AppState.BOOT_BLOCKED:
            self.workspace_message.setText(
                "The worker could not start. Review the local device setting and retry."
            )
        elif state == AppState.INTEGRITY_ERROR_LOCKED:
            self.workspace_message.setText(
                "Integrity or result verification stopped analysis. Restart Gyrus after "
                f"investigating: {self._error_message or 'frozen stack mismatch'}"
            )
        elif state == AppState.RECOVERABLE_ERROR:
            self.workspace_message.setText(self._error_message)
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
