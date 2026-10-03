"""End-to-end Stage 4 selection, response disclosure, and OOD gate checks."""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.gui.input_controller import ValidatedSelection
from gyrus.gui.main_window import GyrusMainWindow
from gyrus.gui.presentation import PresentationError, build_presentation
from gyrus.gui.state import AppState
from gyrus.gui.work_controller import WorkerBridge
from gyrus.worker.process import WorkerSupervisor
from gyrus.worker.protocol import AnalyzeJob, AnalyzeSuccess, DeviceName, WorkerConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs/final_stack/verification/fixtures"


def _wait_for_local_task(qtbot, window):
    qtbot.waitUntil(lambda: window._local_task is None, timeout=20_000)


def test_history_save_runs_off_ui_thread_and_close_waits(
    tmp_path, qtbot, monkeypatch, frozen_responses,
):
    from threading import Event
    from types import SimpleNamespace

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QMessageBox

    entered, release = Event(), Event()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])

    def slow_save(_repository, _payload):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("synthetic save was not released")
        return SimpleNamespace(case_id=window._pending_result.result["case_id"])

    monkeypatch.setattr("gyrus.gui.main_window.save_history_bundle", slow_save)
    try:
        window.save_to_history()
        qtbot.waitUntil(entered.is_set, timeout=5_000)
        ticked = []
        QTimer.singleShot(0, lambda: ticked.append(True))
        qtbot.waitUntil(lambda: bool(ticked), timeout=1_000)
        assert not window.result_inspector.export_pdf_button.isEnabled()
        window.close()
        assert window.isVisible()
    finally:
        release.set()
    qtbot.waitUntil(lambda: window.isHidden(), timeout=5_000)


def _ready():
    from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256

    return SimpleNamespace(
        artifact_count=31, manifest_sha256=EXPECTED_MANIFEST_SHA256, device="cpu"
    )


def _selection(name: str) -> ValidatedSelection:
    decoded = validate_and_decode_image(FIXTURES / name)
    processed = preprocess_for_frozen_stack(decoded)
    image = Image.fromarray(decoded.gray_uint8)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype("uint8")
    return ValidatedSelection(decoded.descriptor, buffer.getvalue(), gray.tobytes())


@pytest.fixture(scope="module")
def frozen_responses():
    supervisor = WorkerSupervisor(
        WorkerConfig(str(ROOT / "runs/final_stack"), DeviceName.CPU)
    )
    try:
        supervisor.start()
        output = {}
        for name in ("elliptic.png", "gradient.png"):
            output[name] = supervisor.analyze(
                AnalyzeJob(
                    str(uuid4()),
                    str(uuid4()),
                    str(FIXTURES / name),
                    datetime.now().astimezone().isoformat(timespec="milliseconds"),
                )
            )
        yield output
    finally:
        supervisor.close()


def test_file_selection_is_validated_without_autorun(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    emitted: list[str] = []
    window.analysis_requested.connect(emitted.append)
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    assert window._selection is not None
    assert window._selection.descriptor.filename == "elliptic.png"
    assert window._selection.descriptor.source_sha256 in window.input_metadata.text()
    assert window.run_button.isEnabled()
    assert emitted == []
    window.run_analysis()
    assert window.machine.state == AppState.ANALYZING
    assert emitted == [str(FIXTURES / "elliptic.png")]
    assert "◌" in window.loading_panel.lines[1].text()


def test_bad_image_fails_before_analysis(tmp_path, qtbot):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    jobs: list[str] = []
    window.analysis_requested.connect(jobs.append)
    window.select_image(str(bad))
    qtbot.waitUntil(lambda: window.machine.state == AppState.RECOVERABLE_ERROR, timeout=20_000)
    assert "INPUT_DECODE_FAILED" in window.workspace_message.text()
    assert jobs == []
    assert window.choose_button.isEnabled()


def test_drop_zone_accepts_one_local_path(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(FIXTURES / "elliptic.png"))])
    enter = QDragEnterEvent(
        QPoint(20, 20),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(window.drop_zone, enter)
    assert enter.isAccepted()
    event = QDropEvent(
        QPointF(20, 20),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    window.drop_zone.dropEvent(event)
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    assert window._selection.descriptor.filename == "elliptic.png"


def test_standard_result_is_visible_only_after_complete_response(qtbot, frozen_responses):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    assert window.result_inspector.isVisible() is False
    window.on_job_finished(frozen_responses["elliptic.png"])
    assert window.machine.state == AppState.RESULT_STANDARD
    assert "STANDARD" in window.result_inspector.status.text()
    assert window.image_canvas.tabs.isTabEnabled(2)
    assert window.image_canvas.tabs.currentIndex() == 2
    qtbot.waitUntil(
        lambda: window.image_canvas.view.mapFromScene(
            window.image_canvas.image_item.sceneBoundingRect()
        ).boundingRect().width() >= window.image_canvas.view.viewport().width() * 0.5,
        timeout=5_000,
    )
    assert "0.9490250173235835" in window.result_inspector.reliability.text()
    assert "0.36854275476100085" in window.result_inspector.ood.text()
    assert window.workspace.width() == 820
    assert window.workspace.height() == 748
    window.result_inspector.provenance_toggle.click()
    assert "v1_fallback" in window.result_inspector.provenance.toPlainText()
    for index in range(5):
        window.image_canvas.tabs.setCurrentIndex(index)
        assert not window.image_canvas.image_item.pixmap().isNull()
    window.image_canvas.opacity.setValue(65)
    window.image_canvas.zoom_in.click()
    window.image_canvas.fit_button.click()
    window.resize(1180, 720)
    qtbot.wait(40)
    assert window.workspace.width() == 640


def test_ood_review_is_modal_before_result_and_no_tumor_has_no_mask(qtbot, frozen_responses):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "gradient.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["gradient.png"])
    assert window.machine.state == AppState.OOD_REVIEW_GATE
    assert not window.result_inspector.isVisible()
    assert window._gate is not None
    assert any(
        "0.36854275476100085" in label.text()
        for label in window._gate.findChildren(type(window.case_label))
    )
    assert "Research prototype — not for clinical use" in window._gate.findChildren(
        type(window.case_label)
    )[-1].text()
    window._gate.view_output.click()
    assert window.machine.state == AppState.RESULT_REVIEW_REQUIRED
    assert "Not applicable" in window.result_inspector.reliability.text()
    assert not window.image_canvas.tabs.isTabEnabled(2)
    assert window.result_inspector.category.text() == "No Tumor"


def test_ood_gate_choose_another_discards_undisclosed_output(qtbot, frozen_responses):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "gradient.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["gradient.png"])
    assert window.machine.state == AppState.OOD_REVIEW_GATE
    window._gate.choose_another.click()
    assert window.machine.state == AppState.READY_EMPTY
    assert window._pending_result is None
    assert window._selection is None


def test_selection_hash_mismatch_fails_closed(frozen_responses):
    response = frozen_responses["elliptic.png"]
    wrong_selection = _selection("gradient.png")
    assert isinstance(response, AnalyzeSuccess)
    with pytest.raises(PresentationError, match="RESULT_INPUT_CHANGED_AFTER_VALIDATION"):
        build_presentation(response, wrong_selection)


def test_mask_editor_save_preserves_frozen_result(tmp_path, qtbot, monkeypatch, frozen_responses):
    from gyrus.domain.mask_provenance import mask_sha256

    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    presentation = window._pending_result
    assert presentation is not None and presentation.mask is not None
    model_hash = mask_sha256(presentation.mask)
    score = presentation.result["reliability"]["score"]
    window.result_inspector.edit_mask_button.click()
    assert window.machine.state == AppState.EDITING_MASK
    assert window.workspace_stack.currentWidget() is window.mask_editor
    assert window.mask_editor.findChildren(type(window.mask_editor.add_button))[0].width() <= 48
    assert not window.result_inspector.edited_badge.isVisible()
    window.mask_editor._stroke([(0, 0)])
    assert window.mask_editor.session is not None
    assert window.mask_editor.session.current[0, 0]
    window.mask_editor.undo_button.click()
    assert not window.mask_editor.session.current[0, 0]
    window.mask_editor.redo_button.click()
    assert window.mask_editor.session.current[0, 0]
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    window.mask_editor.save_button.click()
    assert window.result_inspector.edited_badge.isVisible()
    assert "ORIGINAL_MODEL_MASK_ONLY" in window.result_inspector.provenance.toPlainText()
    assert mask_sha256(presentation.mask) == model_hash
    assert presentation.result["reliability"]["score"] == score
    assert "original model mask only" in window.result_inspector.reliability.text()
    assert len(list((tmp_path / "GyrusResearch" / "edited_layers").glob("gyrus-edit_*"))) == 1
    window.mask_editor.close_button.click()
    assert window.machine.state == AppState.RESULT_STANDARD
    assert window.result_inspector.edited_badge.isVisible()


def test_no_tumor_editor_unavailable_and_edit_tamper_locks(qtbot, frozen_responses):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "gradient.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["gradient.png"])
    window._gate.view_output.click()
    assert not window.result_inspector.edit_mask_button.isEnabled()
    with pytest.raises(RuntimeError, match="No model mask"):
        window.open_mask_editor()

    window.clear_case()
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    window.open_mask_editor()
    window._edit_session.current[0, 0] = True
    window.mask_editor._stroke([(70, 70)])
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert window.result_inspector.isVisible() is False


def test_history_is_opt_in_filters_reopens_and_deletes(tmp_path, qtbot, monkeypatch, frozen_responses):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    root = tmp_path / "GyrusResearch"
    window.history_button.click()
    assert window.machine.state == AppState.HISTORY
    assert window.history_list.table.rowCount() == 0
    assert "Retention is off by default" in window.history_list.empty.text()
    assert not root.exists()
    window.close_history()
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    assert not root.exists()
    window.result_inspector.save_history_button.click()
    _wait_for_local_task(qtbot, window)
    assert (root / "history.db").exists()
    assert window.result_inspector.save_history_button.isEnabled() is False
    window.history_button.click()
    assert window.history_list.table.rowCount() == 1
    assert window.history_list.table.item(0, 5).text() == "SAVED"
    assert window.history_preview.open_button.isEnabled()
    window.history_list.category.setCurrentText("pituitary")
    assert window.history_list.table.rowCount() == 0
    window.history_list.category.setCurrentText("meningioma")
    assert window.history_list.table.rowCount() == 1
    window.history_preview.open_button.click()
    assert window.machine.state == AppState.RESULT_STANDARD
    assert window.image_canvas.tabs.isTabEnabled(2)
    assert window.result_inspector.save_history_button.isEnabled() is False
    window.history_button.click()
    window.history_preview.delete_button.click()
    assert window.history_list.table.rowCount() == 0
    assert not list((root / "cases").iterdir())


def test_history_bundle_hash_mismatch_locks_preview(tmp_path, qtbot, monkeypatch, frozen_responses):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    window.save_to_history()
    _wait_for_local_task(qtbot, window)
    bundle = next((tmp_path / "GyrusResearch" / "cases").iterdir())
    (bundle / "overlay_preview.png").write_bytes(b"tampered")
    window.open_history()
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert window.history_preview.open_button.isEnabled() is False


def test_live_and_saved_pdf_preview_use_verified_three_page_document(
    tmp_path, qtbot, monkeypatch, frozen_responses,
):
    from PySide6.QtWidgets import QMessageBox

    from gyrus.gui.pdf_preview import PdfPreviewDialog

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    states_seen = []

    def close_preview(dialog):
        states_seen.append(dialog.parent().machine.state)
        assert dialog.document.pageCount() == 3
        return 0

    monkeypatch.setattr(PdfPreviewDialog, "exec", close_preview)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    assert window.result_inspector.export_pdf_button.isEnabled()
    window.export_current_pdf()
    _wait_for_local_task(qtbot, window)
    assert window.machine.state == AppState.RESULT_STANDARD
    window.save_to_history()
    _wait_for_local_task(qtbot, window)
    window.open_history()
    assert window.history_preview.export_button.isEnabled()
    window.export_saved_pdf()
    _wait_for_local_task(qtbot, window)
    assert window.machine.state == AppState.HISTORY
    assert states_seen == [AppState.EXPORT_PREVIEW, AppState.EXPORT_PREVIEW]


def test_saved_pdf_export_reverifies_bundle_before_preview(
    tmp_path, qtbot, monkeypatch, frozen_responses,
):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    window.save_to_history()
    _wait_for_local_task(qtbot, window)
    window.open_history()
    assert window.history_preview.export_button.isEnabled()
    bundle = next((tmp_path / "GyrusResearch" / "cases").iterdir())
    (bundle / "overlay_preview.png").write_bytes(b"changed after selection")
    window.export_saved_pdf()
    _wait_for_local_task(qtbot, window)
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED


def test_saved_human_layer_restores_without_recomputing_reliability(
    tmp_path, qtbot, monkeypatch, frozen_responses,
):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    original_score = window._pending_result.result["reliability"]["score"]
    window.open_mask_editor()
    window.mask_editor._stroke([(0, 0)])
    window.save_edited_layer()
    window.close_mask_editor()
    window.save_to_history()
    _wait_for_local_task(qtbot, window)
    window.open_history()
    assert window.history_list.table.item(0, 4).text() == "USER EDITED"
    window.open_saved_history_case()
    assert window.result_inspector.edited_badge.isVisible()
    assert window._edit_session.saved_mask[0, 0]
    assert window._pending_result.result["reliability"]["score"] == original_score
    assert "original model mask only" in window.result_inspector.reliability.text()


def test_no_tumor_history_retains_source_without_mask(tmp_path, qtbot, monkeypatch, frozen_responses):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "gradient.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["gradient.png"])
    window._gate.view_output.click()
    window.save_to_history()
    _wait_for_local_task(qtbot, window)
    bundle = next((tmp_path / "GyrusResearch" / "cases").iterdir())
    assert (bundle / "source.png").exists()
    assert not (bundle / "model_mask.png").exists()
    window.open_history()
    window.open_saved_history_case()
    assert window.machine.state == AppState.RESULT_REVIEW_REQUIRED
    assert window._pending_result.mask is None
    assert not window.result_inspector.edit_mask_button.isEnabled()


def test_history_save_cancel_and_failed_index_leave_no_case(
    tmp_path, qtbot, monkeypatch, frozen_responses,
):
    from PySide6.QtWidgets import QMessageBox

    from gyrus.domain.history import HistoryRepository
    from gyrus.domain.history_bundle import HistoryCasePayload, save_history_bundle

    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.select_image(str(FIXTURES / "elliptic.png"))
    qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
    window.run_analysis()
    window.on_job_finished(frozen_responses["elliptic.png"])
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.No)
    window.save_to_history()
    root = tmp_path / "GyrusResearch"
    assert not root.exists()
    presentation = window._pending_result
    payload = HistoryCasePayload(
        result=presentation.result,
        source_path=presentation.selection.descriptor.source_path,
        preview_png=presentation.selection.preview_png,
        model_gray_128=presentation.selection.model_gray_128,
        probability=presentation.probability,
        model_mask=presentation.mask,
        edited_mask=None,
        edit_provenance=None,
        job_id=presentation.job_id,
        route=presentation.route,
        raw_probability_sha256=presentation.raw_probability_sha256,
    )
    repo = HistoryRepository(root / "history.db", root / "cases")
    monkeypatch.setattr(repo, "save", lambda _record: (_ for _ in ()).throw(RuntimeError("simulated index failure")))
    with pytest.raises(RuntimeError, match="simulated index failure"):
        save_history_bundle(repo, payload)
    assert not list((root / "cases").iterdir())
    altered = tmp_path / "altered.png"
    altered.write_bytes(b"changed after analysis")
    with pytest.raises(Exception, match="HISTORY_SOURCE_CHANGED"):
        save_history_bundle(repo, replace(payload, source_path=str(altered)))
    assert not list((root / "cases").iterdir())


def test_threshold_drift_fails_closed(frozen_responses):
    response = frozen_responses["elliptic.png"]
    result = {**response.result, "ood": {**response.result["ood"], "threshold": "0.37"}}
    drifted = replace(response, result=result)
    with pytest.raises(PresentationError, match="RESULT_OOD_THRESHOLD_DRIFT"):
        build_presentation(drifted, _selection("elliptic.png"))


def test_complete_qt_worker_flow_discloses_only_final_result(qtbot):
    window = GyrusMainWindow()
    bridge = WorkerBridge(ROOT / "runs/final_stack", "cpu")
    qtbot.addWidget(window)
    window.show()
    bridge.verified.connect(window.on_worker_ready)
    bridge.job_finished.connect(window.on_job_finished)
    bridge.job_failed.connect(window.on_job_failed)
    window.analysis_requested.connect(bridge.submit_image)
    try:
        with qtbot.waitSignal(bridge.verified, timeout=60_000):
            bridge.start()
        assert window.machine.state == AppState.READY_EMPTY
        window.select_image(str(FIXTURES / "elliptic.png"))
        qtbot.waitUntil(lambda: window.machine.state == AppState.READY_TO_ANALYZE, timeout=20_000)
        with qtbot.waitSignal(bridge.job_finished, timeout=60_000):
            window.run_analysis()
            assert window.machine.state == AppState.ANALYZING
            assert not window.result_inspector.isVisible()
        assert window.machine.state == AppState.RESULT_STANDARD
        assert window.image_canvas.image_item.pixmap().isNull() is False
    finally:
        bridge.close()
