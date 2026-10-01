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
    assert "Model-predicted label: no_tumor" in window.result_inspector.category.text()


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
