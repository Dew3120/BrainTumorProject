"""Failure-boundary tests use synthetic cases and never open the sealed set."""

from __future__ import annotations

import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gyrus.domain import error_reporting
from gyrus.domain.errors import InputValidationError, PolicyError
from gyrus.domain.history import HistoryRepository
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.gui.input_controller import InputValidationThread
from gyrus.gui.main_window import GyrusMainWindow
from gyrus.gui.pdf_preview import PdfPreviewDialog
from gyrus.gui.state import AppState
from gyrus.gui.work_controller import WorkerBridge
from gyrus.worker import process
from gyrus.worker.protocol import (
    AnalyzeJob,
    DeviceName,
    ShutdownWorker,
    WorkerConfig,
    WorkerFailure,
    WorkerReady,
    WorkerStopped,
)


def _ready() -> SimpleNamespace:
    return SimpleNamespace(
        artifact_count=31, manifest_sha256=EXPECTED_MANIFEST_SHA256, device="cpu"
    )


def _analyzing_window(qtbot) -> GyrusMainWindow:
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    assert window.on_worker_ready(_ready())
    window.machine.enter(AppState.INPUT_VALIDATING, reason="synthetic image")
    window.machine.enter(AppState.READY_TO_ANALYZE, reason="synthetic validation")
    window.machine.enter(AppState.ANALYZING, reason="synthetic job")
    return window


def test_rotating_log_keeps_traceback_out_of_user_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setattr(error_reporting, "_LOG_PATH", None)
    logger = logging.getLogger("gyrus.test.error_reporting")
    monkeypatch.setattr(error_reporting, "_LOGGER", logger)
    try:
        raise PermissionError("private diagnostic detail")
    except PermissionError as exc:
        notice = error_reporting.report_error(
            "HISTORY_WRITE_FAILED", "Case was not saved.", exception=exc
        )
    path = error_reporting.error_log_path()
    assert path == tmp_path / "GyrusResearch" / "logs" / "gyrus.log"
    assert path is not None and path.exists()
    assert "private diagnostic detail" in path.read_text(encoding="utf-8")
    assert "private diagnostic detail" not in notice.copy_details
    assert "HISTORY_WRITE_FAILED" in notice.copy_details
    assert notice.reference in notice.copy_details
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def test_invalid_worker_response_locks_without_raw_detail(qtbot):
    window = _analyzing_window(qtbot)
    window.on_job_failed("raw traceback: private path")
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert "raw traceback" not in window.workspace_message.text()
    assert not window.choose_button.isEnabled()
    assert not window.run_button.isEnabled()
    assert window.copy_error_button.isVisible()
    assert window.close_error_button.isVisible()


def test_worker_protocol_mismatch_locks_and_copies_safe_reference(qtbot):
    window = _analyzing_window(qtbot)
    response = WorkerFailure(
        None, "WORKER_RESPONSE_MISMATCH", "The worker response did not match the job.", True, 1
    )
    window.on_job_failed(response)
    window._copy_error_details()
    copied = QApplication.clipboard().text()
    assert "WORKER_RESPONSE_MISMATCH" in copied
    assert "Reference:" in copied
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert not window.history_button.isEnabled()


def test_rejected_image_is_recoverable_and_never_discloses_result(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.machine.enter(AppState.INPUT_VALIDATING, reason="bad image")
    window._on_input_rejected("INPUT_DECODE_FAILED", "Choose a readable PNG or JPEG.")
    assert window.machine.state == AppState.RECOVERABLE_ERROR
    assert window.choose_button.isEnabled()
    assert window._pending_result is None
    assert "INPUT_DECODE_FAILED" in window.workspace_message.text()


def test_unexpected_preview_exception_is_logged_and_not_disclosed(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def fail_decode(_path):
        raise RuntimeError("private preview traceback detail")

    monkeypatch.setattr("gyrus.gui.input_controller.validate_and_decode_image", fail_decode)
    worker = InputValidationThread("synthetic.png")
    rejected = []
    worker.rejected.connect(lambda code, message: rejected.append((code, message)))
    worker.run()
    assert rejected == [("INPUT_PREVIEW_FAILED", "The image preview could not be created.")]


def test_artifact_mismatch_names_artifact_and_disables_analysis(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_boot_failed(
        "MANIFEST_ARTIFACT_HASH_MISMATCH",
        "MANIFEST_ARTIFACT_HASH_MISMATCH: models/example.pkl: expected x, got y",
    )
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert "models/example.pkl" in window.workspace_message.text()
    assert not window.retry_button.isVisible()
    assert not window.choose_button.isEnabled()


def test_boot_device_failure_offers_explicit_retry_not_fallback(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_boot_failed("WORKER_CUDA_UNAVAILABLE", "CUDA unavailable")
    assert window.machine.state == AppState.BOOT_BLOCKED
    assert "--device cpu" in window.workspace_message.text()
    assert window.retry_button.isVisible()
    assert not window.choose_button.isEnabled()


def test_ood_runtime_load_failure_blocks_retry_and_disclosure(qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_boot_failed(
        "WORKER_RUNTIME_IMPORT_FAILED", "OOD model component could not load"
    )
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert not window.retry_button.isVisible()
    assert not window.choose_button.isEnabled()


def test_ood_policy_failure_never_becomes_recoverable_result(qtbot):
    window = _analyzing_window(qtbot)
    window.on_job_failed(
        WorkerFailure(None, "OOD_POLICY_INVALID", "OOD policy could not be evaluated.", False, 1)
    )
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert window._pending_result is None


def test_pdf_render_error_keeps_no_partial_result(monkeypatch, qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())

    def fail_render(_payload):
        raise ValueError("private PDF renderer detail")

    monkeypatch.setattr("gyrus.gui.main_window.render_pdf", fail_render)
    window._preview_pdf(object())
    qtbot.waitUntil(lambda: window._local_task is None, timeout=20_000)
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert "private PDF renderer detail" not in window.workspace_message.text()


def test_pdf_preview_failure_keeps_case_available(monkeypatch, qtbot):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    monkeypatch.setattr("gyrus.gui.main_window.render_pdf", lambda _payload: b"%PDF-synthetic")

    def fail_viewer(*_args):
        raise RuntimeError("private QPdfView detail")

    monkeypatch.setattr("gyrus.gui.main_window.PdfPreviewDialog", fail_viewer)
    window._preview_pdf(SimpleNamespace(model=SimpleNamespace(default_filename="case.pdf")))
    qtbot.waitUntil(lambda: window._local_task is None, timeout=20_000)
    assert window.machine.state == AppState.READY_EMPTY
    assert "PDF preview could not open" in window.notice.text()
    assert "private QPdfView detail" not in window.notice.text()


def test_pdf_save_permission_error_keeps_preview_and_hides_raw_detail(
    tmp_path, monkeypatch, qtbot
):
    from PySide6.QtWidgets import QDialog, QFileDialog, QMessageBox

    dialog = PdfPreviewDialog.__new__(PdfPreviewDialog)
    QDialog.__init__(dialog)
    qtbot.addWidget(dialog)
    dialog._pdf_bytes = b"synthetic bytes"
    dialog.default_filename = "gyrus-research-summary_case.pdf"
    dialog.saved_path = None
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    destination = tmp_path / "GyrusResearch" / "exports" / dialog.default_filename
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *_args: (str(destination), ""))

    def fail_save(*_args, **_kwargs):
        raise PermissionError("private destination detail")

    shown = []
    monkeypatch.setattr("gyrus.gui.pdf_preview.save_pdf_atomic", fail_save)
    monkeypatch.setattr(QMessageBox, "warning", lambda _parent, title, message: shown.append((title, message)))
    dialog._choose_and_save()
    assert shown and shown[0][0] == "PDF was not saved"
    assert "private destination detail" not in shown[0][1]
    assert dialog.saved_path is None
    assert not destination.exists()


def test_corrupt_history_index_is_preserved_for_investigation(tmp_path):
    database = tmp_path / "history.db"
    original = b"not a sqlite database; preserve this evidence"
    database.write_bytes(original)
    repository = HistoryRepository(database, tmp_path / "cases")
    with pytest.raises(sqlite3.DatabaseError):
        repository.initialize()
    assert database.read_bytes() == original
    assert not list(tmp_path.glob("history.db.quarantine*"))


def test_corrupt_history_index_blocks_ui_without_deleting_index(tmp_path, monkeypatch, qtbot):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    database = tmp_path / "GyrusResearch" / "history.db"
    database.parent.mkdir(parents=True)
    original = b"not a sqlite database"
    database.write_bytes(original)
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.open_history()
    assert window.machine.state == AppState.INTEGRITY_ERROR_LOCKED
    assert database.read_bytes() == original
    assert not window.history_button.isEnabled()


def test_history_write_permission_failure_never_claims_save(monkeypatch, qtbot):
    from PySide6.QtWidgets import QMessageBox

    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.machine.enter(AppState.INPUT_VALIDATING, reason="synthetic")
    window.machine.enter(AppState.READY_TO_ANALYZE, reason="synthetic")
    window.machine.enter(AppState.ANALYZING, reason="synthetic")
    window.machine.enter(AppState.RESULT_STANDARD, reason="synthetic")
    monkeypatch.setattr(window, "_history_payload", lambda: object())
    monkeypatch.setattr("gyrus.gui.main_window.planned_files", lambda _payload: ("source.png",))
    monkeypatch.setattr(QMessageBox, "question", lambda *_args: QMessageBox.StandardButton.Yes)

    def fail_save(*_args):
        raise PermissionError("private index location")

    monkeypatch.setattr("gyrus.gui.main_window.save_history_bundle", fail_save)
    window.save_to_history()
    qtbot.waitUntil(lambda: window._local_task is None, timeout=20_000)
    assert window.machine.state == AppState.RESULT_STANDARD
    assert "Case was not saved" in window.notice.text()
    assert "private index location" not in window.notice.text()
    assert not window._saved_history_ids


def test_edited_layer_permission_failure_keeps_editor_open(monkeypatch, qtbot, tmp_path):
    window = GyrusMainWindow()
    qtbot.addWidget(window)
    window.show()
    window.on_worker_ready(_ready())
    window.machine.enter(AppState.INPUT_VALIDATING, reason="synthetic")
    window.machine.enter(AppState.READY_TO_ANALYZE, reason="synthetic")
    window.machine.enter(AppState.ANALYZING, reason="synthetic")
    window.machine.enter(AppState.RESULT_STANDARD, reason="synthetic")
    window.machine.enter(AppState.EDITING_MASK, reason="synthetic")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def fail_export(_directory):
        raise PermissionError("private edited-layer location")

    window._edit_session = SimpleNamespace(export=fail_export)
    window.save_edited_layer()
    assert window.machine.state == AppState.EDITING_MASK
    assert "Edited layer was not saved" in window.notice.text()
    assert "private edited-layer location" not in window.notice.text()


@pytest.mark.parametrize(
    ("failure", "expected_fatal"),
    [
        (InputValidationError("INPUT_DECODE_FAILED", "Bad input"), False),
        (PolicyError("OOD_POLICY_INVALID", "OOD calculation failed"), True),
    ],
)
def test_worker_contains_domain_failure_and_classifies_integrity(
    monkeypatch, failure, expected_fatal
):
    class Session:
        device = SimpleNamespace(type="cpu")
        stack = SimpleNamespace(
            manifest_sha256=EXPECTED_MANIFEST_SHA256, artifacts=[None] * 31
        )
        load_session_id = str(uuid4())
        loaded_at = datetime.now().astimezone().isoformat()

        def close(self):
            pass

    class Engine:
        def __init__(self, _session, _generation):
            pass

        def analyze(self, _job):
            raise failure

    monkeypatch.setattr(process.FrozenWorkerSession, "open", lambda _config: Session())
    monkeypatch.setattr(process, "FrozenInferenceEngine", Engine)
    monkeypatch.setattr(process, "configure_error_log", lambda: None)
    monkeypatch.setattr(process, "report_error", lambda *args, **kwargs: None)
    job = AnalyzeJob(
        str(uuid4()), str(uuid4()), str(Path.cwd() / "synthetic.png"),
        datetime.now().astimezone().isoformat(),
    )
    requests = queue.Queue()
    responses = queue.Queue()
    requests.put(job)
    requests.put(ShutdownWorker())
    process.worker_main(WorkerConfig(str(Path.cwd()), DeviceName.CPU), 1, requests, responses)
    assert isinstance(responses.get_nowait(), WorkerReady)
    error = responses.get_nowait()
    assert isinstance(error, WorkerFailure)
    assert error.job_id == job.job_id
    assert error.fatal is expected_fatal
    if expected_fatal:
        assert responses.empty()
    else:
        assert isinstance(responses.get_nowait(), WorkerStopped)


def test_worker_gpu_oom_never_silently_retries_on_cpu(monkeypatch):
    class OutOfMemoryError(RuntimeError):
        pass

    class Session:
        device = SimpleNamespace(type="cuda")
        stack = SimpleNamespace(
            manifest_sha256=EXPECTED_MANIFEST_SHA256, artifacts=[None] * 31
        )
        load_session_id = str(uuid4())
        loaded_at = datetime.now().astimezone().isoformat()

        def close(self):
            pass

    class Engine:
        def __init__(self, _session, _generation):
            pass

        def analyze(self, _job):
            raise OutOfMemoryError("private CUDA detail")

    monkeypatch.setattr(process.FrozenWorkerSession, "open", lambda _config: Session())
    monkeypatch.setattr(process, "FrozenInferenceEngine", Engine)
    monkeypatch.setattr(process, "configure_error_log", lambda: None)
    monkeypatch.setattr(process, "report_error", lambda *args, **kwargs: None)
    job = AnalyzeJob(
        str(uuid4()), str(uuid4()), str(Path.cwd() / "synthetic.png"),
        datetime.now().astimezone().isoformat(),
    )
    requests = queue.Queue()
    responses = queue.Queue()
    requests.put(job)
    process.worker_main(WorkerConfig(str(Path.cwd()), DeviceName.CUDA), 1, requests, responses)
    assert isinstance(responses.get_nowait(), WorkerReady)
    error = responses.get_nowait()
    assert isinstance(error, WorkerFailure)
    assert error.error_code == "WORKER_CUDA_OUT_OF_MEMORY"
    assert error.fatal
    assert "--device cpu" in error.safe_message
    assert "private CUDA detail" not in error.safe_message


def test_closing_during_inference_cancels_without_retry(qtbot):
    entered = threading.Event()
    cancelled = threading.Event()
    calls = []

    class Supervisor:
        def start(self):
            return _ready()

        def analyze(self, job):
            calls.append(job.job_id)
            entered.set()
            assert cancelled.wait(5)
            return WorkerFailure(job.job_id, "WORKER_CANCELLED_ON_CLOSE", "No result.", True, 1)

        def request_shutdown(self):
            cancelled.set()

        def close(self):
            cancelled.set()

    bridge = WorkerBridge("synthetic-stack", "cpu", supervisor_factory=lambda *_: Supervisor())
    try:
        with qtbot.waitSignal(bridge.verified, timeout=3_000):
            bridge.start()
        bridge.submit_image("synthetic.png")
        assert entered.wait(3)
        started = time.monotonic()
        bridge.close()
        assert time.monotonic() - started < 3
        assert not bridge.controller_thread.isRunning()
        assert len(calls) == 1
    finally:
        if bridge.controller_thread.isRunning():
            bridge.close()


def test_worker_timeout_restarts_generation_without_resubmitting_job(monkeypatch):
    from gyrus.worker.process import WorkerSupervisor

    supervisor = WorkerSupervisor(WorkerConfig(str(Path.cwd()), DeviceName.CPU))
    supervisor._generation = 1
    supervisor._ready = WorkerReady(
        1, 1234, "cpu", EXPECTED_MANIFEST_SHA256, 31,
        str(uuid4()), datetime.now().astimezone().isoformat(),
    )
    supervisor._process = SimpleNamespace(is_alive=lambda: True)
    supervisor._requests = queue.Queue()
    supervisor._responses = queue.Queue()
    monkeypatch.setattr(supervisor, "_receive_until", lambda _timeout: None)

    def dispose():
        supervisor._process = None
        supervisor._ready = None

    def restart():
        supervisor._generation += 1
        return SimpleNamespace(generation=2)

    monkeypatch.setattr(supervisor, "_dispose_process", dispose)
    monkeypatch.setattr(supervisor, "_start_locked", restart)
    job = AnalyzeJob(
        str(uuid4()), str(uuid4()), str(Path.cwd() / "synthetic.png"),
        datetime.now().astimezone().isoformat(),
    )
    response = supervisor.analyze(job)
    assert isinstance(response, WorkerFailure)
    assert response.error_code == "WORKER_JOB_TIMEOUT_OR_CRASH"
    assert response.generation == 1
    assert supervisor._generation == 2
    assert supervisor._requests is not None
    assert supervisor._requests.qsize() == 1
