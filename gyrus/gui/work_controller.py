"""Qt thread bridge: frozen worker startup and jobs never run on the UI thread."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot

from gyrus.domain.error_reporting import report_error


def _default_supervisor_factory(stack_root: str, device: str) -> Any:
    # Importing torch and the frozen runtime happens only inside QThread.
    from gyrus.worker.process import WorkerSupervisor
    from gyrus.worker.protocol import DeviceName, WorkerConfig

    return WorkerSupervisor(WorkerConfig(stack_root, DeviceName(device)))


class _WorkerService(QObject):
    verified = Signal(object)
    boot_failed = Signal(str, str)
    job_finished = Signal(object)
    job_failed = Signal(object)
    stopped = Signal()

    def __init__(
        self,
        stack_root: str,
        device: str,
        factory: Callable[[str, str], Any],
    ) -> None:
        super().__init__()
        self._stack_root = stack_root
        self._device = device
        self._factory = factory
        self._supervisor: Any = None

    @Slot()
    def initialize(self) -> None:
        try:
            if self._supervisor is not None:
                self._supervisor.close()
            self._supervisor = self._factory(self._stack_root, self._device)
            ready = self._supervisor.start()
        except Exception as exc:  # noqa: BLE001 - Qt boundary must report failures
            description = str(exc)
            if self._supervisor is not None:
                try:
                    self._supervisor.close()
                except Exception as cleanup_exc:  # noqa: BLE001 - report both failures
                    description += f"; cleanup failed: {type(cleanup_exc).__name__}"
                finally:
                    self._supervisor = None
            code = description.split(":", 1)[0]
            report_error(code, "Worker startup failed before verification.", exception=exc,
                         blocking=True)
            self.boot_failed.emit(code, description)
        else:
            self.verified.emit(ready)

    @Slot(str)
    def analyze(self, image_path: str) -> None:
        if self._supervisor is None:
            self.job_failed.emit("WORKER_NOT_READY")
            return
        try:
            from gyrus.worker.protocol import AnalyzeJob, AnalyzeSuccess

            job = AnalyzeJob(
                job_id=str(uuid4()),
                case_id=str(uuid4()),
                image_path=image_path,
                submitted_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
            )
            response = self._supervisor.analyze(job)
            if isinstance(response, AnalyzeSuccess):
                self.job_finished.emit(response)
            else:
                self.job_failed.emit(response)
        except Exception as exc:  # noqa: BLE001 - Qt boundary must report failures
            report_error("WORKER_JOB_FAILED", "Analysis stopped before a result was produced.",
                         exception=exc, blocking=True)
            self.job_failed.emit("WORKER_JOB_FAILED")

    @Slot()
    def shutdown(self) -> None:
        try:
            if self._supervisor is not None:
                self._supervisor.close()
        finally:
            self._supervisor = None
            self.stopped.emit()

    def interrupt_for_close(self) -> None:
        """Signal the supervisor from the GUI thread; it only sets an event and terminates IPC."""

        supervisor = self._supervisor
        if supervisor is not None:
            supervisor.request_shutdown()


class WorkerBridge(QObject):
    """Expose only readiness and complete final results to the Qt main thread."""

    verified = Signal(object)
    boot_failed = Signal(str, str)
    job_finished = Signal(object)
    job_failed = Signal(object)
    _retry_requested = Signal()
    _job_requested = Signal(str)
    _stop_requested = Signal()

    def __init__(
        self,
        stack_root: str | Path,
        device: str,
        *,
        supervisor_factory: Callable[[str, str], Any] | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        if device not in {"cpu", "cuda"}:
            raise ValueError("Select cpu or cuda explicitly")
        self._thread = QThread(self)
        self._thread.setObjectName("GyrusWorkerControllerThread")
        self._service = _WorkerService(
            str(Path(stack_root).resolve()),
            device,
            supervisor_factory or _default_supervisor_factory,
        )
        self._service.moveToThread(self._thread)
        self._thread.started.connect(self._service.initialize)
        self._retry_requested.connect(self._service.initialize, Qt.QueuedConnection)
        self._job_requested.connect(self._service.analyze, Qt.QueuedConnection)
        self._stop_requested.connect(self._service.shutdown, Qt.QueuedConnection)
        self._service.stopped.connect(self._thread.quit, Qt.DirectConnection)
        self._thread.finished.connect(self._service.deleteLater)
        self._service.verified.connect(self._on_verified)
        self._service.boot_failed.connect(self._on_boot_failed)
        self._service.job_finished.connect(self._on_job_finished)
        self._service.job_failed.connect(self._on_job_failed)
        self._ready = False
        self._busy = False

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def controller_thread(self) -> QThread:
        return self._thread

    def start(self) -> None:
        if self._thread.isRunning():
            raise RuntimeError("Worker controller already started")
        self._thread.start()

    def retry_verification(self) -> None:
        if not self._thread.isRunning() or self._ready or self._busy:
            raise RuntimeError("Verification retry is unavailable in this state")
        self._retry_requested.emit()

    def submit_image(self, image_path: str | Path) -> None:
        if not self._ready or self._busy:
            raise RuntimeError("Worker must be ready and idle before a job is submitted")
        self._busy = True
        self._job_requested.emit(str(image_path))

    def close(self) -> None:
        if not self._thread.isRunning():
            return
        # Qt cannot service a queued shutdown while analyze() occupies its worker thread.
        # Interrupt only the isolated process; the queued service cleanup then runs normally.
        if self._busy:
            self._service.interrupt_for_close()
        self._stop_requested.emit()
        if not self._thread.wait(60_000):
            raise RuntimeError("Worker controller did not stop within 60 seconds")
        self._ready = False
        self._busy = False

    @Slot(object)
    def _on_verified(self, snapshot: object) -> None:
        self._ready = True
        self.verified.emit(snapshot)

    @Slot(str, str)
    def _on_boot_failed(self, code: str, description: str) -> None:
        self._ready = False
        self.boot_failed.emit(code, description)

    @Slot(object)
    def _on_job_finished(self, response: object) -> None:
        self._busy = False
        self.job_finished.emit(response)

    @Slot(object)
    def _on_job_failed(self, response: object) -> None:
        self._busy = False
        self.job_failed.emit(response)
