"""Spawn-only worker entry point and restartable single-job supervisor."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.process import BaseProcess
from typing import Any, Self

from gyrus.domain.errors import DomainError
from gyrus.domain.result_schema import validate_result_payload

from .inference import FrozenInferenceEngine
from .loader import FrozenWorkerSession, WorkerLoadError
from .protocol import (
    AnalyzeJob,
    AnalyzeSuccess,
    ShutdownWorker,
    WorkerConfig,
    WorkerFailure,
    WorkerReady,
    WorkerResponse,
    WorkerStopped,
)


class WorkerSupervisorError(RuntimeError):
    """The supervisor could not establish or preserve a worker generation."""


def _validate_success(response: AnalyzeSuccess, job: AnalyzeJob, ready: WorkerReady) -> None:
    """Reject a corrupted or mismatched IPC success before any UI can use it."""

    if (
        response.job_id != job.job_id
        or response.case_id != job.case_id
        or response.generation != ready.generation
        or response.diagnostics.load_session_id != ready.load_session_id
    ):
        raise WorkerSupervisorError("WORKER_RESPONSE_MISMATCH")
    validate_result_payload(response.result)
    if (
        response.result["case_id"] != job.case_id
        or response.result["stack_manifest_sha256"] != ready.manifest_sha256
        or response.result["execution"]["device"] != ready.device
    ):
        raise WorkerSupervisorError("WORKER_RESULT_PROVENANCE_MISMATCH")
    if any(event.disclosure_allowed for event in response.events[:-1]):
        raise WorkerSupervisorError("WORKER_PRE_OOD_DISCLOSURE")
    spatial = response.result["spatial"]
    if spatial is None:
        if response.arrays:
            raise WorkerSupervisorError("WORKER_UNDISCLOSED_ARRAY_PRESENT")
        return
    references = {
        spatial["probability_map"]["role"]: spatial["probability_map"],
        spatial["model_mask"]["role"]: spatial["model_mask"],
    }
    payloads = {item.role: item for item in response.arrays}
    if len(response.arrays) != 2 or set(references) != set(payloads):
        raise WorkerSupervisorError("WORKER_ARRAY_MEMBERSHIP_MISMATCH")
    for role, reference in references.items():
        payload = payloads[role]
        if (
            reference["storage_key"] != payload.storage_key
            or tuple(reference["shape"]) != payload.shape
            or reference["dtype"] != payload.dtype
            or reference["sha256"] != payload.sha256
        ):
            raise WorkerSupervisorError("WORKER_ARRAY_REFERENCE_MISMATCH")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def worker_main(
    config: WorkerConfig,
    generation: int,
    request_queue: Any,
    response_queue: Any,
) -> None:
    """Verify, load once, then process one queued request at a time."""

    session: FrozenWorkerSession | None = None
    try:
        session = FrozenWorkerSession.open(config)
        engine = FrozenInferenceEngine(session, generation)
        response_queue.put(
            WorkerReady(
                generation=generation,
                process_id=os.getpid(),
                device=session.device.type,
                manifest_sha256=session.stack.manifest_sha256,
                artifact_count=len(session.stack.artifacts),
                load_session_id=session.load_session_id,
                loaded_at=session.loaded_at,
            )
        )
    except (WorkerLoadError, DomainError) as exc:
        response_queue.put(
            WorkerFailure(
                job_id=None,
                error_code=getattr(exc, "code", "WORKER_STARTUP_FAILED"),
                safe_message=getattr(exc, "safe_message", getattr(exc, "message", str(exc))),
                fatal=True,
                generation=generation,
            )
        )
        return
    except Exception as exc:  # noqa: BLE001 - process boundary must fail closed
        response_queue.put(
            WorkerFailure(
                job_id=None,
                error_code="WORKER_STARTUP_FAILED",
                safe_message=f"Worker startup failed: {type(exc).__name__}",
                fatal=True,
                generation=generation,
            )
        )
        return

    try:
        while True:
            message = request_queue.get()
            if isinstance(message, ShutdownWorker):
                response_queue.put(WorkerStopped(generation))
                return
            if not isinstance(message, AnalyzeJob):
                response_queue.put(
                    WorkerFailure(
                        job_id=None,
                        error_code="WORKER_PROTOCOL_INVALID",
                        safe_message="The worker received an unsupported request.",
                        fatal=True,
                        generation=generation,
                    )
                )
                return
            try:
                response_queue.put(engine.analyze(message))
            except DomainError as exc:
                response_queue.put(
                    WorkerFailure(
                        job_id=message.job_id,
                        error_code=exc.code,
                        safe_message=exc.message,
                        fatal=False,
                        generation=generation,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - process boundary must fail closed
                response_queue.put(
                    WorkerFailure(
                        job_id=message.job_id,
                        error_code="WORKER_INFERENCE_FAILED",
                        safe_message=f"Frozen inference failed: {type(exc).__name__}",
                        fatal=True,
                        generation=generation,
                    )
                )
                return
    finally:
        if session is not None:
            session.close()


@dataclass(frozen=True, slots=True)
class SupervisorSnapshot:
    generation: int
    process_id: int
    device: str
    manifest_sha256: str
    load_session_id: str
    artifact_count: int


class WorkerSupervisor:
    """Own one spawn worker, serialize jobs, and replace failed generations."""

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._context = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._generation = 0
        self._process: BaseProcess | None = None
        self._requests: Any = None
        self._responses: Any = None
        self._ready: WorkerReady | None = None

    @property
    def snapshot(self) -> SupervisorSnapshot | None:
        ready = self._ready
        if ready is None:
            return None
        return SupervisorSnapshot(
            generation=ready.generation,
            process_id=ready.process_id,
            device=ready.device,
            manifest_sha256=ready.manifest_sha256,
            load_session_id=ready.load_session_id,
            artifact_count=ready.artifact_count,
        )

    @property
    def process_id(self) -> int | None:
        return None if self._process is None else self._process.pid

    def _dispose_process(self) -> None:
        process = self._process
        if process is not None:
            process.join(timeout=2.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            process.close()
        for channel in (self._requests, self._responses):
            if channel is not None:
                channel.close()
                channel.join_thread()
        self._process = None
        self._requests = None
        self._responses = None
        self._ready = None

    def start(self) -> SupervisorSnapshot:
        with self._lock:
            return self._start_locked()

    def _start_locked(self) -> SupervisorSnapshot:
        if self._process is not None and self._process.is_alive() and self._ready:
            snapshot = self.snapshot
            assert snapshot is not None
            return snapshot
        self._dispose_process()
        self._generation += 1
        self._requests = self._context.Queue(maxsize=1)
        self._responses = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=worker_main,
            args=(self.config, self._generation, self._requests, self._responses),
            name=f"GyrusWorker-{self._generation}",
            daemon=True,
        )
        self._process.start()
        response = self._receive_until(self.config.startup_timeout_seconds)
        if response is None:
            self._dispose_process()
            raise WorkerSupervisorError("WORKER_STARTUP_TIMEOUT_OR_CRASH")
        if isinstance(response, WorkerFailure):
            self._dispose_process()
            raise WorkerSupervisorError(
                f"{response.error_code}: {response.safe_message}"
            )
        if (
            not isinstance(response, WorkerReady)
            or response.generation != self._generation
            or response.manifest_sha256 != self.config.expected_manifest_sha256
            or response.device != self.config.device.value
        ):
            self._dispose_process()
            raise WorkerSupervisorError("WORKER_STARTUP_PROTOCOL_INVALID")
        self._ready = response
        snapshot = self.snapshot
        assert snapshot is not None
        return snapshot

    def ensure_ready(self) -> SupervisorSnapshot:
        with self._lock:
            if self._process is None or not self._process.is_alive() or self._ready is None:
                return self._start_locked()
            snapshot = self.snapshot
            assert snapshot is not None
            return snapshot

    def analyze(self, job: AnalyzeJob) -> AnalyzeSuccess | WorkerFailure:
        """Submit one job; no second job can enter until its final response exists."""

        with self._lock:
            if self._process is None or not self._process.is_alive() or self._ready is None:
                self._start_locked()
            assert self._requests is not None and self._responses is not None
            try:
                self._requests.put(job, timeout=1.0)
            except queue.Full as exc:
                self._dispose_process()
                raise WorkerSupervisorError("WORKER_REQUEST_QUEUE_FULL") from exc
            response = self._receive_until(self.config.job_timeout_seconds)
            if response is None:
                failed_generation = self._generation
                self._dispose_process()
                self._start_locked()
                return WorkerFailure(
                    job_id=job.job_id,
                    error_code="WORKER_JOB_TIMEOUT_OR_CRASH",
                    safe_message="The worker stopped or timed out and was restarted; the job was not retried.",
                    fatal=True,
                    generation=failed_generation,
                )
            if isinstance(response, AnalyzeSuccess):
                try:
                    assert self._ready is not None
                    _validate_success(response, job, self._ready)
                except (WorkerSupervisorError, DomainError, KeyError, TypeError, AttributeError):
                    failed_generation = self._generation
                    self._dispose_process()
                    self._start_locked()
                    return WorkerFailure(
                        job_id=job.job_id,
                        error_code="WORKER_RESPONSE_MISMATCH",
                        safe_message="The worker response did not match the submitted job.",
                        fatal=True,
                        generation=failed_generation,
                    )
                return response
            if isinstance(response, WorkerFailure):
                if response.job_id != job.job_id or response.generation != self._generation:
                    failed_generation = self._generation
                    self._dispose_process()
                    self._start_locked()
                    return WorkerFailure(
                        job_id=job.job_id,
                        error_code="WORKER_RESPONSE_MISMATCH",
                        safe_message="The worker failure did not match the submitted job.",
                        fatal=True,
                        generation=failed_generation,
                    )
                if response.fatal:
                    self._dispose_process()
                    self._start_locked()
                return response
            self._dispose_process()
            self._start_locked()
            return WorkerFailure(
                job_id=job.job_id,
                error_code="WORKER_RESPONSE_INVALID",
                safe_message="The worker returned an unsupported response.",
                fatal=True,
                generation=self._generation,
            )

    def _receive_until(self, timeout_seconds: float) -> WorkerResponse | None:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                return self._responses.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    return None

    def close(self) -> None:
        with self._lock:
            if self._process is not None and self._process.is_alive() and self._requests:
                try:
                    self._requests.put(ShutdownWorker(), timeout=1.0)
                    if self._responses is not None:
                        self._responses.get(timeout=5.0)
                except (queue.Empty, queue.Full, OSError, EOFError):
                    pass
            self._dispose_process()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
