"""Typed, pickle-safe messages exchanged with the isolated Gyrus worker."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np

from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256

WORKER_PROTOCOL_VERSION = 1


class DeviceName(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class WorkerStage(StrEnum):
    PREPROCESS_INPUT = "PREPROCESS_INPUT"
    SILENT_CLASSIFIER = "SILENT_CLASSIFIER"
    SILENT_V1_SEGMENTER = "SILENT_V1_SEGMENTER"
    OOD_O02 = "OOD_O02"
    OOD_O03 = "OOD_O03"
    OOD_O04 = "OOD_O04"
    OOD_O05 = "OOD_O05"
    OOD_POLICY = "OOD_POLICY"
    FULL_E08_C03_R02 = "FULL_E08_C03_R02"
    RESULT_ASSEMBLY = "RESULT_ASSEMBLY"


class WorkerProtocolError(ValueError):
    """A local IPC message violates the registered worker contract."""


def _uuid(value: str, field: str) -> str:
    try:
        return str(UUID(value))
    except (TypeError, ValueError, AttributeError) as exc:
        raise WorkerProtocolError(f"{field} must be a canonical UUID") from exc


def _timestamp(value: str, field: str) -> str:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise WorkerProtocolError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise WorkerProtocolError(f"{field} must include a timezone")
    return value


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise WorkerProtocolError(f"{field} must be a lowercase SHA256")
    return value


@dataclass(frozen=True, slots=True)
class WorkerConfig:
    stack_root: str
    device: DeviceName
    expected_manifest_sha256: str = EXPECTED_MANIFEST_SHA256
    startup_timeout_seconds: float = 45.0
    job_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if not self.stack_root or "\x00" in self.stack_root:
            raise WorkerProtocolError("stack_root must be a nonempty local path")
        if not isinstance(self.device, DeviceName):
            try:
                object.__setattr__(self, "device", DeviceName(self.device))
            except ValueError as exc:
                raise WorkerProtocolError("device must be explicitly cpu or cuda") from exc
        _sha256(self.expected_manifest_sha256, "expected_manifest_sha256")
        for name, value in (
            ("startup_timeout_seconds", self.startup_timeout_seconds),
            ("job_timeout_seconds", self.job_timeout_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise WorkerProtocolError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class AnalyzeJob:
    job_id: str
    case_id: str
    image_path: str
    submitted_at: str
    protocol_version: int = WORKER_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != WORKER_PROTOCOL_VERSION:
            raise WorkerProtocolError("unsupported worker protocol version")
        _uuid(self.job_id, "job_id")
        _uuid(self.case_id, "case_id")
        _timestamp(self.submitted_at, "submitted_at")
        if not self.image_path or "\x00" in self.image_path:
            raise WorkerProtocolError("image_path must be a nonempty local path")


@dataclass(frozen=True, slots=True)
class ShutdownWorker:
    protocol_version: int = WORKER_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class WorkerEvent:
    stage: WorkerStage
    completed_at: str
    disclosure_allowed: bool
    detail: str

    def __post_init__(self) -> None:
        _timestamp(self.completed_at, "completed_at")
        if not self.detail:
            raise WorkerProtocolError("worker event detail is required")


@dataclass(frozen=True, slots=True)
class ArrayPayload:
    role: str
    storage_key: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    data: bytes

    def __post_init__(self) -> None:
        if not self.role or not self.storage_key:
            raise WorkerProtocolError("array role and storage_key are required")
        if not self.shape or any(type(item) is not int or item <= 0 for item in self.shape):
            raise WorkerProtocolError("array shape must contain positive integers")
        if self.dtype not in {"float64", "uint8"}:
            raise WorkerProtocolError(f"unsupported array dtype: {self.dtype}")
        _sha256(self.sha256, "array sha256")
        expected_bytes = math.prod(self.shape) * np.dtype(self.dtype).itemsize
        if len(self.data) != expected_bytes:
            raise WorkerProtocolError("array payload byte count does not match shape")
        if hashlib.sha256(self.data).hexdigest() != self.sha256:
            raise WorkerProtocolError("array payload hash mismatch")


@dataclass(frozen=True, slots=True)
class WorkerDiagnostics:
    stage_order: tuple[str, ...]
    ood_components: tuple[tuple[str, float], ...]
    normalized_ood_components: tuple[tuple[str, float], ...]
    route: str | None
    raw_probability_sha256: str | None
    load_session_id: str

    def __post_init__(self) -> None:
        expected = tuple(stage.value for stage in WorkerStage)
        if self.stage_order != expected:
            raise WorkerProtocolError("worker stage order differs from the registered order")
        _uuid(self.load_session_id, "load_session_id")
        for group in (self.ood_components, self.normalized_ood_components):
            if any(not math.isfinite(float(value)) for _, value in group):
                raise WorkerProtocolError("OOD diagnostics must be finite")
        if self.raw_probability_sha256 is not None:
            _sha256(self.raw_probability_sha256, "raw_probability_sha256")


@dataclass(frozen=True, slots=True)
class WorkerReady:
    generation: int
    process_id: int
    device: str
    manifest_sha256: str
    artifact_count: int
    load_session_id: str
    loaded_at: str

    def __post_init__(self) -> None:
        if self.generation < 1 or self.process_id <= 0 or self.artifact_count != 31:
            raise WorkerProtocolError("invalid worker readiness metadata")
        if self.device not in {item.value for item in DeviceName}:
            raise WorkerProtocolError("invalid ready device")
        _sha256(self.manifest_sha256, "manifest_sha256")
        _uuid(self.load_session_id, "load_session_id")
        _timestamp(self.loaded_at, "loaded_at")


@dataclass(frozen=True, slots=True)
class AnalyzeSuccess:
    job_id: str
    case_id: str
    result: dict[str, Any]
    arrays: tuple[ArrayPayload, ...]
    events: tuple[WorkerEvent, ...]
    diagnostics: WorkerDiagnostics
    generation: int

    def __post_init__(self) -> None:
        _uuid(self.job_id, "job_id")
        _uuid(self.case_id, "case_id")
        if self.generation < 1:
            raise WorkerProtocolError("generation must be positive")
        if not self.events or self.events[-1].stage != WorkerStage.RESULT_ASSEMBLY:
            raise WorkerProtocolError("a success response requires a complete stage trace")


@dataclass(frozen=True, slots=True)
class WorkerFailure:
    job_id: str | None
    error_code: str
    safe_message: str
    fatal: bool
    generation: int

    def __post_init__(self) -> None:
        if self.job_id is not None:
            _uuid(self.job_id, "job_id")
        if not self.error_code or not self.safe_message or self.generation < 1:
            raise WorkerProtocolError("incomplete worker failure")


@dataclass(frozen=True, slots=True)
class WorkerStopped:
    generation: int


WorkerRequest = AnalyzeJob | ShutdownWorker
WorkerResponse = WorkerReady | AnalyzeSuccess | WorkerFailure | WorkerStopped


def canonical_local_path(value: str) -> Path:
    """Resolve a protocol path without accepting a URI or NUL byte."""

    if "://" in value or "\x00" in value:
        raise WorkerProtocolError("only local filesystem paths are accepted")
    return Path(value).expanduser()
