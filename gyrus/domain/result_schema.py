"""Versioned immutable records and validation for worker-to-UI results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from .constants import CLASS_NAMES, SEGMENTATION_THRESHOLD_TEXT, DisplayStatus
from .errors import ResultValidationError
from .input_contract import InputDescriptor
from .policy import FinalPolicyDecision, OodDecision, ReliabilityDecision


def _invalid(code: str, message: str) -> ResultValidationError:
    return ResultValidationError(code, message)


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise _invalid("RESULT_SHA256_INVALID", f"{field} must be lowercase SHA256")
    return value


def _timestamp(value: str, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise _invalid("RESULT_TIMESTAMP_INVALID", f"{field}: {value!r}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _invalid("RESULT_TIMESTAMP_INVALID", f"{field} must include timezone")
    return parsed


@dataclass(frozen=True, slots=True)
class ArrayReference:
    """Content-addressed array output owned by the worker transport layer."""

    role: str
    shape: tuple[int, ...]
    dtype: str
    sha256: str
    storage_key: str

    def __post_init__(self) -> None:
        if not self.role or not self.storage_key:
            raise _invalid("RESULT_ARRAY_REFERENCE_INVALID", "role and storage_key required")
        if not self.shape or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.shape):
            raise _invalid("RESULT_ARRAY_SHAPE_INVALID", repr(self.shape))
        if not self.dtype:
            raise _invalid("RESULT_ARRAY_DTYPE_INVALID", repr(self.dtype))
        _sha256(self.sha256, f"array:{self.role}")


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    probabilities: tuple[float, float, float, float]
    predicted_category: str

    def __post_init__(self) -> None:
        if self.predicted_category not in CLASS_NAMES:
            raise _invalid("RESULT_CATEGORY_INVALID", repr(self.predicted_category))
        if len(self.probabilities) != len(CLASS_NAMES):
            raise _invalid("RESULT_PROBABILITY_COUNT_INVALID", repr(self.probabilities))
        values = tuple(float(value) for value in self.probabilities)
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise _invalid("RESULT_PROBABILITY_INVALID", repr(values))
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise _invalid("RESULT_PROBABILITY_SUM_INVALID", repr(sum(values)))
        expected = CLASS_NAMES[max(range(4), key=values.__getitem__)]
        if self.predicted_category != expected:
            raise _invalid(
                "RESULT_CATEGORY_ARGMAX_MISMATCH",
                f"expected {expected}, got {self.predicted_category}",
            )


@dataclass(frozen=True, slots=True)
class MaskStatistics:
    foreground_pixels: int
    total_pixels: int
    area_fraction: float
    component_count: int
    bounding_box_xywh: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        if self.total_pixels <= 0 or not 0 <= self.foreground_pixels <= self.total_pixels:
            raise _invalid("RESULT_MASK_PIXEL_COUNT_INVALID", repr(self))
        expected_fraction = self.foreground_pixels / self.total_pixels
        if not math.isclose(self.area_fraction, expected_fraction, abs_tol=1e-12):
            raise _invalid("RESULT_MASK_AREA_FRACTION_INVALID", repr(self.area_fraction))
        if self.component_count < 0:
            raise _invalid("RESULT_MASK_COMPONENT_COUNT_INVALID", repr(self.component_count))
        if self.foreground_pixels == 0 and self.bounding_box_xywh is not None:
            raise _invalid("RESULT_MASK_BOUNDING_BOX_INVALID", "empty mask has a box")
        if self.foreground_pixels > 0:
            box = self.bounding_box_xywh
            if box is None or len(box) != 4 or any(value < 0 for value in box):
                raise _invalid("RESULT_MASK_BOUNDING_BOX_INVALID", repr(box))
            if box[2] <= 0 or box[3] <= 0:
                raise _invalid("RESULT_MASK_BOUNDING_BOX_INVALID", repr(box))


@dataclass(frozen=True, slots=True)
class SpatialResult:
    probability_map: ArrayReference
    model_mask: ArrayReference
    statistics: MaskStatistics
    segmentation_threshold: str = SEGMENTATION_THRESHOLD_TEXT

    def __post_init__(self) -> None:
        if self.segmentation_threshold != SEGMENTATION_THRESHOLD_TEXT:
            raise _invalid("RESULT_SEGMENTATION_THRESHOLD_DRIFT", self.segmentation_threshold)
        if self.probability_map.shape != (128, 128) or self.model_mask.shape != (128, 128):
            raise _invalid("RESULT_SPATIAL_SHAPE_INVALID", "expected two 128x128 arrays")


@dataclass(frozen=True, slots=True)
class ExecutionInfo:
    device: str
    started_at: str
    completed_at: str
    duration_ms: int

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"}:
            raise _invalid("RESULT_DEVICE_INVALID", repr(self.device))
        started = _timestamp(self.started_at, "started_at")
        completed = _timestamp(self.completed_at, "completed_at")
        if completed < started or self.duration_ms < 0:
            raise _invalid("RESULT_EXECUTION_TIME_INVALID", repr(self))


@dataclass(frozen=True, slots=True)
class AnalysisResult:
    """Complete disclosure-safe result assembled by the domain orchestrator."""

    schema_version: int
    case_id: str
    stack_manifest_sha256: str
    input: InputDescriptor
    execution: ExecutionInfo
    ood: OodDecision
    classification: ClassificationResult | None
    spatial: SpatialResult | None
    reliability: ReliabilityDecision | None
    final_policy: FinalPolicyDecision
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise _invalid("RESULT_SCHEMA_VERSION_INVALID", repr(self.schema_version))
        try:
            UUID(self.case_id)
        except ValueError as exc:
            raise _invalid("RESULT_CASE_ID_INVALID", self.case_id) from exc
        _sha256(self.stack_manifest_sha256, "stack_manifest_sha256")
        if not self.ood.valid:
            if self.classification is not None or self.spatial is not None or self.reliability is not None:
                raise _invalid(
                    "RESULT_PRE_OOD_DISCLOSURE",
                    "invalid OOD output must withhold classification and segmentation",
                )
            if self.final_policy.status != DisplayStatus.REVIEW_REQUIRED:
                raise _invalid("RESULT_POLICY_INCONSISTENT", "invalid OOD cannot be STANDARD")
            return
        if self.classification is None or self.reliability is None:
            raise _invalid("RESULT_DISCLOSURE_INCOMPLETE", "classification and policy required")
        if self.classification.predicted_category == "no_tumor":
            if self.spatial is not None:
                raise _invalid("RESULT_NO_TUMOR_SPATIAL_PRESENT", "spatial output must be withheld")
            if self.reliability.score is not None:
                raise _invalid("RESULT_NO_TUMOR_RELIABILITY_PRESENT", "score must be null")
        elif self.spatial is None:
            raise _invalid("RESULT_TUMOR_SPATIAL_MISSING", "tumor category requires spatial output")
        if self.final_policy.status == DisplayStatus.STANDARD and (
            self.ood.status.value != "ACCEPT"
            or self.reliability.status != DisplayStatus.STANDARD
        ):
            raise _invalid("RESULT_POLICY_INCONSISTENT", "STANDARD requires both gates")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible object with enum values normalized to strings."""

        return _normalize(asdict(self))


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    return value


def validate_result_payload(payload: Mapping[str, Any]) -> None:
    """Validate the stable top-level serialized contract before persistence/export."""

    required = {
        "schema_version",
        "case_id",
        "stack_manifest_sha256",
        "input",
        "execution",
        "ood",
        "classification",
        "spatial",
        "reliability",
        "final_policy",
        "warnings",
    }
    if set(payload) != required:
        raise _invalid(
            "RESULT_SERIALIZED_KEYS_INVALID",
            f"missing={sorted(required - set(payload))}; extra={sorted(set(payload) - required)}",
        )
    if payload.get("schema_version") != 1:
        raise _invalid("RESULT_SCHEMA_VERSION_INVALID", repr(payload.get("schema_version")))
    try:
        UUID(str(payload.get("case_id")))
    except ValueError as exc:
        raise _invalid("RESULT_CASE_ID_INVALID", repr(payload.get("case_id"))) from exc
    manifest_hash = payload.get("stack_manifest_sha256")
    if not isinstance(manifest_hash, str):
        raise _invalid("RESULT_SHA256_INVALID", "stack_manifest_sha256 must be text")
    _sha256(manifest_hash, "stack_manifest_sha256")
    final_policy = payload.get("final_policy")
    if not isinstance(final_policy, Mapping) or final_policy.get("status") not in {
        DisplayStatus.STANDARD.value,
        DisplayStatus.REVIEW_REQUIRED.value,
    }:
        raise _invalid("RESULT_POLICY_SERIALIZATION_INVALID", repr(final_policy))
    warnings = payload.get("warnings")
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        raise _invalid("RESULT_WARNINGS_INVALID", repr(warnings))
