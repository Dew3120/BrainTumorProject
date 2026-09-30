"""Non-destructive, hash-chained provenance for manual mask edits."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

import numpy as np

from .errors import ProvenanceError


class MaskEditTool(StrEnum):
    BRUSH_ADD = "BRUSH_ADD"
    BRUSH_REMOVE = "BRUSH_REMOVE"
    UNDO = "UNDO"
    REDO = "REDO"
    RESET_TO_MODEL = "RESET_TO_MODEL"


def mask_sha256(mask: np.ndarray) -> str:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ProvenanceError("EDIT_MASK_INVALID", "Mask must be a 2D bool or uint8 array.")
    normalized = np.ascontiguousarray(value.astype(np.uint8) > 0, dtype=np.uint8)
    return hashlib.sha256(normalized.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class MaskEditRecord:
    sequence: int
    timestamp: str
    tool: MaskEditTool
    before_sha256: str
    after_sha256: str
    added_pixels: int
    removed_pixels: int
    parameters: tuple[tuple[str, str | int | float | bool | None], ...]


@dataclass(frozen=True, slots=True)
class MaskEditProvenance:
    case_id: str
    shape: tuple[int, int]
    model_mask_sha256: str
    current_mask_sha256: str
    edited: bool
    records: tuple[MaskEditRecord, ...]
    reliability_scope: str = "ORIGINAL_MODEL_MASK_ONLY"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for row in value["records"]:
            row["tool"] = row["tool"].value
            row["parameters"] = dict(row["parameters"])
        return value


def _normalize_mask(mask: np.ndarray, expected_shape: tuple[int, int] | None = None) -> np.ndarray:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ProvenanceError("EDIT_MASK_INVALID", "Mask must be a 2D bool or uint8 array.")
    if expected_shape is not None and value.shape != expected_shape:
        raise ProvenanceError(
            "EDIT_MASK_SHAPE_MISMATCH",
            f"Expected {expected_shape}, got {value.shape}.",
        )
    return np.ascontiguousarray(value > 0, dtype=bool)


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ProvenanceError("EDIT_TIMESTAMP_INVALID", value) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ProvenanceError("EDIT_TIMESTAMP_INVALID", "Timezone is required.")


def begin_mask_edit(case_id: str, model_mask: np.ndarray) -> MaskEditProvenance:
    """Start a provenance chain without modifying or storing the model mask."""

    try:
        UUID(case_id)
    except ValueError as exc:
        raise ProvenanceError("EDIT_CASE_ID_INVALID", case_id) from exc
    normalized = _normalize_mask(model_mask)
    digest = mask_sha256(normalized)
    return MaskEditProvenance(
        case_id=case_id,
        shape=normalized.shape,
        model_mask_sha256=digest,
        current_mask_sha256=digest,
        edited=False,
        records=(),
    )


def record_mask_edit(
    provenance: MaskEditProvenance,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    *,
    tool: MaskEditTool,
    timestamp: str,
    parameters: Mapping[str, str | int | float | bool | None] | None = None,
) -> MaskEditProvenance:
    """Append one deterministic edit record and return a new immutable chain."""

    _validate_timestamp(timestamp)
    before = _normalize_mask(before_mask, provenance.shape)
    after = _normalize_mask(after_mask, provenance.shape)
    before_hash = mask_sha256(before)
    if before_hash != provenance.current_mask_sha256:
        raise ProvenanceError(
            "EDIT_HASH_CHAIN_BROKEN",
            "The supplied before-mask is not the current edited mask.",
        )
    after_hash = mask_sha256(after)
    if after_hash == before_hash:
        raise ProvenanceError("EDIT_NO_CHANGE", "An edit must change at least one pixel.")
    added = int((after & ~before).sum())
    removed = int((before & ~after).sum())
    safe_parameters = dict(parameters or {})
    if any(not isinstance(key, str) for key in safe_parameters):
        raise ProvenanceError("EDIT_PARAMETERS_INVALID", "Parameter keys must be text.")
    record = MaskEditRecord(
        sequence=len(provenance.records) + 1,
        timestamp=timestamp,
        tool=tool,
        before_sha256=before_hash,
        after_sha256=after_hash,
        added_pixels=added,
        removed_pixels=removed,
        parameters=tuple(sorted(safe_parameters.items())),
    )
    return MaskEditProvenance(
        case_id=provenance.case_id,
        shape=provenance.shape,
        model_mask_sha256=provenance.model_mask_sha256,
        current_mask_sha256=after_hash,
        edited=after_hash != provenance.model_mask_sha256,
        records=(*provenance.records, record),
    )
