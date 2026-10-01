"""Verified, opt-in case bundles for local Gyrus history."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

from .constants import (
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from .errors import HistoryRepositoryError
from .history import HistoryRecord, HistoryRepository
from .input_contract import (
    InputDescriptor,
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from .manifest import EXPECTED_MANIFEST_SHA256
from .mask_provenance import (
    MaskEditProvenance,
    MaskEditRecord,
    MaskEditTool,
    mask_sha256,
)
from .policy import combine_policy, decide_ood, decide_reliability
from .result_schema import validate_result_payload

_SOURCE_NAME = {"PNG": "source.png", "JPEG": "source.jpg"}
_BASE_FILES = {"result.json", "model_gray_128.bin", "source_preview.png", "overlay_preview.png"}
_TUMOR_FILES = {"probability.npy", "model_mask.png"}
_EDIT_FILES = {"edited_mask.png", "edit_provenance.json"}


def _fail(code: str, message: str) -> HistoryRepositoryError:
    return HistoryRepositoryError(code, message)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class HistoryCasePayload:
    result: dict[str, Any]
    source_path: str
    preview_png: bytes
    model_gray_128: bytes
    probability: np.ndarray | None
    model_mask: np.ndarray | None
    edited_mask: np.ndarray | None
    edit_provenance: MaskEditProvenance | None
    job_id: str
    route: str | None
    raw_probability_sha256: str | None


@dataclass(frozen=True, slots=True)
class LoadedHistoryCase:
    record: HistoryRecord
    result: dict[str, Any]
    descriptor: InputDescriptor
    preview_png: bytes
    overlay_preview_png: bytes
    model_gray_128: bytes
    probability: np.ndarray | None
    model_mask: np.ndarray | None
    edited_mask: np.ndarray | None
    edit_provenance: MaskEditProvenance | None
    job_id: str
    route: str | None
    raw_probability_sha256: str | None


def _planned_files(image_format: str, *, has_mask: bool, has_edit: bool) -> tuple[str, ...]:
    if image_format not in _SOURCE_NAME:
        raise _fail("HISTORY_SOURCE_FORMAT_INVALID", repr(image_format))
    files = set(_BASE_FILES)
    files.add(_SOURCE_NAME[image_format])
    files.add("bundle_manifest.json")
    if has_mask:
        files |= _TUMOR_FILES
    if has_edit:
        files |= _EDIT_FILES
    return tuple(sorted(files))


def planned_files(payload: HistoryCasePayload) -> tuple[str, ...]:
    """The exact file list displayed before the user consents to retention."""
    return _planned_files(
        payload.result["input"]["image_format"],
        has_mask=payload.model_mask is not None,
        has_edit=payload.edited_mask is not None,
    )


def _verify_frozen_policy(result: dict[str, Any]) -> None:
    ood = result["ood"]
    reliability = result["reliability"]
    spatial = result["spatial"]
    if (
        ood["threshold"] != OOD_REVIEW_THRESHOLD_TEXT
        or reliability["threshold"] != RELIABILITY_THRESHOLD_TEXT
        or (spatial is not None and spatial["segmentation_threshold"] != SEGMENTATION_THRESHOLD_TEXT)
    ):
        raise _fail("HISTORY_THRESHOLD_DRIFT", result["case_id"])
    expected_ood = decide_ood(ood["score"])
    expected_reliability = decide_reliability(
        result["classification"]["predicted_category"], reliability["score"]
    )
    expected_policy = combine_policy(expected_ood, expected_reliability)
    if (
        not expected_ood.valid
        or ood["status"] != expected_ood.status.value
        or ood["reason"] != expected_ood.reason.value
        or reliability["status"] != expected_reliability.status.value
        or reliability["reason"] != expected_reliability.reason.value
        or result["final_policy"]["status"] != expected_policy.status.value
        or result["final_policy"]["primary_reason"] != expected_policy.primary_reason
    ):
        raise _fail("HISTORY_POLICY_MISMATCH", result["case_id"])


def _png(mask: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(buffer, format="PNG")
    return buffer.getvalue()


def _overlay(gray_bytes: bytes, mask: np.ndarray | None) -> bytes:
    gray = np.frombuffer(gray_bytes, dtype=np.uint8).reshape(128, 128)
    rgb = np.repeat(gray[..., None], 3, axis=2).copy()
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
        rgb[active] = (rgb[active] * 0.6 + np.array((53, 208, 224)) * 0.4).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()


def _npy(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value, dtype=np.float64), allow_pickle=False)
    return buffer.getvalue()


def _validate_payload(payload: HistoryCasePayload) -> bytes:
    result = payload.result
    validate_result_payload(result)
    if result["stack_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise _fail("HISTORY_STACK_MISMATCH", "Frozen stack provenance changed")
    _verify_frozen_policy(result)
    source = Path(payload.source_path)
    if source.is_symlink() or not source.is_file():
        raise _fail("HISTORY_SOURCE_INVALID", "Original source is unavailable or linked")
    source_bytes = source.read_bytes()
    input_data = result["input"]
    if _digest(source_bytes) != input_data["source_sha256"] or len(source_bytes) != input_data["file_bytes"]:
        raise _fail("HISTORY_SOURCE_CHANGED", "Source changed after analysis")
    decoded = validate_and_decode_image(source)
    if decoded.descriptor.grayscale_sha256 != input_data["grayscale_sha256"]:
        raise _fail("HISTORY_SOURCE_CHANGED", "Decoded source changed after analysis")
    processed = preprocess_for_frozen_stack(decoded)
    gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype(np.uint8).tobytes()
    if gray != payload.model_gray_128 or len(gray) != 128 * 128:
        raise _fail("HISTORY_PREPROCESS_MISMATCH", "Model-space preview changed")
    spatial = result["spatial"]
    if spatial is None:
        if payload.probability is not None or payload.model_mask is not None:
            raise _fail("HISTORY_NO_TUMOR_MASK", "No-tumor output cannot retain a mask")
    else:
        probability, mask = payload.probability, payload.model_mask
        if probability is None or mask is None or probability.shape != (128, 128) or mask.shape != (128, 128):
            raise _fail("HISTORY_ARRAY_MISSING", "Tumor output requires 128x128 arrays")
        if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
            raise _fail("HISTORY_PROBABILITY_INVALID", "Probability array invalid")
        if not np.isin(mask, (0, 1)).all() or not np.array_equal(mask, probability >= 0.50):
            raise _fail("HISTORY_MASK_MISMATCH", "Frozen 0.50 mask does not match probabilities")
        if _digest(np.asarray(probability, dtype=np.float64).tobytes()) != spatial["probability_map"]["sha256"]:
            raise _fail("HISTORY_PROBABILITY_HASH", "Probability reference changed")
        if _digest(np.asarray(mask, dtype=np.uint8).tobytes()) != spatial["model_mask"]["sha256"]:
            raise _fail("HISTORY_MODEL_MASK_HASH", "Model mask reference changed")
        if int(mask.sum()) != spatial["statistics"]["foreground_pixels"]:
            raise _fail("HISTORY_MASK_COUNT", "Model mask count changed")
    if (payload.edited_mask is None) != (payload.edit_provenance is None):
        raise _fail("HISTORY_EDIT_INCOMPLETE", "Edited mask and provenance must be paired")
    if payload.edited_mask is not None:
        if payload.model_mask is None or payload.edited_mask.shape != (128, 128):
            raise _fail("HISTORY_EDIT_SHAPE", "Edited mask shape invalid")
        provenance = payload.edit_provenance
        if provenance.case_id != result["case_id"] or provenance.model_mask_sha256 != mask_sha256(payload.model_mask):
            raise _fail("HISTORY_EDIT_MODEL_MISMATCH", "Edit chain refers to a different model mask")
        if provenance.current_mask_sha256 != mask_sha256(payload.edited_mask) or not provenance.edited:
            raise _fail("HISTORY_EDIT_HASH_MISMATCH", "Saved human layer does not match chain")
        _verify_edit_chain(provenance)
    return source_bytes


def _verify_edit_chain(value: MaskEditProvenance) -> None:
    previous = value.model_mask_sha256
    for number, record in enumerate(value.records, 1):
        if record.sequence != number or record.before_sha256 != previous:
            raise _fail("HISTORY_EDIT_CHAIN_BROKEN", "Edit sequence/hash link invalid")
        previous = record.after_sha256
    if previous != value.current_mask_sha256 or value.reliability_scope != "ORIGINAL_MODEL_MASK_ONLY":
        raise _fail("HISTORY_EDIT_CHAIN_BROKEN", "Edit terminal hash or reliability scope invalid")


def save_history_bundle(repository: HistoryRepository, payload: HistoryCasePayload) -> HistoryRecord:
    """Commit files first, then the SQLite index; roll back files if indexing fails."""
    source_bytes = _validate_payload(payload)
    result = json.loads(json.dumps(payload.result))
    case_id = result["case_id"]
    source_name = _SOURCE_NAME[result["input"]["image_format"]]
    result["input"]["source_path"] = source_name  # avoid retaining the original directory
    files: dict[str, bytes] = {
        source_name: source_bytes,
        "result.json": (json.dumps(result, sort_keys=True, indent=2) + "\n").encode(),
        "source_preview.png": payload.preview_png,
        "model_gray_128.bin": payload.model_gray_128,
        "overlay_preview.png": _overlay(payload.model_gray_128, payload.model_mask),
    }
    if payload.model_mask is not None:
        files["probability.npy"] = _npy(payload.probability)
        files["model_mask.png"] = _png(payload.model_mask)
    if payload.edited_mask is not None:
        files["edited_mask.png"] = _png(payload.edited_mask)
        files["edit_provenance.json"] = (
            json.dumps(payload.edit_provenance.to_dict(), sort_keys=True, indent=2) + "\n"
        ).encode()
    if set(files) | {"bundle_manifest.json"} != set(planned_files(payload)):
        raise _fail("HISTORY_FILE_PLAN_MISMATCH", "Retention list did not match the files")
    saved_at = datetime.now(UTC).isoformat(timespec="seconds")
    manifest = {
        "schema_version": 1,
        "case_id": case_id,
        "saved_at": saved_at,
        "job_id": payload.job_id,
        "route": payload.route,
        "raw_probability_sha256": payload.raw_probability_sha256,
        "files": {name: _digest(content) for name, content in sorted(files.items())},
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    record = HistoryRecord(
        case_id=case_id,
        saved_at=saved_at,
        filename=payload.result["input"]["filename"],
        source_sha256=payload.result["input"]["source_sha256"],
        predicted_category=payload.result["classification"]["predicted_category"],
        ood_status=payload.result["ood"]["status"],
        final_status=payload.result["final_policy"]["status"],
        edited=payload.edited_mask is not None,
        bundle_path=str(repository.cases_root / case_id),
        model_mask_sha256=mask_sha256(payload.model_mask) if payload.model_mask is not None else None,
        edited_mask_sha256=mask_sha256(payload.edited_mask) if payload.edited_mask is not None else None,
        source_saved=True,
        bundle_manifest_sha256=_digest(manifest_bytes),
    )
    repository.initialize()
    final = repository.cases_root / case_id
    if final.exists() or repository.get(case_id) is not None:
        raise _fail("HISTORY_DUPLICATE_CASE", case_id)
    staging = repository.cases_root / f".pending-{uuid4().hex}"
    staging.mkdir()
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)
        (staging / "bundle_manifest.json").write_bytes(manifest_bytes)
        if any(_digest((staging / name).read_bytes()) != digest for name, digest in manifest["files"].items()):
            raise _fail("HISTORY_STAGE_HASH_MISMATCH", "Staged file changed")
        os.replace(staging, final)
        try:
            repository.save(record)
        except Exception:
            shutil.rmtree(final)
            raise
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return record


def _read_edit_provenance(data: bytes) -> MaskEditProvenance:
    raw = json.loads(data)
    records = tuple(
        MaskEditRecord(
            sequence=row["sequence"], timestamp=row["timestamp"],
            tool=MaskEditTool(row["tool"]), before_sha256=row["before_sha256"],
            after_sha256=row["after_sha256"], added_pixels=row["added_pixels"],
            removed_pixels=row["removed_pixels"],
            parameters=tuple(sorted(row["parameters"].items())),
        )
        for row in raw["records"]
    )
    value = MaskEditProvenance(
        case_id=raw["case_id"], shape=tuple(raw["shape"]),
        model_mask_sha256=raw["model_mask_sha256"],
        current_mask_sha256=raw["current_mask_sha256"], edited=raw["edited"],
        records=records, reliability_scope=raw["reliability_scope"],
    )
    _verify_edit_chain(value)
    if value.shape != (128, 128) or not value.edited:
        raise _fail("HISTORY_EDIT_SHAPE_INVALID", value.case_id)
    return value


def load_history_bundle(repository: HistoryRepository, case_id: str) -> LoadedHistoryCase:
    """Verify DB, manifest, exact file membership, and frozen result before display."""
    record = repository.get(case_id)
    if record is None or record.bundle_manifest_sha256 is None:
        raise _fail("HISTORY_NOT_FOUND_OR_LEGACY", case_id)
    directory = Path(record.bundle_path)
    if directory.is_symlink() or directory.resolve().parent != repository.cases_root:
        raise _fail("HISTORY_BUNDLE_PATH_ESCAPE", str(directory))
    manifest_bytes = (directory / "bundle_manifest.json").read_bytes()
    if _digest(manifest_bytes) != record.bundle_manifest_sha256:
        raise _fail("HISTORY_MANIFEST_HASH_MISMATCH", case_id)
    manifest = json.loads(manifest_bytes)
    if (manifest["schema_version"] != 1 or manifest["case_id"] != case_id
        or manifest["saved_at"] != record.saved_at):
        raise _fail("HISTORY_MANIFEST_INVALID", case_id)
    expected = set(manifest["files"]) | {"bundle_manifest.json"}
    if {path.name for path in directory.iterdir()} != expected:
        raise _fail("HISTORY_FILE_MEMBERSHIP_MISMATCH", case_id)
    files = {}
    for name, digest in manifest["files"].items():
        if Path(name).name != name or (directory / name).is_symlink():
            raise _fail("HISTORY_FILE_PATH_INVALID", name)
        content = (directory / name).read_bytes()
        if _digest(content) != digest:
            raise _fail("HISTORY_FILE_HASH_MISMATCH", name)
        files[name] = content
    result = json.loads(files["result.json"])
    validate_result_payload(result)
    if result["case_id"] != case_id or result["stack_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise _fail("HISTORY_RESULT_MISMATCH", case_id)
    if (
        result["classification"]["predicted_category"] != record.predicted_category
        or result["ood"]["status"] != record.ood_status
        or result["final_policy"]["status"] != record.final_status
        or not record.source_saved
    ):
        raise _fail("HISTORY_INDEX_RESULT_MISMATCH", case_id)
    source_name = _SOURCE_NAME.get(result["input"]["image_format"])
    if source_name not in files or result["input"]["source_path"] != source_name:
        raise _fail("HISTORY_SOURCE_PATH_INVALID", case_id)
    if set(files) != set(_planned_files(
        result["input"]["image_format"],
        has_mask=result["spatial"] is not None, has_edit=record.edited,
    )) - {"bundle_manifest.json"}:
        raise _fail("HISTORY_FILE_PLAN_MISMATCH", case_id)
    source_path = directory / source_name
    decoded = validate_and_decode_image(source_path)
    original = result["input"]
    if original["filename"] != record.filename:
        raise _fail("HISTORY_FILENAME_MISMATCH", case_id)
    if _digest(files[source_name]) != record.source_sha256 or record.source_sha256 != original["source_sha256"]:
        raise _fail("HISTORY_SOURCE_HASH_MISMATCH", case_id)
    for field in ("image_format", "source_mode", "width", "height", "file_bytes", "grayscale_sha256"):
        if getattr(decoded.descriptor, field) != original[field]:
            raise _fail("HISTORY_SOURCE_METADATA_MISMATCH", field)
    processed = preprocess_for_frozen_stack(decoded)
    gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype(np.uint8).tobytes()
    if gray != files["model_gray_128.bin"]:
        raise _fail("HISTORY_PREPROCESS_MISMATCH", case_id)
    probability = mask = edited = None
    if result["spatial"] is not None:
        probability = np.load(io.BytesIO(files["probability.npy"]), allow_pickle=False)
        with Image.open(io.BytesIO(files["model_mask.png"])) as image:
            mask = (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        if probability.dtype != np.float64 or mask.shape != (128, 128):
            raise _fail("HISTORY_ARRAY_INVALID", case_id)
        if _digest(probability.tobytes()) != result["spatial"]["probability_map"]["sha256"]:
            raise _fail("HISTORY_PROBABILITY_HASH", case_id)
        if _digest(mask.tobytes()) != result["spatial"]["model_mask"]["sha256"]:
            raise _fail("HISTORY_MODEL_MASK_HASH", case_id)
        if not np.isfinite(probability).all() or not np.array_equal(mask, probability >= 0.50):
            raise _fail("HISTORY_MASK_MISMATCH", case_id)
        if int(mask.sum()) != result["spatial"]["statistics"]["foreground_pixels"]:
            raise _fail("HISTORY_MASK_COUNT", case_id)
        if mask_sha256(mask) != record.model_mask_sha256:
            raise _fail("HISTORY_MODEL_MASK_HASH", case_id)
    provenance = None
    if record.edited:
        with Image.open(io.BytesIO(files["edited_mask.png"])) as image:
            edited = (np.asarray(image.convert("L"), dtype=np.uint8) > 0).astype(np.uint8)
        provenance = _read_edit_provenance(files["edit_provenance.json"])
        if (edited.shape != (128, 128)
            or provenance.case_id != case_id
            or mask_sha256(edited) != record.edited_mask_sha256
            or provenance.current_mask_sha256 != record.edited_mask_sha256
            or provenance.model_mask_sha256 != record.model_mask_sha256):
            raise _fail("HISTORY_EDIT_HASH_MISMATCH", case_id)
    _verify_frozen_policy(result)
    descriptor = replace(decoded.descriptor, filename=record.filename)
    for array in (probability, mask, edited):
        if array is not None:
            array.setflags(write=False)
    return LoadedHistoryCase(
        record, result, descriptor, files["source_preview.png"],
        files["overlay_preview.png"], gray,
        probability, mask, edited, provenance, manifest["job_id"],
        manifest["route"], manifest["raw_probability_sha256"],
    )
