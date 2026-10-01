"""Verify a final worker response before any model value reaches a widget."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gyrus.domain.constants import (
    CLASS_NAMES,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.domain.policy import combine_policy, decide_ood, decide_reliability
from gyrus.domain.result_schema import validate_result_payload
from gyrus.worker.protocol import AnalyzeSuccess, ArrayPayload, WorkerStage

from .input_controller import ValidatedSelection


class PresentationError(RuntimeError):
    """A response/selection mismatch must suppress all result display."""


@dataclass(frozen=True, slots=True)
class ResultPresentation:
    result: dict
    selection: ValidatedSelection
    probability: np.ndarray | None
    mask: np.ndarray | None
    job_id: str
    route: str | None
    raw_probability_sha256: str | None


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise PresentationError(code)


def _map_array(payload: ArrayPayload, dtype: str) -> np.ndarray:
    _require(payload.dtype == dtype and payload.shape == (128, 128), "ARRAY_CONTRACT_DRIFT")
    value = np.frombuffer(payload.data, dtype=np.dtype(dtype)).reshape(128, 128).copy()
    value.setflags(write=False)
    return value


def build_presentation(
    response: AnalyzeSuccess,
    selection: ValidatedSelection,
) -> ResultPresentation:
    """Apply the UI trust boundary after Stage 2's parent-side IPC validation."""

    _require(isinstance(response, AnalyzeSuccess), "WORKER_RESPONSE_NOT_FINAL")
    result = response.result
    validate_result_payload(result)
    _require(response.events[-1].disclosure_allowed, "RESULT_DISCLOSURE_INCOMPLETE")
    _require(
        all(not event.disclosure_allowed for event in response.events[:-1]),
        "RESULT_PRE_OOD_DISCLOSURE",
    )
    _require(
        response.diagnostics.stage_order == tuple(stage.value for stage in WorkerStage),
        "RESULT_STAGE_ORDER_DRIFT",
    )
    _require(
        result["case_id"] == response.case_id
        and result["stack_manifest_sha256"] == EXPECTED_MANIFEST_SHA256,
        "RESULT_STACK_PROVENANCE_MISMATCH",
    )
    image = result["input"]
    descriptor = selection.descriptor
    _require(
        image["source_sha256"] == descriptor.source_sha256
        and image["grayscale_sha256"] == descriptor.grayscale_sha256
        and image["source_path"] == descriptor.source_path
        and image["width"] == descriptor.width
        and image["height"] == descriptor.height,
        "RESULT_INPUT_CHANGED_AFTER_VALIDATION",
    )
    ood = result["ood"]
    _require(ood["threshold"] == OOD_REVIEW_THRESHOLD_TEXT, "RESULT_OOD_THRESHOLD_DRIFT")
    expected_ood = decide_ood(ood["score"])
    _require(
        ood["status"] == expected_ood.status.value
        and ood["reason"] == expected_ood.reason.value
        and ood["valid"] == expected_ood.valid,
        "RESULT_OOD_POLICY_MISMATCH",
    )
    _require(expected_ood.valid, "RESULT_OOD_INVALID")
    classification = result["classification"]
    _require(isinstance(classification, dict), "RESULT_CLASSIFICATION_MISSING")
    probabilities = classification["probabilities"]
    _require(len(probabilities) == 4, "RESULT_CLASS_COUNT_DRIFT")
    _require(
        all(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1
            for value in probabilities)
        and abs(sum(probabilities) - 1.0) <= 1e-6,
        "RESULT_CLASS_PROBABILITIES_INVALID",
    )
    category = classification["predicted_category"]
    _require(
        category == CLASS_NAMES[max(range(4), key=probabilities.__getitem__)],
        "RESULT_CLASS_ARGMAX_MISMATCH",
    )
    reliability = result["reliability"]
    _require(isinstance(reliability, dict), "RESULT_RELIABILITY_MISSING")
    _require(
        reliability["threshold"] == RELIABILITY_THRESHOLD_TEXT,
        "RESULT_RELIABILITY_THRESHOLD_DRIFT",
    )
    expected_reliability = decide_reliability(category, reliability["score"])
    _require(
        reliability["status"] == expected_reliability.status.value
        and reliability["reason"] == expected_reliability.reason.value,
        "RESULT_RELIABILITY_POLICY_MISMATCH",
    )
    expected_final = combine_policy(expected_ood, expected_reliability)
    _require(
        result["final_policy"]["status"] == expected_final.status.value
        and result["final_policy"]["primary_reason"] == expected_final.primary_reason,
        "RESULT_FINAL_POLICY_MISMATCH",
    )

    spatial = result["spatial"]
    if category == "no_tumor":
        _require(spatial is None and not response.arrays, "RESULT_NO_TUMOR_MASK_DISCLOSED")
        probability = mask = None
    else:
        _require(isinstance(spatial, dict), "RESULT_SPATIAL_MISSING")
        _require(
            spatial["segmentation_threshold"] == SEGMENTATION_THRESHOLD_TEXT,
            "RESULT_SEGMENTATION_THRESHOLD_DRIFT",
        )
        payloads = {item.role: item for item in response.arrays}
        _require(
            len(response.arrays) == 2 and set(payloads) == {"probability_map", "model_mask"},
            "RESULT_ARRAY_MEMBERSHIP_MISMATCH",
        )
        for role, item in payloads.items():
            reference = spatial[role]
            _require(
                item.sha256 == reference["sha256"]
                and item.storage_key == reference["storage_key"]
                and list(item.shape) == reference["shape"]
                and item.dtype == reference["dtype"],
                "RESULT_ARRAY_REFERENCE_MISMATCH",
            )
        probability = _map_array(payloads["probability_map"], "float64")
        mask = _map_array(payloads["model_mask"], "uint8")
        _require(
            np.isfinite(probability).all()
            and ((0 <= probability) & (probability <= 1)).all(),
            "RESULT_PROBABILITY_MAP_INVALID",
        )
        _require(
            np.isin(mask, (0, 1)).all()
            and np.array_equal(mask, probability >= 0.50)
            and int(mask.sum()) == spatial["statistics"]["foreground_pixels"],
            "RESULT_HARD_MASK_MISMATCH",
        )

    return ResultPresentation(
        result,
        selection,
        probability,
        mask,
        response.job_id,
        response.diagnostics.route,
        response.diagnostics.raw_probability_sha256,
    )
