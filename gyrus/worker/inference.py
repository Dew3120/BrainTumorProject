"""Ordered frozen inference executed wholly inside the worker process."""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from scipy.ndimage import label

from gyrus.domain.constants import CLASS_NAMES, SEGMENTATION_THRESHOLD_TEXT
from gyrus.domain.errors import InputValidationError
from gyrus.domain.input_contract import (
    PreprocessedInput,
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.domain.manifest import sha256_file
from gyrus.domain.orchestration import (
    FrozenAnalysisOrchestrator,
    FrozenInferenceEvidence,
)
from gyrus.domain.policy import decide_ood
from gyrus.domain.result_schema import (
    ArrayReference,
    ExecutionInfo,
    MaskStatistics,
    SpatialResult,
    validate_result_payload,
)

from .loader import FrozenWorkerSession
from .protocol import (
    AnalyzeJob,
    AnalyzeSuccess,
    ArrayPayload,
    WorkerDiagnostics,
    WorkerEvent,
    WorkerStage,
    canonical_local_path,
)


@dataclass(frozen=True, slots=True)
class _EvidenceBackend:
    evidence: FrozenInferenceEvidence

    def infer_silently(self, _image: PreprocessedInput) -> FrozenInferenceEvidence:
        return self.evidence


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _array_payload(job_id: str, role: str, value: np.ndarray) -> ArrayPayload:
    contiguous = np.ascontiguousarray(value)
    data = contiguous.tobytes()
    return ArrayPayload(
        role=role,
        storage_key=f"{job_id}/{role}",
        shape=tuple(contiguous.shape),
        dtype=contiguous.dtype.name,
        sha256=hashlib.sha256(data).hexdigest(),
        data=data,
    )


def _mask_statistics(mask: np.ndarray) -> MaskStatistics:
    hard = np.asarray(mask, dtype=bool)
    foreground = int(hard.sum())
    components = int(label(hard, structure=np.ones((3, 3), dtype=np.uint8))[1])
    box = None
    if foreground:
        y, x = np.where(hard)
        x0, x1 = int(x.min()), int(x.max())
        y0, y1 = int(y.min()), int(y.max())
        box = (x0, y0, x1 - x0 + 1, y1 - y0 + 1)
    return MaskStatistics(
        foreground_pixels=foreground,
        total_pixels=int(hard.size),
        area_fraction=foreground / hard.size,
        component_count=components,
        bounding_box_xywh=box,
    )


class FrozenInferenceEngine:
    """Execute one image at a time with no intermediate IPC disclosure."""

    def __init__(self, session: FrozenWorkerSession, generation: int) -> None:
        self.session = session
        self.generation = generation

    def _event(
        self,
        events: list[WorkerEvent],
        stage: WorkerStage,
        detail: str,
        *,
        disclosure_allowed: bool = False,
    ) -> None:
        events.append(WorkerEvent(stage, _now(), disclosure_allowed, detail))

    def analyze(self, job: AnalyzeJob) -> AnalyzeSuccess:
        started_clock = time.perf_counter()
        started_at = _now()
        events: list[WorkerEvent] = []
        runtime = self.session.runtime
        module = self.session.runtime_module
        device = self.session.device

        source_path = canonical_local_path(job.image_path)
        decoded = validate_and_decode_image(source_path)
        if sha256_file(source_path) != decoded.descriptor.source_sha256:
            raise InputValidationError(
                "INPUT_CHANGED_DURING_DECODE",
                "The selected image changed while it was being decoded.",
            )
        image = preprocess_for_frozen_stack(decoded)
        self._event(events, WorkerStage.PREPROCESS_INPUT, "input contract passed")

        classifier_tensor = torch.from_numpy(
            np.array(image.classifier_chw, copy=True)
        )[None].float().to(device)
        segmenter_tensor = torch.from_numpy(
            np.array(image.segmenter_chw, copy=True)
        )[None].float().to(device)

        with torch.inference_mode():
            logits = runtime.classifier(classifier_tensor)
            class_probability = (
                logits.softmax(dim=1)[0].detach().cpu().numpy().astype(np.float64)
            )
        if class_probability.shape != (4,) or not np.isfinite(class_probability).all():
            raise RuntimeError("classifier output is not one finite four-class vector")
        predicted_index = int(np.argmax(class_probability))
        predicted_category = CLASS_NAMES[predicted_index]
        self._event(
            events,
            WorkerStage.SILENT_CLASSIFIER,
            "raw classifier probabilities computed and withheld",
        )

        runtime._bottleneck = None
        with torch.inference_mode():
            v1 = runtime.segmenters["v1"](segmenter_tensor).sigmoid()[0, 0]
        bottleneck = runtime._bottleneck
        if bottleneck is None or tuple(bottleneck.shape)[1] != 128:
            raise RuntimeError("O03 bottleneck capture failed")
        embedding = bottleneck.mean(dim=(2, 3))[0].detach().cpu().numpy().astype(np.float64)
        self._event(
            events,
            WorkerStage.SILENT_V1_SEGMENTER,
            "v1 probability and O03 bottleneck computed and withheld",
        )

        o02_values = module.o02_features(image.gray_uint8, image.descriptor.file_bytes)
        o02_vector = [o02_values[name] for name in module.O02_FEATURES]
        o02_score = module._manual_logistic(
            o02_vector,
            runtime.o02["scaler_mean"],
            runtime.o02["scaler_scale"],
            runtime.o02["coefficients"],
            runtime.o02["intercept"],
        )
        self._event(events, WorkerStage.OOD_O02, "O02 feature score computed")

        centered = embedding - runtime.o03_location
        o03_score = float(centered @ runtime.o03_precision @ centered)
        self._event(events, WorkerStage.OOD_O03, "O03 Mahalanobis score computed")

        # The registered O04 entropy is defined on raw E08. These component
        # forwards are therefore part of silent O04 computation; their maps
        # remain private until O05 and O06 complete.
        with torch.inference_mode():
            e05 = runtime.segmenters["e05"](segmenter_tensor).sigmoid()[0, 0]
            e06 = runtime.segmenters["e06"](segmenter_tensor).sigmoid()[0, 0]
            specialist = runtime.segmenters["e07"](segmenter_tensor).sigmoid()[0, 0]
        routed = specialist if predicted_index == 0 else v1
        raw = ((e05 + e06 + routed) / 3.0).detach().cpu().numpy().astype(np.float64)
        o04_classifier = float(1.0 - class_probability.max())
        o04_segmenter = float(module.binary_entropy(raw).mean())
        self._event(
            events,
            WorkerStage.OOD_O04,
            "O04 classifier and registered raw-E08 entropy scores computed",
        )

        components = {
            "o02_probability": float(o02_score),
            "o03_mahalanobis": o03_score,
            "o04_classifier": o04_classifier,
            "o04_segmenter_entropy": o04_segmenter,
        }
        if any(not math.isfinite(value) for value in components.values()):
            raise RuntimeError("OOD component score is non-finite")
        normalized = {
            name: runtime._cdf(name, components[name]) for name in module.O05_SCORE_ORDER
        }
        ood_score = module._manual_logistic(
            [normalized[name] for name in module.O05_SCORE_ORDER],
            runtime.o05["scaler_mean"],
            runtime.o05["scaler_scale"],
            runtime.o05["coefficients"],
            runtime.o05["intercept"],
        )
        self._event(events, WorkerStage.OOD_O05, "O05 ensemble score validated")

        ood_decision = decide_ood(ood_score)
        if not ood_decision.valid:
            raise RuntimeError("O05 produced an invalid policy input")
        self._event(events, WorkerStage.OOD_POLICY, ood_decision.reason.value)

        x = np.asarray(runtime.c03["x_thresholds"], dtype=np.float64)
        y = np.asarray(runtime.c03["y_thresholds"], dtype=np.float64)
        calibrated = np.interp(raw.reshape(-1), x, y).reshape(raw.shape)
        hard = calibrated >= float(runtime.segmentation_threshold)
        reliability: float | None = None
        spatial: SpatialResult | None = None
        arrays: tuple[ArrayPayload, ...] = ()
        route: str | None = None
        raw_hash: str | None = None
        if predicted_category != "no_tumor":
            route = "glioma_specialist" if predicted_index == 0 else "v1_fallback"
            feature_values = module.prediction_features(
                calibrated,
                float(runtime.segmentation_threshold),
            )
            vector = np.asarray(
                [[feature_values[name] for name in module.R02_FEATURES]],
                dtype=np.float64,
            )
            reliability = float(
                runtime.r02_logistic.predict_proba(runtime.r02_scaler.transform(vector))[0, 1]
            )
            probability_payload = _array_payload(job.job_id, "probability_map", calibrated)
            mask_payload = _array_payload(job.job_id, "model_mask", hard.astype(np.uint8))
            arrays = (probability_payload, mask_payload)
            spatial = SpatialResult(
                probability_map=ArrayReference(
                    role=probability_payload.role,
                    shape=probability_payload.shape,
                    dtype=probability_payload.dtype,
                    sha256=probability_payload.sha256,
                    storage_key=probability_payload.storage_key,
                ),
                model_mask=ArrayReference(
                    role=mask_payload.role,
                    shape=mask_payload.shape,
                    dtype=mask_payload.dtype,
                    sha256=mask_payload.sha256,
                    storage_key=mask_payload.storage_key,
                ),
                statistics=_mask_statistics(hard),
                segmentation_threshold=SEGMENTATION_THRESHOLD_TEXT,
            )
            raw_hash = module.array_sha256(raw)
        self._event(
            events,
            WorkerStage.FULL_E08_C03_R02,
            "cached raw E08 released after OOD policy; C03 and conditional R02 applied",
        )

        completed_at = _now()
        duration_ms = max(0, round((time.perf_counter() - started_clock) * 1000))
        evidence = FrozenInferenceEvidence(
            class_probabilities=tuple(float(value) for value in class_probability),
            predicted_category=predicted_category,
            ood_score=float(ood_score),
            reliability_score=reliability,
            spatial=spatial,
            execution=ExecutionInfo(
                device=device.type,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            ),
        )
        orchestrator = FrozenAnalysisOrchestrator(
            self.session.stack,
            _EvidenceBackend(evidence),
        )
        outcome = orchestrator.analyze(image, case_id=job.case_id)
        result = outcome.result.to_dict()
        validate_result_payload(result)
        self._event(
            events,
            WorkerStage.RESULT_ASSEMBLY,
            "complete schema-validated result ready for IPC disclosure",
            disclosure_allowed=True,
        )
        return AnalyzeSuccess(
            job_id=job.job_id,
            case_id=job.case_id,
            result=result,
            arrays=arrays,
            events=tuple(events),
            diagnostics=WorkerDiagnostics(
                stage_order=tuple(event.stage.value for event in events),
                ood_components=tuple((name, components[name]) for name in module.O05_SCORE_ORDER),
                normalized_ood_components=tuple(
                    (name, normalized[name]) for name in module.O05_SCORE_ORDER
                ),
                route=route,
                raw_probability_sha256=raw_hash,
                load_session_id=self.session.load_session_id,
            ),
            generation=self.generation,
        )
