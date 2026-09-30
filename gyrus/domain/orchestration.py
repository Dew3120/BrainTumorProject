"""Qt-free orchestration of silent frozen inference and ordered disclosure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from .constants import DisplayStatus, PipelineStage
from .errors import ResultValidationError
from .input_contract import PreprocessedInput
from .manifest import VerifiedFrozenStack
from .policy import FinalPolicyDecision, combine_policy, decide_ood, decide_reliability
from .result_schema import (
    AnalysisResult,
    ClassificationResult,
    ExecutionInfo,
    SpatialResult,
)


@dataclass(frozen=True, slots=True)
class FrozenInferenceEvidence:
    """Complete private evidence returned by the future worker process."""

    class_probabilities: tuple[float, float, float, float]
    predicted_category: str
    ood_score: float | None
    reliability_score: float | None
    spatial: SpatialResult | None
    execution: ExecutionInfo


class FrozenInferenceBackend(Protocol):
    """Worker-facing interface; implementations may use PyTorch but never Qt."""

    def infer_silently(self, image: PreprocessedInput) -> FrozenInferenceEvidence:
        """Return private evidence without publishing intermediate outputs."""


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    stage: PipelineStage
    disclosure_allowed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class OrchestrationOutcome:
    result: AnalysisResult
    events: tuple[PipelineEvent, ...]


class FrozenAnalysisOrchestrator:
    """Enforce Amendment 22 ordering around a verified frozen worker backend."""

    def __init__(
        self,
        stack: VerifiedFrozenStack,
        backend: FrozenInferenceBackend,
    ) -> None:
        self._stack = stack
        self._backend = backend

    def analyze(
        self,
        image: PreprocessedInput,
        *,
        case_id: str | None = None,
    ) -> OrchestrationOutcome:
        events = [
            PipelineEvent(PipelineStage.INPUT_VALIDATED, False, "input contract passed"),
            PipelineEvent(
                PipelineStage.SILENT_FROZEN_FORWARD_PASSES,
                False,
                "worker executing frozen stack; model outputs withheld",
            ),
        ]
        evidence = self._backend.infer_silently(image)
        ood = decide_ood(evidence.ood_score)
        events.append(
            PipelineEvent(
                PipelineStage.O05_VALIDATED,
                False,
                ood.reason.value,
            )
        )
        events.append(
            PipelineEvent(
                PipelineStage.O06_POLICY,
                False,
                ood.status.value,
            )
        )

        selected_case_id = case_id or str(uuid4())
        if not ood.valid:
            reason = ood.reason.value
            final = FinalPolicyDecision(
                DisplayStatus.REVIEW_REQUIRED,
                reason,
                (reason,),
            )
            result = AnalysisResult(
                schema_version=1,
                case_id=selected_case_id,
                stack_manifest_sha256=self._stack.manifest_sha256,
                input=image.descriptor,
                execution=evidence.execution,
                ood=ood,
                classification=None,
                spatial=None,
                reliability=None,
                final_policy=final,
                warnings=("Frozen model outputs withheld because OOD validation failed.",),
            )
            events.append(PipelineEvent(PipelineStage.RESULT, True, final.status.value))
            return OrchestrationOutcome(result, tuple(events))

        classification = ClassificationResult(
            probabilities=evidence.class_probabilities,
            predicted_category=evidence.predicted_category,
        )
        events.append(
            PipelineEvent(
                PipelineStage.CLASSIFICATION_DISCLOSURE,
                True,
                classification.predicted_category,
            )
        )
        reliability = decide_reliability(
            classification.predicted_category,
            evidence.reliability_score,
        )
        spatial = None if classification.predicted_category == "no_tumor" else evidence.spatial
        if classification.predicted_category != "no_tumor" and spatial is None:
            raise ResultValidationError(
                "WORKER_SPATIAL_OUTPUT_MISSING",
                "Tumor-category inference completed without a spatial result.",
            )
        events.append(
            PipelineEvent(
                PipelineStage.CONDITIONAL_SEGMENTATION_RELIABILITY_POLICY,
                True,
                reliability.reason.value,
            )
        )
        final = combine_policy(ood, reliability)
        warnings: tuple[str, ...] = ()
        if classification.predicted_category == "no_tumor":
            warnings = (
                "The no_tumor label is a model prediction; segmentation reliability is not applicable.",
            )
        result = AnalysisResult(
            schema_version=1,
            case_id=selected_case_id,
            stack_manifest_sha256=self._stack.manifest_sha256,
            input=image.descriptor,
            execution=evidence.execution,
            ood=ood,
            classification=classification,
            spatial=spatial,
            reliability=reliability,
            final_policy=final,
            warnings=warnings,
        )
        events.append(PipelineEvent(PipelineStage.RESULT, True, final.status.value))
        return OrchestrationOutcome(result, tuple(events))
