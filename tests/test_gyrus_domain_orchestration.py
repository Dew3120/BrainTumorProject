from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from PIL import Image

from gyrus.domain.constants import DisplayStatus, PipelineStage, ReliabilityReason
from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.domain.manifest import verify_frozen_stack
from gyrus.domain.orchestration import (
    FrozenAnalysisOrchestrator,
    FrozenInferenceEvidence,
)
from gyrus.domain.result_schema import (
    ArrayReference,
    ExecutionInfo,
    MaskStatistics,
    SpatialResult,
    validate_result_payload,
)

ROOT = Path(__file__).resolve().parents[1]
HASH = "a" * 64


@dataclass
class Backend:
    evidence: FrozenInferenceEvidence

    def infer_silently(self, _image):
        return self.evidence


def _execution():
    return ExecutionInfo(
        device="cpu",
        started_at="2026-09-30T10:00:00+05:30",
        completed_at="2026-09-30T10:00:01+05:30",
        duration_ms=1000,
    )


def _spatial():
    return SpatialResult(
        probability_map=ArrayReference("probability", (128, 128), "float32", HASH, "p"),
        model_mask=ArrayReference("model_mask", (128, 128), "uint8", HASH, "m"),
        statistics=MaskStatistics(100, 16384, 100 / 16384, 1, (10, 20, 10, 10)),
    )


def _image(tmp_path):
    path = tmp_path / "case.png"
    Image.new("L", (64, 64), 20).save(path)
    return preprocess_for_frozen_stack(validate_and_decode_image(path))


def _orchestrator(evidence):
    return FrozenAnalysisOrchestrator(
        verify_frozen_stack(ROOT / "runs/final_stack"),
        Backend(evidence),
    )


def test_tumor_result_discloses_only_after_ood_validation(tmp_path):
    evidence = FrozenInferenceEvidence(
        (0.7, 0.1, 0.1, 0.1),
        "glioma",
        0.1,
        1.0,
        _spatial(),
        _execution(),
    )
    outcome = _orchestrator(evidence).analyze(_image(tmp_path))
    stages = [event.stage for event in outcome.events]
    assert stages == [
        PipelineStage.INPUT_VALIDATED,
        PipelineStage.SILENT_FROZEN_FORWARD_PASSES,
        PipelineStage.O05_VALIDATED,
        PipelineStage.O06_POLICY,
        PipelineStage.CLASSIFICATION_DISCLOSURE,
        PipelineStage.CONDITIONAL_SEGMENTATION_RELIABILITY_POLICY,
        PipelineStage.RESULT,
    ]
    disclosure_index = stages.index(PipelineStage.CLASSIFICATION_DISCLOSURE)
    assert all(not event.disclosure_allowed for event in outcome.events[:disclosure_index])
    assert outcome.result.final_policy.status == DisplayStatus.STANDARD
    UUID(outcome.result.case_id)
    validate_result_payload(outcome.result.to_dict())


def test_invalid_ood_withholds_all_model_outputs(tmp_path):
    evidence = FrozenInferenceEvidence(
        (0.7, 0.1, 0.1, 0.1),
        "glioma",
        float("nan"),
        1.0,
        _spatial(),
        _execution(),
    )
    outcome = _orchestrator(evidence).analyze(_image(tmp_path))
    assert outcome.result.classification is None
    assert outcome.result.spatial is None
    assert outcome.result.reliability is None
    assert outcome.result.final_policy.status == DisplayStatus.REVIEW_REQUIRED
    assert PipelineStage.CLASSIFICATION_DISCLOSURE not in {
        event.stage for event in outcome.events
    }


def test_no_tumor_discards_silent_spatial_output_and_fails_closed(tmp_path):
    evidence = FrozenInferenceEvidence(
        (0.01, 0.01, 0.01, 0.97),
        "no_tumor",
        0.1,
        1.0,
        _spatial(),
        _execution(),
    )
    result = _orchestrator(evidence).analyze(_image(tmp_path)).result
    assert result.spatial is None
    assert result.reliability is not None
    assert (
        result.reliability.reason
        == ReliabilityReason.SEGMENTATION_RELIABILITY_NOT_APPLICABLE
    )
    assert result.final_policy.status == DisplayStatus.REVIEW_REQUIRED
