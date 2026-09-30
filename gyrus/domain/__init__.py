"""Qt-free domain contracts for the Gyrus desktop application."""

from .constants import (
    DISCLAIMER,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
    DisplayStatus,
    OodStatus,
    PipelineStage,
    ReliabilityReason,
)
from .history import HistoryRecord, HistoryRepository
from .input_contract import (
    DecodedInput,
    InputDescriptor,
    InputLimits,
    PreprocessedInput,
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from .manifest import VerifiedFrozenStack, verify_frozen_stack
from .mask_provenance import (
    MaskEditProvenance,
    MaskEditTool,
    begin_mask_edit,
    record_mask_edit,
)
from .orchestration import (
    FrozenAnalysisOrchestrator,
    FrozenInferenceBackend,
    FrozenInferenceEvidence,
    OrchestrationOutcome,
    PipelineEvent,
)
from .policy import combine_policy, decide_ood, decide_reliability
from .report_model import PdfReportModel, ReportAsset, build_pdf_report_model
from .result_schema import AnalysisResult, validate_result_payload

__all__ = [
    "DISCLAIMER",
    "OOD_REVIEW_THRESHOLD_TEXT",
    "RELIABILITY_THRESHOLD_TEXT",
    "SEGMENTATION_THRESHOLD_TEXT",
    "AnalysisResult",
    "DecodedInput",
    "DisplayStatus",
    "FrozenAnalysisOrchestrator",
    "FrozenInferenceBackend",
    "FrozenInferenceEvidence",
    "HistoryRecord",
    "HistoryRepository",
    "InputDescriptor",
    "InputLimits",
    "MaskEditProvenance",
    "MaskEditTool",
    "OodStatus",
    "OrchestrationOutcome",
    "PdfReportModel",
    "PipelineEvent",
    "PipelineStage",
    "PreprocessedInput",
    "ReliabilityReason",
    "ReportAsset",
    "VerifiedFrozenStack",
    "begin_mask_edit",
    "build_pdf_report_model",
    "combine_policy",
    "decide_ood",
    "decide_reliability",
    "preprocess_for_frozen_stack",
    "record_mask_edit",
    "validate_and_decode_image",
    "validate_result_payload",
    "verify_frozen_stack",
]
