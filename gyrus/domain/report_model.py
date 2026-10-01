"""Immutable input model for the later deterministic three-page PDF renderer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from .constants import (
    DISCLAIMER,
    OOD_REVIEW_THRESHOLD_TEXT,
    PDF_CREATOR,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from .errors import ResultValidationError
from .mask_provenance import MaskEditProvenance
from .result_schema import AnalysisResult, ArrayReference, validate_result_payload


@dataclass(frozen=True, slots=True)
class ReportAsset:
    role: str
    reference: ArrayReference
    caption: str

    def __post_init__(self) -> None:
        if not self.role or not self.caption:
            raise ResultValidationError(
                "REPORT_ASSET_INVALID", "Report asset role and caption are required."
            )


@dataclass(frozen=True, slots=True)
class ReportPage:
    page_number: int
    title: str
    section_ids: tuple[str, ...]
    footer: str = DISCLAIMER

    def __post_init__(self) -> None:
        if self.page_number not in {1, 2, 3} or not self.title or not self.section_ids:
            raise ResultValidationError("REPORT_PAGE_INVALID", repr(self))
        if self.footer != DISCLAIMER:
            raise ResultValidationError(
                "REPORT_DISCLAIMER_MISSING", f"page {self.page_number}"
            )


@dataclass(frozen=True, slots=True)
class PdfReportModel:
    schema_version: int
    case_id: str
    creator: str
    default_filename: str
    generated_at: str
    disclaimer: str
    result: dict[str, Any]
    pages: tuple[ReportPage, ReportPage, ReportPage]
    assets: tuple[ReportAsset, ...]
    edit_provenance: dict[str, Any] | None
    technical_thresholds: dict[str, str]

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.creator != PDF_CREATOR:
            raise ResultValidationError("REPORT_SCHEMA_INVALID", repr(self.schema_version))
        if self.default_filename != f"gyrus-research-summary_{self.case_id}.pdf":
            raise ResultValidationError(
                "REPORT_FILENAME_INVALID", self.default_filename
            )
        if self.disclaimer != DISCLAIMER or any(
            page.footer != DISCLAIMER for page in self.pages
        ):
            raise ResultValidationError(
                "REPORT_DISCLAIMER_MISSING", "Every report page requires the disclaimer."
            )
        if tuple(page.page_number for page in self.pages) != (1, 2, 3):
            raise ResultValidationError("REPORT_PAGE_ORDER_INVALID", repr(self.pages))
        try:
            generated = datetime.fromisoformat(self.generated_at)
        except ValueError as exc:
            raise ResultValidationError(
                "REPORT_TIMESTAMP_INVALID", self.generated_at
            ) from exc
        if generated.tzinfo is None or generated.utcoffset() is None:
            raise ResultValidationError(
                "REPORT_TIMESTAMP_INVALID", "Timezone is required."
            )
        expected = {
            "segmentation": SEGMENTATION_THRESHOLD_TEXT,
            "reliability": RELIABILITY_THRESHOLD_TEXT,
            "ood_review": OOD_REVIEW_THRESHOLD_TEXT,
        }
        if self.technical_thresholds != expected:
            raise ResultValidationError(
                "REPORT_THRESHOLD_DRIFT", repr(self.technical_thresholds)
            )
        validate_result_payload(self.result)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_pdf_report_model(
    result: AnalysisResult,
    *,
    generated_at: str,
    assets: tuple[ReportAsset, ...] = (),
    edit_provenance: MaskEditProvenance | None = None,
) -> PdfReportModel:
    """Create the renderer-independent fixed three-page report contract."""

    return build_pdf_report_model_from_payload(
        result.to_dict(), generated_at=generated_at, assets=assets,
        edit_provenance=edit_provenance,
    )


def build_pdf_report_model_from_payload(
    result: dict[str, Any], *, generated_at: str,
    assets: tuple[ReportAsset, ...] = (),
    edit_provenance: MaskEditProvenance | None = None,
) -> PdfReportModel:
    """Build the same fixed contract from a verified live or archived result."""

    validate_result_payload(result)
    if edit_provenance is not None:
        if edit_provenance.case_id != result["case_id"]:
            raise ResultValidationError(
                "REPORT_EDIT_CASE_MISMATCH", edit_provenance.case_id
            )
        if result["spatial"] is None:
            raise ResultValidationError(
                "REPORT_EDIT_WITHOUT_MODEL_MASK",
                "Edited-mask provenance requires an original model mask.",
            )
        if edit_provenance.model_mask_sha256 != result["spatial"]["model_mask"]["sha256"]:
            raise ResultValidationError(
                "REPORT_EDIT_MODEL_HASH_MISMATCH",
                "Edit provenance does not reference this result's model mask.",
            )
    pages = (
        ReportPage(
            1,
            "Research result overview",
            ("case_status", "source_metadata", "category_probabilities", "images"),
        ),
        ReportPage(
            2,
            "Spatial evidence",
            ("probability_map", "model_mask", "edited_mask", "pixel_measurements"),
        ),
        ReportPage(
            3,
            "Provenance and limitations",
            ("frozen_stack", "policy", "edit_provenance", "limitations"),
        ),
    )
    return PdfReportModel(
        schema_version=1,
        case_id=result["case_id"],
        creator=PDF_CREATOR,
        default_filename=f"gyrus-research-summary_{result['case_id']}.pdf",
        generated_at=generated_at,
        disclaimer=DISCLAIMER,
        result=result,
        pages=pages,
        assets=assets,
        edit_provenance=(None if edit_provenance is None else edit_provenance.to_dict()),
        technical_thresholds={
            "segmentation": SEGMENTATION_THRESHOLD_TEXT,
            "reliability": RELIABILITY_THRESHOLD_TEXT,
            "ood_review": OOD_REVIEW_THRESHOLD_TEXT,
        },
    )
