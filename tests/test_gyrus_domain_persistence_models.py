from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.errors import HistoryRepositoryError, ProvenanceError
from gyrus.domain.history import HistoryRecord, HistoryRepository
from gyrus.domain.input_contract import InputDescriptor
from gyrus.domain.mask_provenance import (
    MaskEditTool,
    begin_mask_edit,
    mask_sha256,
    record_mask_edit,
)
from gyrus.domain.policy import combine_policy, decide_ood, decide_reliability
from gyrus.domain.report_model import build_pdf_report_model
from gyrus.domain.result_schema import (
    AnalysisResult,
    ArrayReference,
    ClassificationResult,
    ExecutionInfo,
    MaskStatistics,
    SpatialResult,
)


def _analysis(case_id: str, model_mask_sha: str) -> AnalysisResult:
    ood = decide_ood(0.1)
    reliability = decide_reliability("glioma", 1.0)
    spatial = SpatialResult(
        probability_map=ArrayReference(
            "probability", (128, 128), "float32", "a" * 64, "probability"
        ),
        model_mask=ArrayReference(
            "model_mask", (128, 128), "uint8", model_mask_sha, "mask"
        ),
        statistics=MaskStatistics(1, 16384, 1 / 16384, 1, (10, 10, 1, 1)),
    )
    return AnalysisResult(
        schema_version=1,
        case_id=case_id,
        stack_manifest_sha256="b" * 64,
        input=InputDescriptor(
            "/tmp/case.png",
            "case.png",
            "PNG",
            "L",
            128,
            128,
            100,
            "c" * 64,
            "d" * 64,
        ),
        execution=ExecutionInfo(
            "cpu",
            "2026-09-30T10:00:00+05:30",
            "2026-09-30T10:00:01+05:30",
            1000,
        ),
        ood=ood,
        classification=ClassificationResult((0.7, 0.1, 0.1, 0.1), "glioma"),
        spatial=spatial,
        reliability=reliability,
        final_policy=combine_policy(ood, reliability),
    )


def test_mask_edit_chain_is_non_destructive_and_hash_linked():
    case_id = str(uuid4())
    model = np.zeros((128, 128), dtype=np.uint8)
    model[10, 10] = 1
    edited = model.copy()
    edited[11, 11] = 1
    provenance = begin_mask_edit(case_id, model)
    updated = record_mask_edit(
        provenance,
        model,
        edited,
        tool=MaskEditTool.BRUSH_ADD,
        timestamp="2026-09-30T10:00:00+05:30",
        parameters={"radius": 3},
    )
    assert updated.model_mask_sha256 == mask_sha256(model)
    assert updated.current_mask_sha256 == mask_sha256(edited)
    assert updated.records[0].added_pixels == 1
    assert updated.reliability_scope == "ORIGINAL_MODEL_MASK_ONLY"
    assert int(model.sum()) == 1

    with pytest.raises(ProvenanceError, match="current edited mask"):
        record_mask_edit(
            updated,
            model,
            edited,
            tool=MaskEditTool.UNDO,
            timestamp="2026-09-30T10:00:01+05:30",
        )


def test_history_is_opt_in_transactional_and_duplicate_safe(tmp_path: Path):
    cases = tmp_path / "cases"
    bundle = cases / "case-1"
    bundle.mkdir(parents=True)
    repository = HistoryRepository(tmp_path / "app.db", cases)
    repository.initialize()
    record = HistoryRecord(
        case_id=str(uuid4()),
        saved_at="2026-09-30T10:00:00+05:30",
        filename="case.png",
        source_sha256="a" * 64,
        predicted_category="glioma",
        ood_status="ACCEPT",
        final_status="STANDARD",
        edited=False,
        bundle_path=str(bundle),
        model_mask_sha256="b" * 64,
    )
    repository.save(record)
    assert repository.get(record.case_id) == record
    assert repository.list_recent() == [record]
    with pytest.raises(HistoryRepositoryError) as captured:
        repository.save(record)
    assert captured.value.code == "HISTORY_DUPLICATE_CASE"


def test_report_model_has_fixed_pages_disclaimer_and_edit_scope():
    case_id = str(uuid4())
    model = np.zeros((128, 128), dtype=np.uint8)
    model[10, 10] = 1
    edited = model.copy()
    edited[10, 11] = 1
    provenance = begin_mask_edit(case_id, model)
    provenance = record_mask_edit(
        provenance,
        model,
        edited,
        tool=MaskEditTool.BRUSH_ADD,
        timestamp="2026-09-30T10:00:00+05:30",
    )
    report = build_pdf_report_model(
        _analysis(case_id, mask_sha256(model)),
        generated_at="2026-09-30T10:05:00+05:30",
        edit_provenance=provenance,
    )
    assert len(report.pages) == 3
    assert all(page.footer == DISCLAIMER for page in report.pages)
    assert report.default_filename == f"gyrus-research-summary_{case_id}.pdf"
    assert report.edit_provenance["reliability_scope"] == "ORIGINAL_MODEL_MASK_ONLY"
