"""Deterministic report, frozen evidence, preview, and local-save contracts."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from pypdf import PdfReader

from gyrus.domain.constants import (
    DISCLAIMER,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.domain.mask_editor import MaskEditSession
from gyrus.domain.pdf_export import (
    PdfExportError,
    PdfRenderPayload,
    _difference_image,
    render_pdf,
)
from gyrus.domain.report_model import build_pdf_report_model_from_payload
from gyrus.gui.input_controller import ValidatedSelection
from gyrus.gui.pdf_preview import PdfPreviewDialog, PdfSaveError, save_pdf_atomic
from gyrus.gui.presentation import build_presentation
from gyrus.worker.process import WorkerSupervisor
from gyrus.worker.protocol import AnalyzeJob, DeviceName, WorkerConfig

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "runs/final_stack/verification/fixtures"


@pytest.fixture(scope="module")
def reports():
    supervisor = WorkerSupervisor(WorkerConfig(str(ROOT / "runs/final_stack"), DeviceName.CPU))
    try:
        supervisor.start()
        payloads = {}
        for name in ("elliptic.png", "gradient.png"):
            decoded = validate_and_decode_image(FIXTURES / name)
            processed = preprocess_for_frozen_stack(decoded)
            import PIL.Image

            buffer = io.BytesIO()
            PIL.Image.fromarray(decoded.gray_uint8).save(buffer, format="PNG")
            gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype(np.uint8).tobytes()
            selection = ValidatedSelection(decoded.descriptor, buffer.getvalue(), gray)
            response = supervisor.analyze(AnalyzeJob(
                str(uuid4()), str(uuid4()), str(FIXTURES / name),
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
            ))
            presentation = build_presentation(response, selection)
            result = presentation.result
            report = build_pdf_report_model_from_payload(
                result, generated_at=result["execution"]["completed_at"],
            )
            payloads[name] = PdfRenderPayload(
                report, str(FIXTURES / name), selection.preview_png, gray,
                presentation.probability, presentation.mask, None, None,
            )
        yield payloads
    finally:
        supervisor.close()


def test_tumor_pdf_is_three_pages_deterministic_and_traceable(reports):
    payload = reports["elliptic.png"]
    first = render_pdf(payload)
    assert first == render_pdf(payload)
    reader = PdfReader(io.BytesIO(first))
    assert len(reader.pages) == 3
    assert reader.metadata.creator == "Gyrus 0.1.0"
    assert not reader.get_fields()
    pages = [page.extract_text() for page in reader.pages]
    for page in pages:
        assert "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE" in page
        assert DISCLAIMER in page
    assert payload.model.result["classification"]["predicted_category"] in pages[0]
    assert "FROZEN MODEL MASK" in pages[1]
    assert payload.model.result["stack_manifest_sha256"] in pages[2]
    assert payload.model.result["input"]["source_sha256"] in pages[2]
    for threshold in (SEGMENTATION_THRESHOLD_TEXT, RELIABILITY_THRESHOLD_TEXT, OOD_REVIEW_THRESHOLD_TEXT):
        assert threshold in pages[2]
    assert "REVIEW_REQUIRED" not in pages[0]  # synthetic elliptic fixture is standard


def test_no_tumor_pdf_reports_absent_mask_without_fabricating_measurements(reports):
    pdf = render_pdf(reports["gradient.png"])
    pages = [page.extract_text() for page in PdfReader(io.BytesIO(pdf)).pages]
    assert all("REVIEW_REQUIRED" in page for page in pages)
    assert "did not produce a segmentation mask" in " ".join(pages[1].split())
    assert "not applicable" in pages[1]


def test_saved_human_layer_is_separate_and_keeps_frozen_reliability(reports):
    original = reports["elliptic.png"]
    result = original.model.result
    session = MaskEditSession(
        result["case_id"], original.model_mask, result["reliability"]["score"],
    )
    assert session.brush([(0, 0)], 3, add=True)
    session.commit()
    report = build_pdf_report_model_from_payload(
        result, generated_at=result["execution"]["completed_at"],
        edit_provenance=session.saved_provenance,
    )
    edited = replace(original, model=report, edited_mask=session.saved_mask,
                     edit_provenance=session.saved_provenance)
    page_two = PdfReader(io.BytesIO(render_pdf(edited))).pages[1].extract_text()
    assert "HUMAN-EDITED MASK" in page_two
    assert "DIFFERENCE" in page_two
    assert "original model mask" in page_two
    assert result["reliability"]["score"] == original.model.result["reliability"]["score"]


def test_pdf_refuses_threshold_array_and_source_mismatches(tmp_path, reports):
    payload = reports["elliptic.png"]
    altered = payload.probability.copy()
    altered[0, 0] = 1.0 - altered[0, 0]
    with pytest.raises(PdfExportError, match="REPORT_"):
        render_pdf(replace(payload, probability=altered))
    result = __import__("copy").deepcopy(payload.model.result)
    result["ood"]["threshold"] = "0.37"
    with pytest.raises(PdfExportError, match="REPORT_THRESHOLD_DRIFT"):
        render_pdf(replace(payload, model=replace(payload.model, result=result)))
    changed = tmp_path / "elliptic.png"
    changed.write_bytes((FIXTURES / "gradient.png").read_bytes())
    with pytest.raises(PdfExportError, match="REPORT_SOURCE_CHANGED"):
        render_pdf(replace(payload, source_path=str(changed)))


def test_preview_loads_three_pages_and_atomic_local_save(tmp_path, qtbot, reports):
    payload = reports["elliptic.png"]
    pdf = render_pdf(payload)
    dialog = PdfPreviewDialog(pdf, payload.model.default_filename)
    qtbot.addWidget(dialog)
    assert dialog.document.pageCount() == 3
    assert dialog.save_button.isEnabled()
    path = tmp_path / payload.model.default_filename
    assert save_pdf_atomic(pdf, path).read_bytes() == pdf
    with pytest.raises(PdfSaveError, match="already exists"):
        save_pdf_atomic(pdf, path)
    assert save_pdf_atomic(pdf, path, overwrite=True).read_bytes() == pdf
    with pytest.raises(PdfSaveError):
        save_pdf_atomic(b"invalid", tmp_path / "gyrus-research-summary_bad.pdf")
    cloud = tmp_path / "OneDrive" / "export.pdf"
    cloud.parent.mkdir()
    with pytest.raises(PdfSaveError, match="cloud folder"):
        save_pdf_atomic(pdf, cloud)


def test_pdf_difference_uses_distinct_addition_and_removal_colors():
    gray = np.zeros((128, 128), dtype=np.uint8)
    model = np.zeros_like(gray)
    edited = np.zeros_like(gray)
    model[0, 0] = 1
    edited[1, 1] = 1
    rgb = np.asarray(_difference_image(gray, model, edited))
    assert tuple(rgb[0, 0]) == (255, 107, 138)
    assert tuple(rgb[1, 1]) == (167, 139, 250)
