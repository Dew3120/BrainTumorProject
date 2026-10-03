"""Deterministic, three-page PDF rendering from verified frozen evidence."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import reportlab
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

from .constants import (
    DISCLAIMER,
    OOD_REVIEW_THRESHOLD_TEXT,
    RELIABILITY_THRESHOLD_TEXT,
    SEGMENTATION_THRESHOLD_TEXT,
)
from .explanation import ood_explanation, policy_explanation, reliability_explanation
from .input_contract import preprocess_for_frozen_stack, validate_and_decode_image
from .manifest import EXPECTED_MANIFEST_SHA256
from .mask_provenance import MaskEditProvenance, mask_sha256
from .policy import combine_policy, decide_ood, decide_reliability
from .report_model import PdfReportModel
from .result_schema import ClassificationResult, ExecutionInfo, MaskStatistics


class PdfExportError(ValueError):
    """Missing or inconsistent report evidence must stop preview and export."""


@dataclass(frozen=True, slots=True)
class PdfRenderPayload:
    model: PdfReportModel
    source_path: str
    source_preview_png: bytes
    model_gray_128: bytes
    probability: np.ndarray | None
    model_mask: np.ndarray | None
    edited_mask: np.ndarray | None
    edit_provenance: MaskEditProvenance | None


_INK = colors.HexColor("#17202A")
_MUTED = colors.HexColor("#526273")
_LINE = colors.HexColor("#DCE4EA")
_PALE = colors.HexColor("#F4F8FB")
_CYAN = colors.HexColor("#35D0E0")
_VIOLET = colors.HexColor("#A78BFA")
_PINK = colors.HexColor("#FF6B8A")
_AMBER = colors.HexColor("#F2B84B")
_GREEN = colors.HexColor("#2FC38C")
_PAGE_W, _PAGE_H = A4


def _register_fonts() -> None:
    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, file in (("GyrusSans", "Vera.ttf"), ("GyrusSans-Bold", "VeraBd.ttf")):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verify(payload: PdfRenderPayload) -> tuple[Image.Image, np.ndarray]:
    model = payload.model
    result = model.result
    ClassificationResult(
        tuple(result["classification"]["probabilities"]),
        result["classification"]["predicted_category"],
    )
    ExecutionInfo(**result["execution"])
    if model.case_id != result["case_id"] or result["stack_manifest_sha256"] != EXPECTED_MANIFEST_SHA256:
        raise PdfExportError("REPORT_STACK_OR_CASE_MISMATCH")
    if model.generated_at != result["execution"]["completed_at"]:
        raise PdfExportError("REPORT_NONDETERMINISTIC_TIMESTAMP")
    if model.edit_provenance != (
        None if payload.edit_provenance is None else payload.edit_provenance.to_dict()
    ):
        raise PdfExportError("REPORT_EDIT_PROVENANCE_MISMATCH")
    ood = result["ood"]
    reliability = result["reliability"]
    category = result["classification"]["predicted_category"]
    spatial = result["spatial"]
    if (
        ood["threshold"] != OOD_REVIEW_THRESHOLD_TEXT
        or reliability["threshold"] != RELIABILITY_THRESHOLD_TEXT
        or (spatial is not None and spatial["segmentation_threshold"] != SEGMENTATION_THRESHOLD_TEXT)
    ):
        raise PdfExportError("REPORT_THRESHOLD_DRIFT")
    expected_ood = decide_ood(ood["score"])
    expected_reliability = decide_reliability(category, reliability["score"])
    expected_final = combine_policy(expected_ood, expected_reliability)
    if (
        not expected_ood.valid
        or ood["status"] != expected_ood.status.value
        or ood["reason"] != expected_ood.reason.value
        or reliability["status"] != expected_reliability.status.value
        or reliability["reason"] != expected_reliability.reason.value
        or result["final_policy"]["status"] != expected_final.status.value
        or result["final_policy"]["primary_reason"] != expected_final.primary_reason
    ):
        raise PdfExportError("REPORT_POLICY_MISMATCH")
    source = Path(payload.source_path)
    if source.is_symlink() or not source.is_file():
        raise PdfExportError("REPORT_SOURCE_MISSING")
    decoded = validate_and_decode_image(source)
    descriptor = decoded.descriptor
    if (
        descriptor.source_sha256 != result["input"]["source_sha256"]
        or descriptor.grayscale_sha256 != result["input"]["grayscale_sha256"]
        or descriptor.width != result["input"]["width"]
        or descriptor.height != result["input"]["height"]
        or descriptor.image_format != result["input"]["image_format"]
        or descriptor.source_mode != result["input"]["source_mode"]
        or descriptor.file_bytes != result["input"]["file_bytes"]
    ):
        raise PdfExportError("REPORT_SOURCE_CHANGED")
    try:
        with Image.open(io.BytesIO(payload.source_preview_png)) as preview:
            source_preview = preview.convert("L")
            source_preview.load()
    except Exception as exc:
        raise PdfExportError("REPORT_SOURCE_PREVIEW_INVALID") from exc
    if not np.array_equal(np.asarray(source_preview), decoded.gray_uint8):
        raise PdfExportError("REPORT_SOURCE_PREVIEW_MISMATCH")
    processed = preprocess_for_frozen_stack(decoded)
    gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype(np.uint8)
    if payload.model_gray_128 != gray.tobytes():
        raise PdfExportError("REPORT_PREPROCESS_MISMATCH")
    if spatial is None:
        if any(value is not None for value in (
            payload.probability, payload.model_mask, payload.edited_mask, payload.edit_provenance,
        )):
            raise PdfExportError("REPORT_NO_TUMOR_HAS_MASK")
    else:
        statistics = spatial["statistics"]
        MaskStatistics(
            statistics["foreground_pixels"], statistics["total_pixels"],
            statistics["area_fraction"], statistics["component_count"],
            None if statistics["bounding_box_xywh"] is None else tuple(statistics["bounding_box_xywh"]),
        )
        probability, mask = payload.probability, payload.model_mask
        if probability is None or mask is None or probability.shape != (128, 128) or mask.shape != (128, 128):
            raise PdfExportError("REPORT_SPATIAL_MISSING")
        if probability.dtype != np.float64 or mask.dtype != np.uint8:
            raise PdfExportError("REPORT_ARRAY_DTYPE_MISMATCH")
        if not np.isfinite(probability).all() or not ((0 <= probability) & (probability <= 1)).all():
            raise PdfExportError("REPORT_PROBABILITY_INVALID")
        if not np.isin(mask, (0, 1)).all() or not np.array_equal(mask, probability >= 0.50):
            raise PdfExportError("REPORT_MASK_THRESHOLD_MISMATCH")
        if (
            _hash(probability.tobytes()) != spatial["probability_map"]["sha256"]
            or _hash(mask.tobytes()) != spatial["model_mask"]["sha256"]
            or int(mask.sum()) != spatial["statistics"]["foreground_pixels"]
        ):
            raise PdfExportError("REPORT_ARRAY_REFERENCE_MISMATCH")
    if payload.edited_mask is not None or payload.edit_provenance is not None:
        edited, provenance, mask = payload.edited_mask, payload.edit_provenance, payload.model_mask
        if edited is None or provenance is None or mask is None:
            raise PdfExportError("REPORT_EDIT_INCOMPLETE")
        if (edited.shape != (128, 128) or edited.dtype not in (np.dtype("bool"), np.dtype("uint8"))
            or not np.isin(edited, (0, 1)).all()):
            raise PdfExportError("REPORT_EDIT_ARRAY_INVALID")
        if (
            provenance.case_id != model.case_id
            or provenance.shape != (128, 128)
            or not provenance.edited
            or provenance.model_mask_sha256 != mask_sha256(mask)
            or provenance.current_mask_sha256 != mask_sha256(edited)
            or provenance.reliability_scope != "ORIGINAL_MODEL_MASK_ONLY"
        ):
            raise PdfExportError("REPORT_EDIT_HASH_MISMATCH")
        previous = provenance.model_mask_sha256
        for index, record in enumerate(provenance.records, 1):
            if record.sequence != index or record.before_sha256 != previous:
                raise PdfExportError("REPORT_EDIT_CHAIN_BROKEN")
            previous = record.after_sha256
        if previous != provenance.current_mask_sha256:
            raise PdfExportError("REPORT_EDIT_CHAIN_BROKEN")
    return source_preview, gray


def _fit_text(text: str, width: float, *, font: str = "GyrusSans", size: float = 9) -> list[str]:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split(" "):
            proposed = word if not line else f"{line} {word}"
            if pdfmetrics.stringWidth(proposed, font, size) <= width:
                line = proposed
                continue
            if line:
                lines.append(line)
                line = ""
            while pdfmetrics.stringWidth(word, font, size) > width:
                length = max(1, int(len(word) * width / pdfmetrics.stringWidth(word, font, size)))
                while pdfmetrics.stringWidth(word[:length], font, size) > width:
                    length -= 1
                lines.append(word[:length])
                word = word[length:]
            line = word
        lines.append(line)
    return lines


def _paragraph(pdf: canvas.Canvas, text: str, x: float, top: float, width: float,
               *, size: float = 9, leading: float = 13, color: colors.Color = _INK,
               font: str = "GyrusSans") -> float:
    pdf.setFillColor(color)
    pdf.setFont(font, size)
    for line in _fit_text(text, width, font=font, size=size):
        pdf.drawString(x, top, line)
        top -= leading
    return top


def _page_frame(pdf: canvas.Canvas, page_number: int, title: str, status: str) -> None:
    pdf.setFillColor(colors.white)
    pdf.rect(0, 0, _PAGE_W, _PAGE_H, stroke=0, fill=1)
    pdf.setFillColor(_INK)
    pdf.rect(0, _PAGE_H - 49, _PAGE_W, 49, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("GyrusSans-Bold", 10.5)
    pdf.drawString(42, _PAGE_H - 30, "RESEARCH PROTOTYPE — NOT FOR CLINICAL USE")
    pdf.setFillColor(_INK)
    pdf.setFont("GyrusSans-Bold", 19)
    pdf.drawString(42, _PAGE_H - 82, title)
    if status == "REVIEW_REQUIRED":
        pdf.setFillColor(_AMBER)
        pdf.roundRect(_PAGE_W - 181, _PAGE_H - 86, 139, 22, 5, stroke=0, fill=1)
        pdf.setFillColor(_INK)
        pdf.setFont("GyrusSans-Bold", 8.5)
        pdf.drawCentredString(_PAGE_W - 111.5, _PAGE_H - 78, "REVIEW REQUIRED")
    pdf.setStrokeColor(_LINE)
    pdf.line(42, 62, _PAGE_W - 42, 62)
    pdf.setFillColor(_MUTED)
    pdf.setFont("GyrusSans", 8)
    pdf.drawString(42, 45, DISCLAIMER)
    pdf.drawRightString(_PAGE_W - 42, 45, f"Gyrus 0.1.0  ·  {page_number} / 3")


def _label(pdf: canvas.Canvas, text: str, x: float, y: float) -> None:
    pdf.setFillColor(_MUTED)
    pdf.setFont("GyrusSans-Bold", 8)
    pdf.drawString(x, y, text.upper())


def _image_card(pdf: canvas.Canvas, title: str, image: Image.Image, x: float,
                y: float, width: float, height: float) -> None:
    pdf.setStrokeColor(_LINE)
    pdf.setFillColor(_PALE)
    pdf.roundRect(x, y, width, height, 6, stroke=1, fill=1)
    _label(pdf, title, x + 12, y + height - 21)
    area_w, area_h = width - 24, height - 46
    ratio = min(area_w / image.width, area_h / image.height)
    draw_w, draw_h = image.width * ratio, image.height * ratio
    pdf.drawImage(
        ImageReader(image), x + (width - draw_w) / 2,
        y + 11 + (area_h - draw_h) / 2, draw_w, draw_h,
        preserveAspectRatio=True, mask="auto",
    )


def _gray_image(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray, mode="L").convert("RGB")


def _overlay_image(gray: np.ndarray, mask: np.ndarray | None) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=2).copy()
    if mask is not None:
        active = mask.astype(bool)
        rgb[active] = (rgb[active] * 0.58 + np.array((53, 208, 224)) * 0.42).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _probability_image(probability: np.ndarray) -> Image.Image:
    scaled = np.rint(probability * 255).astype(np.uint8)
    rgb = np.stack((scaled // 3, scaled, np.minimum(scaled.astype(np.uint16) + 35, 255)), axis=-1)
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")


def _difference_image(gray: np.ndarray, model_mask: np.ndarray, edited_mask: np.ndarray) -> Image.Image:
    rgb = np.repeat(gray[..., None], 3, axis=2).copy()
    shared = (model_mask > 0) & (edited_mask > 0)
    added = (model_mask == 0) & (edited_mask > 0)
    removed = (model_mask > 0) & (edited_mask == 0)
    rgb[shared] = (53, 208, 224)
    rgb[added] = (167, 139, 250)
    yy, xx = np.indices(removed.shape)
    hatch = ((xx + yy) % 6 < 3)[removed]
    rgb[removed] = np.where(hatch[:, None], (255, 107, 138), (45, 28, 40))
    return Image.fromarray(rgb, mode="RGB")


def _page_one(pdf: canvas.Canvas, payload: PdfRenderPayload,
              source_preview: Image.Image, gray: np.ndarray) -> None:
    result = payload.model.result
    status = result["final_policy"]["status"]
    _page_frame(pdf, 1, "Research result overview", status)
    pdf.setFillColor(_PALE)
    pdf.roundRect(42, 635, _PAGE_W - 84, 96, 7, stroke=0, fill=1)
    _label(pdf, "Case status", 56, 711)
    pdf.setFillColor(_AMBER if status == "REVIEW_REQUIRED" else _GREEN)
    pdf.setFont("GyrusSans-Bold", 16)
    pdf.drawString(56, 687, status.replace("_", " "))
    _paragraph(pdf, policy_explanation(result), 56, 666,
               _PAGE_W - 112, size=8.5)
    _label(pdf, "Source image", 42, 614)
    filename = result["input"]["filename"]
    _paragraph(pdf, f"{filename}  |  {result['input']['width']} × {result['input']['height']} px  |  "
               f"{result['input']['image_format']} / {result['input']['source_mode']}",
               42, 598, _PAGE_W - 84, size=9)
    _label(pdf, "Model-predicted category", 42, 563)
    category = result["classification"]["predicted_category"]
    _paragraph(pdf, category.replace("_", " ").title(), 42, 546,
               _PAGE_W - 84, size=13, leading=17, font="GyrusSans-Bold")
    probabilities = result["classification"]["probabilities"]
    for index, name in enumerate(("Glioma", "Meningioma", "Pituitary", "No tumor")):
        x = 42 + index * 129
        _label(pdf, name, x, 515)
        pdf.setFillColor(_INK)
        pdf.setFont("GyrusSans-Bold", 11)
        pdf.drawString(x, 496, f"{probabilities[index]:.4f}")
    _image_card(pdf, "Original source preview", source_preview, 42, 213, 247, 257)
    _image_card(pdf, "Frozen model overlay", _overlay_image(gray, payload.model_mask), 306, 213, 247, 257)
    _paragraph(pdf, "Category probabilities and overlay are research outputs. A highlighted area is "
               "not a verified lesion boundary. Review status is determined by the frozen OOD and "
               "segmentation-reliability policy.", 42, 183, _PAGE_W - 84, size=9, leading=13)
    pdf.showPage()


def _page_two(pdf: canvas.Canvas, payload: PdfRenderPayload, gray: np.ndarray) -> None:
    result = payload.model.result
    _page_frame(pdf, 2, "Spatial evidence", result["final_policy"]["status"])
    spatial = result["spatial"]
    if spatial is None:
        _label(pdf, "Spatial output unavailable", 42, 724)
        _paragraph(pdf, "The model-predicted category is no tumor. The frozen workflow did not "
                   "produce a segmentation mask or mask-reliability score for this case.",
                   42, 700, _PAGE_W - 84, size=11, leading=17)
        _image_card(pdf, "Original model-space image", _gray_image(gray), 160, 354, 275, 280)
        _paragraph(pdf, "Pixel-only measurements: not applicable. No edited mask is present.",
                   42, 300, _PAGE_W - 84)
        pdf.showPage()
        return
    stats = spatial["statistics"]
    _label(pdf, "Pixel-only measurements", 42, 725)
    _paragraph(pdf, f"Model foreground: {stats['foreground_pixels']} / {stats['total_pixels']} pixels  |  "
               f"Area fraction: {stats['area_fraction']:.6f}  |  "
               f"Components: {stats['component_count']}", 42, 704, _PAGE_W - 84)
    box = stats["bounding_box_xywh"]
    _paragraph(pdf, f"Model-space bounding box (x, y, width, height): {box if box is not None else 'none'}",
               42, 685, _PAGE_W - 84)
    probability, model_mask = payload.probability, payload.model_mask
    _image_card(pdf, "C03-calibrated probability", _probability_image(probability), 42, 406, 247, 250)
    _image_card(pdf, "Frozen model mask · 0.50", _overlay_image(gray, model_mask), 306, 406, 247, 250)
    if payload.edited_mask is not None:
        _image_card(pdf, "Human-edited mask", _overlay_image(gray, payload.edited_mask), 42, 136, 247, 250)
        _image_card(pdf, "Difference · cyan / violet / pink", _difference_image(gray, model_mask, payload.edited_mask),
                    306, 136, 247, 250)
        _paragraph(pdf, "The human-edited mask is separate from the frozen model mask. "
                   "Reliability describes the original model mask only.", 42, 112, _PAGE_W - 84,
                   size=8, leading=10)
    else:
        _paragraph(pdf, "No human-edited mask was saved. Mask area and dimensions are in model pixels "
                   "only; physical millimeters are not established by this image.",
                   42, 370, _PAGE_W - 84, size=10, leading=15)
    pdf.showPage()


def _page_three(pdf: canvas.Canvas, payload: PdfRenderPayload) -> None:
    model = payload.model
    result = model.result
    _page_frame(pdf, 3, "Understanding this output", result["final_policy"]["status"])
    _label(pdf, "Why this result has its status", 42, 724)
    _paragraph(pdf, policy_explanation(result), 42, 704, _PAGE_W - 84,
               size=10, leading=15)
    ood, reliability = result["ood"], result["reliability"]
    _label(pdf, "Input similarity", 42, 645)
    _paragraph(pdf, ood_explanation(result), 42, 628, _PAGE_W - 84,
               size=9, leading=13)
    _paragraph(pdf, f"Input similarity score: {ood['score']:.4f}", 42, 597,
               _PAGE_W - 84, size=9)
    _label(pdf, "Mask reliability", 42, 574)
    score = "not applicable" if reliability["score"] is None else f"{reliability['score']:.4f}"
    _paragraph(pdf, f"Score: {score}. {reliability_explanation(result)}", 42, 557,
               _PAGE_W - 84, size=9, leading=13)
    _label(pdf, "Exact policy thresholds · audit reference", 42, 506)
    for y, label, value in (
        (488, "Model mask", SEGMENTATION_THRESHOLD_TEXT),
        (472, "Mask reliability", RELIABILITY_THRESHOLD_TEXT),
        (456, "Input similarity review", OOD_REVIEW_THRESHOLD_TEXT),
    ):
        _paragraph(pdf, f"{label}: {value}", 42, y, _PAGE_W - 84, font="Courier", size=8.5)
    _label(pdf, "Technical trace · SHA256", 42, 428)
    for y, label, value in (
        (410, "Frozen stack manifest", result["stack_manifest_sha256"]),
        (378, "Source image", result["input"]["source_sha256"]),
        (346, "Decoded grayscale", result["input"]["grayscale_sha256"]),
    ):
        _label(pdf, label, 42, y)
        _paragraph(pdf, value, 42, y - 14, _PAGE_W - 84, font="Courier", size=7.5)
    _label(pdf, "Execution", 42, 311)
    execution = result["execution"]
    _paragraph(pdf, f"Device: {execution['device']}  |  Completed: {execution['completed_at']}  |  "
               f"Duration: {execution['duration_ms']} ms", 42, 294, _PAGE_W - 84,
               size=8, leading=11)
    _paragraph(pdf, f"Case ID: {model.case_id}  |  Creator: {model.creator}",
               42, 270, _PAGE_W - 84, size=8, leading=11)
    if payload.edit_provenance is not None:
        _paragraph(pdf, f"Human edit records: {len(payload.edit_provenance.records)}  |  "
                   "Reliability scope: original model mask only", 42, 253,
                   _PAGE_W - 84, size=8, leading=11)
    _label(pdf, "Limits of this output", 42, 231)
    limitations = (
        "This research prototype is not a clinical tool and is not validated for patient care.",
        "BRISC patient independence was not verified. External segmentation transfer was weak.",
        "Glioma masks were less reliable than the other internal categories.",
        "External reliability probabilities were miscalibrated; OOD review does not prove a mask is correct.",
        "All measurements here are pixel-based. Physical size and depth are not established.",
    )
    y = 214
    for item in limitations:
        y = _paragraph(pdf, f"• {item}", 42, y, _PAGE_W - 84, size=8, leading=11) - 5
    pdf.showPage()


def render_pdf(payload: PdfRenderPayload) -> bytes:
    """Return byte-identical PDF bytes for byte-identical verified input."""
    source_preview, gray = _verify(payload)
    _register_fonts()
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4, pageCompression=0, invariant=1)
    pdf.setCreator(payload.model.creator)
    pdf.setTitle("Gyrus brain MRI research summary")
    pdf.setAuthor("Gyrus")
    _page_one(pdf, payload, source_preview, gray)
    _page_two(pdf, payload, gray)
    _page_three(pdf, payload)
    pdf.save()
    return output.getvalue()
