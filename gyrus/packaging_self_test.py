"""Exercise a packaged CPU worker on synthetic frozen-stack fixtures only."""

from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image
from pypdf import PdfReader

from gyrus.domain.constants import DISCLAIMER
from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.domain.pdf_export import PdfRenderPayload, render_pdf
from gyrus.domain.report_model import build_pdf_report_model_from_payload
from gyrus.domain.result_schema import validate_result_payload
from gyrus.gui.input_controller import ValidatedSelection
from gyrus.gui.presentation import build_presentation
from gyrus.worker.process import WorkerSupervisor
from gyrus.worker.protocol import AnalyzeJob, AnalyzeSuccess, DeviceName, WorkerConfig


def run_packaging_self_test(stack_root: Path, report_path: Path) -> None:
    """Fail if the bundled stack cannot verify and infer in a spawned worker."""

    supervisor = WorkerSupervisor(WorkerConfig(str(stack_root), DeviceName.CPU))
    cases: list[dict[str, str]] = []
    pdf_pages = 0
    try:
        ready = supervisor.start()
        if ready.manifest_sha256 != EXPECTED_MANIFEST_SHA256 or ready.artifact_count != 31:
            raise RuntimeError("Packaged frozen stack did not verify")
        for filename in ("elliptic.png", "gradient.png"):
            image = stack_root / "verification" / "fixtures" / filename
            output = supervisor.analyze(AnalyzeJob(
                str(uuid4()), str(uuid4()), str(image),
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
            ))
            if not isinstance(output, AnalyzeSuccess):
                raise TypeError(f"Packaged worker did not analyze {filename}")
            validate_result_payload(output.result)
            cases.append({
                "fixture": filename,
                "category": output.result["classification"]["predicted_category"],
                "status": output.result["final_policy"]["status"],
                "device": output.result["execution"]["device"],
            })
            if filename == "elliptic.png":
                decoded = validate_and_decode_image(image)
                processed = preprocess_for_frozen_stack(decoded)
                preview = BytesIO()
                Image.fromarray(decoded.gray_uint8).save(preview, format="PNG")
                gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255)
                gray_bytes = gray.astype(np.uint8).tobytes()
                selection = ValidatedSelection(decoded.descriptor, preview.getvalue(), gray_bytes)
                presentation = build_presentation(output, selection)
                model = build_pdf_report_model_from_payload(
                    presentation.result,
                    generated_at=presentation.result["execution"]["completed_at"],
                )
                pdf_bytes = render_pdf(PdfRenderPayload(
                    model, str(image), selection.preview_png, gray_bytes,
                    presentation.probability, presentation.mask, None, None,
                ))
                reader = PdfReader(BytesIO(pdf_bytes))
                pdf_pages = len(reader.pages)
                if pdf_pages != 3 or reader.metadata.creator != "Gyrus 0.1.0":
                    raise RuntimeError("Packaged PDF structure or creator is invalid")
                if any(DISCLAIMER not in page.extract_text() for page in reader.pages):
                    raise RuntimeError("Packaged PDF is missing the research disclaimer")
        if [case["device"] for case in cases] != ["cpu", "cpu"]:
            raise RuntimeError("Packaged self-test did not run on CPU")
        if pdf_pages != 3:
            raise RuntimeError("Packaged PDF was not verified")
        report = {
            "status": "PASS",
            "scope": "Synthetic software reconstruction only; no medical performance claim",
            "manifest_sha256": ready.manifest_sha256,
            "artifact_count": ready.artifact_count,
            "cases": cases,
            "pdf_pages": pdf_pages,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    finally:
        supervisor.close()
