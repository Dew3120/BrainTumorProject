"""Capture a synthetic frozen-fixture result for visual UI review."""

from __future__ import annotations

import argparse
import io
import sys
from datetime import datetime
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication

from gyrus.domain.input_contract import (
    preprocess_for_frozen_stack,
    validate_and_decode_image,
)
from gyrus.gui.app import create_application
from gyrus.gui.input_controller import ValidatedSelection
from gyrus.gui.main_window import GyrusMainWindow
from gyrus.gui.state import AppState
from gyrus.worker.inference import FrozenInferenceEngine
from gyrus.worker.loader import FrozenWorkerSession
from gyrus.worker.protocol import AnalyzeJob, DeviceName, WorkerConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    image = root / "runs/final_stack/verification/fixtures/elliptic.png"
    app: QApplication = create_application(["Gyrus QA"])
    session = FrozenWorkerSession.open(WorkerConfig(str(root / "runs/final_stack"), DeviceName.CPU))
    try:
        decoded = validate_and_decode_image(image)
        processed = preprocess_for_frozen_stack(decoded)
        preview = io.BytesIO()
        Image.fromarray(decoded.gray_uint8).save(preview, format="PNG")
        gray = np.rint(processed.segmenter_chw[0] * 255).clip(0, 255).astype("uint8")
        selection = ValidatedSelection(decoded.descriptor, preview.getvalue(), gray.tobytes())
        response = FrozenInferenceEngine(session, 1).analyze(
            AnalyzeJob(
                str(uuid4()), str(uuid4()), str(image),
                datetime.now().astimezone().isoformat(timespec="milliseconds"),
            )
        )
        window = GyrusMainWindow()
        window.show()
        from types import SimpleNamespace

        if not window.on_worker_ready(SimpleNamespace(
            artifact_count=31,
            manifest_sha256=session.stack.manifest_sha256,
            device="cpu",
        )):
            raise RuntimeError("Frozen stack verification failed")
        window._selection = selection
        window.machine.enter(AppState.INPUT_VALIDATING, reason="synthetic QA fixture")
        window.machine.enter(AppState.READY_TO_ANALYZE, reason="synthetic QA fixture")
        window.machine.enter(AppState.ANALYZING, reason="synthetic QA fixture")
        window.on_job_finished(response)
        if window.machine.state != AppState.RESULT_STANDARD:
            raise RuntimeError("Synthetic QA fixture did not reach standard result")
        app.processEvents()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not window.grab().save(str(args.output)):
            raise RuntimeError("Could not save QA screenshot")
        window.close()
        print(args.output)
        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
