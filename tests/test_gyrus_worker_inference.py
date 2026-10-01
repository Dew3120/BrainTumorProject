import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from gyrus.domain.constants import ReliabilityReason
from gyrus.worker.inference import FrozenInferenceEngine
from gyrus.worker.loader import FrozenWorkerSession
from gyrus.worker.process import WorkerSupervisorError, _validate_success
from gyrus.worker.protocol import (
    AnalyzeJob,
    DeviceName,
    WorkerConfig,
    WorkerReady,
    WorkerStage,
)

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "runs/final_stack"


def _job(path):
    return AnalyzeJob(
        job_id=str(uuid4()),
        case_id=str(uuid4()),
        image_path=str(path),
        submitted_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


@pytest.fixture(scope="module")
def cpu_engine():
    session = FrozenWorkerSession.open(WorkerConfig(str(FINAL), DeviceName.CPU))
    try:
        yield FrozenInferenceEngine(session, generation=1)
    finally:
        session.close()


def test_cpu_engine_reconstructs_frozen_elliptic_output(cpu_engine):
    expected = json.loads(
        (FINAL / "verification/expected_outputs.json").read_text(encoding="utf-8")
    )["elliptic.png"]
    response = cpu_engine.analyze(
        _job(FINAL / "verification/fixtures/elliptic.png")
    )
    result = response.result
    assert response.diagnostics.stage_order == tuple(stage.value for stage in WorkerStage)
    assert all(not event.disclosure_allowed for event in response.events[:-1])
    assert response.events[-1].disclosure_allowed is True
    assert response.diagnostics.load_session_id == cpu_engine.session.load_session_id

    actual_probabilities = result["classification"]["probabilities"]
    assert actual_probabilities == pytest.approx(
        expected["classifier"]["probabilities"], rel=0.0, abs=1e-12
    )
    assert result["classification"]["predicted_category"] == "meningioma"
    assert result["ood"]["score"] == pytest.approx(
        expected["ood"]["score"], rel=0.0, abs=1e-12
    )
    assert result["reliability"]["score"] == pytest.approx(
        expected["reliability"]["score"], rel=0.0, abs=1e-12
    )
    assert dict(response.diagnostics.ood_components) == pytest.approx(
        expected["ood"]["components"], rel=0.0, abs=1e-12
    )
    assert dict(response.diagnostics.normalized_ood_components) == pytest.approx(
        expected["ood"]["normalized_components"], rel=0.0, abs=1e-12
    )
    assert response.diagnostics.raw_probability_sha256 == expected["segmenter"][
        "raw_probability_sha256"
    ]
    payloads = {item.role: item for item in response.arrays}
    assert payloads["probability_map"].sha256 == expected["segmenter"][
        "calibrated_probability_sha256"
    ]
    assert payloads["model_mask"].sha256 == expected["segmenter"][
        "hard_mask_sha256"
    ]
    probability = np.frombuffer(
        payloads["probability_map"].data, dtype=np.float64
    ).reshape(128, 128)
    mask = np.frombuffer(payloads["model_mask"].data, dtype=np.uint8).reshape(128, 128)
    assert np.isfinite(probability).all()
    assert int(mask.sum()) == expected["segmenter"]["hard_foreground_pixels"]


def test_no_tumor_discards_spatial_payload_and_r02_score(cpu_engine):
    response = cpu_engine.analyze(
        _job(FINAL / "verification/fixtures/gradient.png")
    )
    assert response.result["classification"]["predicted_category"] == "no_tumor"
    assert response.result["spatial"] is None
    assert response.arrays == ()
    assert response.result["reliability"]["score"] is None
    assert (
        response.result["reliability"]["reason"]
        == ReliabilityReason.SEGMENTATION_RELIABILITY_NOT_APPLICABLE.value
    )
    assert response.diagnostics.route is None
    assert response.diagnostics.raw_probability_sha256 is None


def test_one_loaded_session_serves_multiple_jobs(cpu_engine):
    first = cpu_engine.analyze(_job(FINAL / "verification/fixtures/elliptic.png"))
    second = cpu_engine.analyze(_job(FINAL / "verification/fixtures/elliptic.png"))
    assert first.diagnostics.load_session_id == second.diagnostics.load_session_id
    assert first.result["classification"] == second.result["classification"]
    assert first.result["ood"] == second.result["ood"]
    assert [item.sha256 for item in first.arrays] == [
        item.sha256 for item in second.arrays
    ]


def test_parent_rejects_corrupted_ipc_result(cpu_engine):
    job = _job(FINAL / "verification/fixtures/elliptic.png")
    response = cpu_engine.analyze(job)
    ready = WorkerReady(
        generation=1,
        process_id=1,
        device="cpu",
        manifest_sha256=cpu_engine.session.stack.manifest_sha256,
        artifact_count=31,
        load_session_id=cpu_engine.session.load_session_id,
        loaded_at=cpu_engine.session.loaded_at,
    )
    _validate_success(response, job, ready)
    corrupted_result = dict(response.result)
    corrupted_result["stack_manifest_sha256"] = "0" * 64
    with pytest.raises(WorkerSupervisorError, match="PROVENANCE_MISMATCH"):
        _validate_success(replace(response, result=corrupted_result), job, ready)
