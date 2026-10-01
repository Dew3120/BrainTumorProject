from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from gyrus.worker.process import WorkerSupervisor, WorkerSupervisorError
from gyrus.worker.protocol import AnalyzeJob, AnalyzeSuccess, DeviceName, WorkerConfig

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "runs/final_stack"


def _job():
    return AnalyzeJob(
        job_id=str(uuid4()),
        case_id=str(uuid4()),
        image_path=str(FINAL / "verification/fixtures/elliptic.png"),
        submitted_at=datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def test_spawn_worker_recovers_after_process_termination():
    supervisor = WorkerSupervisor(
        WorkerConfig(
            str(FINAL),
            DeviceName.CPU,
            startup_timeout_seconds=60.0,
            job_timeout_seconds=60.0,
        )
    )
    try:
        first = supervisor.start()
        assert first.generation == 1
        assert first.device == "cpu"
        assert isinstance(supervisor.analyze(_job()), AnalyzeSuccess)

        assert supervisor._process is not None
        supervisor._process.terminate()
        supervisor._process.join(timeout=10.0)
        assert not supervisor._process.is_alive()

        second = supervisor.ensure_ready()
        assert second.generation == 2
        assert second.process_id != first.process_id
        assert second.load_session_id != first.load_session_id
        result = supervisor.analyze(_job())
        assert isinstance(result, AnalyzeSuccess)
        assert result.generation == 2
        assert result.result["classification"]["predicted_category"] == "meningioma"
    finally:
        supervisor.close()


def test_manifest_mismatch_stops_worker_before_ready_or_job():
    supervisor = WorkerSupervisor(
        WorkerConfig(
            str(FINAL),
            DeviceName.CPU,
            expected_manifest_sha256="0" * 64,
            startup_timeout_seconds=30.0,
        )
    )
    try:
        with pytest.raises(WorkerSupervisorError, match="MANIFEST_HASH_MISMATCH"):
            supervisor.start()
        assert supervisor.snapshot is None
        assert supervisor.process_id is None
    finally:
        supervisor.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_spawn_worker_runs_on_explicit_cuda_device():
    supervisor = WorkerSupervisor(
        WorkerConfig(
            str(FINAL),
            DeviceName.CUDA,
            startup_timeout_seconds=60.0,
            job_timeout_seconds=60.0,
        )
    )
    try:
        ready = supervisor.start()
        assert ready.device == "cuda"
        response = supervisor.analyze(_job())
        assert isinstance(response, AnalyzeSuccess)
        assert response.result["execution"]["device"] == "cuda"
    finally:
        supervisor.close()
