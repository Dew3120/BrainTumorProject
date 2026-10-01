import ast
import hashlib
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from gyrus.domain.manifest import EXPECTED_MANIFEST_SHA256
from gyrus.worker.loader import WorkerLoadError, select_device
from gyrus.worker.protocol import (
    AnalyzeJob,
    ArrayPayload,
    DeviceName,
    WorkerConfig,
    WorkerProtocolError,
)

ROOT = Path(__file__).resolve().parents[1]


def _now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def test_worker_config_requires_an_explicit_supported_device():
    config = WorkerConfig(str(ROOT / "runs/final_stack"), DeviceName.CPU)
    assert config.device is DeviceName.CPU
    assert config.expected_manifest_sha256 == EXPECTED_MANIFEST_SHA256
    with pytest.raises(WorkerProtocolError, match="explicitly cpu or cuda"):
        WorkerConfig(str(ROOT / "runs/final_stack"), "auto")


def test_cuda_request_never_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(WorkerLoadError, match="WORKER_CUDA_UNAVAILABLE"):
        select_device(DeviceName.CUDA)
    assert select_device(DeviceName.CPU).type == "cpu"


def test_job_and_array_payload_validate_before_ipc():
    with pytest.raises(WorkerProtocolError, match="job_id"):
        AnalyzeJob("bad", str(uuid4()), "case.png", _now())
    value = np.arange(16, dtype=np.uint8).reshape(4, 4)
    data = value.tobytes()
    payload = ArrayPayload(
        "model_mask",
        "job/model_mask",
        value.shape,
        value.dtype.name,
        hashlib.sha256(data).hexdigest(),
        data,
    )
    assert payload.shape == (4, 4)
    with pytest.raises(WorkerProtocolError, match="hash mismatch"):
        ArrayPayload(
            "model_mask",
            "job/model_mask",
            value.shape,
            value.dtype.name,
            "0" * 64,
            data,
        )
    with pytest.raises(WorkerProtocolError, match="byte count"):
        ArrayPayload(
            "model_mask",
            "job/model_mask",
            (2, 2),
            value.dtype.name,
            hashlib.sha256(data).hexdigest(),
            data,
        )


def test_worker_modules_have_no_qt_or_network_server_imports():
    forbidden_roots = {
        "PySide6",
        "PyQt5",
        "PyQt6",
        "socket",
        "socketserver",
        "http",
        "requests",
        "fastapi",
        "flask",
        "uvicorn",
    }
    paths = sorted((ROOT / "gyrus/worker").glob("*.py"))
    assert paths
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = {node.module.split(".")[0]}
            else:
                continue
            assert names.isdisjoint(forbidden_roots), (path, names & forbidden_roots)


def test_supervisor_source_locks_spawn_and_single_slot_queue():
    source = (ROOT / "gyrus/worker/process.py").read_text(encoding="utf-8")
    assert 'mp.get_context("spawn")' in source
    assert source.count("Queue(maxsize=1)") == 2
    assert "threading.Lock()" in source
    assert "TCP" not in source and "HTTP" not in source
