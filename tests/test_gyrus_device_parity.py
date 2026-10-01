"""Check frozen CPU/CUDA disclosure and mask parity on non-test fixtures."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
import torch

from gyrus.worker.inference import FrozenInferenceEngine
from gyrus.worker.loader import FrozenWorkerSession
from gyrus.worker.protocol import AnalyzeJob, DeviceName, WorkerConfig

FINAL = Path(__file__).resolve().parents[1] / "runs" / "final_stack"
FIXTURES = FINAL / "verification" / "fixtures"
NUMERIC_TOLERANCE = 1e-4  # QA tolerance only; registered policy thresholds remain unchanged.


def _job(filename: str) -> AnalyzeJob:
    return AnalyzeJob(
        str(uuid4()), str(uuid4()), str(FIXTURES / filename),
        datetime.now().astimezone().isoformat(timespec="seconds"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_cpu_cuda_disclosure_policy_and_hard_masks_match() -> None:
    cpu = FrozenWorkerSession.open(WorkerConfig(str(FINAL), DeviceName.CPU))
    cuda = FrozenWorkerSession.open(WorkerConfig(str(FINAL), DeviceName.CUDA))
    try:
        cpu_engine = FrozenInferenceEngine(cpu, generation=1)
        cuda_engine = FrozenInferenceEngine(cuda, generation=1)
        for filename in ("elliptic.png", "gradient.png"):
            cpu_result = cpu_engine.analyze(_job(filename))
            cuda_result = cuda_engine.analyze(_job(filename))
            a, b = cpu_result.result, cuda_result.result
            assert a["classification"]["predicted_category"] == b["classification"]["predicted_category"]
            np.testing.assert_allclose(
                a["classification"]["probabilities"], b["classification"]["probabilities"],
                rtol=0, atol=NUMERIC_TOLERANCE,
            )
            assert a["ood"]["status"] == b["ood"]["status"]
            assert abs(a["ood"]["score"] - b["ood"]["score"]) <= NUMERIC_TOLERANCE
            assert a["final_policy"]["status"] == b["final_policy"]["status"]
            a_reliability = a["reliability"]["score"]
            b_reliability = b["reliability"]["score"]
            if a_reliability is None:
                assert b_reliability is None
                assert cpu_result.arrays == cuda_result.arrays == ()
            else:
                assert b_reliability is not None
                assert abs(a_reliability - b_reliability) <= NUMERIC_TOLERANCE
                a_arrays = {item.role: item for item in cpu_result.arrays}
                b_arrays = {item.role: item for item in cuda_result.arrays}
                np.testing.assert_array_equal(
                    np.frombuffer(a_arrays["model_mask"].data, dtype=np.uint8),
                    np.frombuffer(b_arrays["model_mask"].data, dtype=np.uint8),
                )
                a_prob = np.frombuffer(a_arrays["probability_map"].data, dtype=np.float64)
                b_prob = np.frombuffer(b_arrays["probability_map"].data, dtype=np.float64)
                assert float(np.max(np.abs(a_prob - b_prob))) <= NUMERIC_TOLERANCE
    finally:
        cuda.close()
        cpu.close()
