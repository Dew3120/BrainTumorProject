"""Read-only loading of the verified frozen stack on one explicit device."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime
from types import ModuleType
from uuid import uuid4

import torch

from gyrus.domain.manifest import VerifiedFrozenStack, verify_frozen_stack

from .protocol import DeviceName, WorkerConfig


class WorkerLoadError(RuntimeError):
    """The worker cannot establish a verified, usable frozen session."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.safe_message = message


def select_device(requested: DeviceName) -> torch.device:
    """Resolve exactly the requested device; never fall back silently."""

    if requested == DeviceName.CPU:
        return torch.device("cpu")
    if requested != DeviceName.CUDA:
        raise WorkerLoadError("WORKER_DEVICE_INVALID", repr(requested))
    if not torch.cuda.is_available():
        raise WorkerLoadError(
            "WORKER_CUDA_UNAVAILABLE",
            "CUDA was requested, but PyTorch cannot access a CUDA device.",
        )
    try:
        device = torch.device("cuda")
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
    except Exception as exc:
        raise WorkerLoadError(
            "WORKER_CUDA_INITIALIZATION_FAILED",
            f"CUDA initialization failed: {type(exc).__name__}",
        ) from exc
    return device


def _load_verified_runtime(stack: VerifiedFrozenStack) -> ModuleType:
    runtime_path = stack.root / "runtime/final_stack_runtime.py"
    module_name = f"gyrus_verified_runtime_{stack.manifest_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, runtime_path)
    if spec is None or spec.loader is None:
        raise WorkerLoadError(
            "WORKER_RUNTIME_IMPORT_FAILED",
            "The verified frozen runtime could not be imported.",
        )
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise WorkerLoadError(
            "WORKER_RUNTIME_IMPORT_FAILED",
            f"The verified runtime failed to initialize: {type(exc).__name__}",
        ) from exc
    finally:
        sys.dont_write_bytecode = previous
    return module


@dataclass(slots=True)
class FrozenWorkerSession:
    """All frozen models and states loaded exactly once for one worker generation."""

    stack: VerifiedFrozenStack
    runtime_module: ModuleType
    runtime: object
    device: torch.device
    load_session_id: str
    loaded_at: str
    _closed: bool = False

    @classmethod
    def open(cls, config: WorkerConfig) -> FrozenWorkerSession:
        stack = verify_frozen_stack(
            config.stack_root,
            expected_manifest_sha256=config.expected_manifest_sha256,
        )
        device = select_device(config.device)
        runtime_module = _load_verified_runtime(stack)
        try:
            runtime = runtime_module.FinalStackRuntime(stack.root, verify_hashes=False)
            runtime.classifier.to(device).eval()
            for model in runtime.segmenters.values():
                model.to(device).eval()
            for model in (runtime.classifier, *runtime.segmenters.values()):
                if any(parameter.device.type != device.type for parameter in model.parameters()):
                    raise WorkerLoadError(
                        "WORKER_MODEL_DEVICE_MISMATCH",
                        f"A frozen model did not load on the requested {device.type} device.",
                    )
            # Detect time-of-check/time-of-use changes and accidental bytecode output.
            verify_frozen_stack(
                stack.root,
                expected_manifest_sha256=config.expected_manifest_sha256,
            )
        except WorkerLoadError:
            if "runtime" in locals():
                runtime.close()
            raise
        except Exception as exc:
            if "runtime" in locals():
                runtime.close()
            raise WorkerLoadError(
                "WORKER_MODEL_LOAD_FAILED",
                f"Frozen model loading failed: {type(exc).__name__}",
            ) from exc
        return cls(
            stack=stack,
            runtime_module=runtime_module,
            runtime=runtime,
            device=device,
            load_session_id=str(uuid4()),
            loaded_at=datetime.now().astimezone().isoformat(timespec="milliseconds"),
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.runtime.close()
        finally:
            self._closed = True
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
