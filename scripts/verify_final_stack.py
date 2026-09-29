"""Verify the Stage 8 package allowlist, hashes, thresholds, and reconstruction."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

EXPECTED_THRESHOLDS = {
    "segmentation": "0.50",
    "reliability": "0.9490250173235835",
    "ood_review": "0.36854275476100085",
    "ood_refuse": None,
}


def load_runtime(root: Path):
    path = root / "runtime/final_stack_runtime.py"
    spec = importlib.util.spec_from_file_location("verified_final_stack_runtime", path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Frozen runtime loader is unavailable")
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN":
        raise ValueError("Final stack does not declare FROZEN status")
    if manifest["thresholds"] != EXPECTED_THRESHOLDS:
        raise ValueError("A registered threshold drifted")
    if manifest["ood"]["o03_reproduction_max_abs_error"] > 1e-12:
        raise ValueError("O03 reconstruction gate is not satisfied")
    if manifest["ood"]["o05_reproduction_max_abs_error"] > 1e-12:
        raise ValueError("O05 reconstruction gate is not satisfied")
    if manifest["reliability"]["feature_order"] != [
        "empty_prediction",
        "predicted_area_fraction",
        "whole_probability_mean",
        "whole_probability_sd",
        "predicted_region_entropy_mean",
        "boundary_entropy_mean",
        "high_confidence_foreground_fraction",
        "uncertain_pixel_fraction",
        "bounding_box_fill_ratio",
        "normalized_perimeter",
    ]:
        raise ValueError("R02 feature order drifted")
    runtime_module = load_runtime(root)
    expected = json.loads(
        (root / "verification/expected_outputs.json").read_text(encoding="utf-8")
    )
    runtime = runtime_module.FinalStackRuntime(root, verify_hashes=True)
    fixtures = sorted((root / "verification/fixtures").glob("*.png"))
    if [path.name for path in fixtures] != sorted(expected):
        raise ValueError("Verification fixture membership differs")
    try:
        for fixture in fixtures:
            first = runtime.infer_path(fixture)
            second = runtime.infer_path(fixture)
            runtime_module.compare_outputs(first, expected[fixture.name], tolerance=1e-12)
            runtime_module.compare_outputs(second, expected[fixture.name], tolerance=1e-12)
    finally:
        runtime.close()
    return {
        "status": "PASS",
        "artifact_count": len(manifest["artifacts"]),
        "fixture_count": len(fixtures),
        "thresholds": manifest["thresholds"],
        "o03_reproduction_max_abs_error": manifest["ood"]["o03_reproduction_max_abs_error"],
        "o05_reproduction_max_abs_error": manifest["ood"]["o05_reproduction_max_abs_error"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/final_stack"))
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))


if __name__ == "__main__":
    main()
