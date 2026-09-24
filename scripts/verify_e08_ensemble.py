"""Verify the closed E08 fixed-ensemble experiment and hash its artifacts."""

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e08_ensemble"
ARTIFACTS = EXPERIMENT / "artifacts"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    config = json.loads((EXPERIMENT / "config.json").read_text(encoding="utf-8"))
    result = json.loads((EXPERIMENT / "result.json").read_text(encoding="utf-8"))
    registry = json.loads(
        (ROOT / "experiments/registry.json").read_text(encoding="utf-8")
    )
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e08_ensemble"
    )
    if config["status"] != "COMPLETE" or registered["status"] != "COMPLETE":
        raise ValueError("E08 lifecycle status mismatch")
    if result["status"] != "PASS" or not result["gate_passed"]:
        raise ValueError("E08 expected a closed local pass")
    if result["parent_g1"]["status"] != "FAIL":
        raise ValueError("E08 parent replacement-gate posture changed")
    with (ARTIFACTS / "per_case_comparison.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680 or len({row["path"] for row in rows}) != 680:
        raise ValueError("E08 case population mismatch")
    glioma = [row for row in rows if row["category"] == "glioma"]
    mean = sum(float(row["ensemble_dice"]) for row in glioma) / len(glioma)
    if not math.isclose(
        mean, result["mean_dice"]["ensemble"]["glioma"], abs_tol=1e-12
    ):
        raise ValueError("E08 glioma mean mismatch")
    if result["validation_cases"] != 680 or result["test_images_accessed"] != 0:
        raise ValueError("E08 validation or test-access mismatch")
    required = (
        "hypothesis.md",
        "config.json",
        "result.json",
        "decision.md",
        "artifacts/per_case_comparison.csv",
        "artifacts/dice_comparison.png",
        "artifacts/visual_audit_selections.csv",
        "artifacts/worst_validation_by_category.png",
        "artifacts/best_validation_by_category.png",
        "artifacts/visual_review.json",
        "artifacts/summary.md",
    )
    missing = [name for name in required if not (EXPERIMENT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing E08 artifacts: {missing}")
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file()
        and path != ARTIFACTS / "experiment_manifest.json"
        and path.name != ".gitkeep"
    )
    manifest = {
        "status": "PASS",
        "experiment_id": "e08_ensemble",
        "registered_decision": "PASS_LOCAL_FAIL_PARENT_G1",
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in included
        ],
    }
    payload = json.dumps(manifest, indent=2) + "\n"
    (ARTIFACTS / "experiment_manifest.json").write_bytes(payload.encode("utf-8"))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
