"""Record the preregistered X04 controlled-access stop without touching images."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import stage6_external_common as external

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x04_meningioma_seg_class"
ARTIFACTS = EXPERIMENT / "artifacts"
REPORT = ROOT / "audit/x04_meningioma_seg_class_access_audit.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def local_payload_candidates() -> list[str]:
    external_root = ROOT / "data/external"
    if not external_root.exists():
        return []
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in external_root.iterdir()
        if "meningioma" in path.name.casefold() or "seg-class" in path.name.casefold()
    )


def build_result(*, free_bytes: int, frozen_hashes: dict[str, str]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x04_meningioma_seg_class",
        "status": "ACCESS_BLOCKED",
        "audit_outcome": "ACCESS_BLOCKED",
        "dataset": {
            "name": "Meningioma-SEG-CLASS",
            "version": 1,
            "doi": "10.7937/0TKV-1A36",
            "access_state": "CONTROLLED_ACCESS_NOT_GRANTED",
        },
        "reason": (
            "No verified human-account access grant and no local dataset payload existed after "
            "X01-X03 completed. Amendment 13 requires X04 to be recorded as ACCESS_BLOCKED."
        ),
        "performance_metrics": "NOT_ESTIMABLE: dataset access blocked",
        "dataset_files_accessed": 0,
        "dataset_images_accessed": 0,
        "brisc_sealed_test_images_accessed": 0,
        "neural_retraining": 0,
        "calibrator_refits": 0,
        "reliability_refits": 0,
        "segmentation_threshold": 0.5,
        "reliability_policy_threshold": "0.9490250173235835",
        "threshold_changes": 0,
        "frozen_hashes": frozen_hashes,
        "disk_free_bytes": free_bytes,
        "synthesis_handling": "Omit X04 performance metrics and preserve ACCESS_BLOCKED status.",
    }


def experiment_manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x04_meningioma_seg_class",
        "status": "ACCESS_BLOCKED",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(paths))
        ],
    }


def main() -> None:
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    by_id = {item["id"]: item for item in registry["experiments"]}
    for predecessor in (
        "x01_cheng_overlap_control",
        "x02_openneuro_pituitary",
        "x03_upenn_gbm",
    ):
        if by_id[predecessor]["status"] != "COMPLETE":
            raise ValueError(f"X04 access decision is premature: {predecessor} is not complete")
    if by_id["x04_meningioma_seg_class"]["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("X04 registry state changed")

    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if (
        config["status"] != "REGISTERED_NOT_STARTED"
        or config["dataset"]["access_state"] != "CONTROLLED_ACCESS_PENDING_HUMAN_REQUEST"
        or config["segmentation_threshold"] != 0.5
        or config["reliability_policy_threshold"] != "0.9490250173235835"
    ):
        raise ValueError("X04 registered access or threshold contract changed")
    candidates = local_payload_candidates()
    if candidates:
        raise ValueError(f"Possible X04 payload exists and requires audit: {candidates}")

    frozen_hashes = external.verify_frozen_artifacts()
    amendment = ROOT / "audit/preregistration_amendment_13.md"
    if sha256(amendment) != "728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5":
        raise ValueError("Amendment 13 hash mismatch")
    free_bytes = shutil.disk_usage(ROOT.drive + "\\").free
    result = build_result(free_bytes=free_bytes, frozen_hashes=frozen_hashes)

    audit_path = ARTIFACTS / "access_audit.json"
    write_json(audit_path, result)
    write_json(EXPERIMENT / "result.json", result)
    config.update(
        {
            "status": "ACCESS_BLOCKED",
            "dataset": {
                **config["dataset"],
                "access_state": "CONTROLLED_ACCESS_NOT_GRANTED",
                "audit_outcome": "ACCESS_BLOCKED",
            },
            "r02_external_inference_state": "OPERATIONAL_ARTIFACT_FROZEN_NOT_USED",
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    write_json(config_path, config)

    decision = """# X04 decision

**Decision:** `ACCESS_BLOCKED`

No verified controlled-access grant or local Meningioma-SEG-CLASS payload existed after X01-X03 completed. Under Amendment 13, X04 therefore ends without image access or performance metrics. X05 must preserve this status and must not infer, impute, or pool meningioma transport performance.
"""
    (EXPERIMENT / "decision.md").write_text(decision, encoding="utf-8", newline="\n")
    report = f"""# X04 Meningioma-SEG-CLASS access audit

**Audit outcome:** `ACCESS_BLOCKED`

The registered TCIA Meningioma-SEG-CLASS version 1 cohort requires a verified human-account access grant. No grant was confirmed and no local payload was present after X01-X03 completed. Amendment 13 therefore requires this terminal access outcome.

## Verification

- Dataset files opened: 0
- Dataset images opened: 0
- Performance metrics: `NOT_ESTIMABLE`
- BRISC sealed-test images opened: 0
- Neural retraining, calibration refitting, reliability refitting: 0
- Segmentation threshold: `0.50`, unchanged
- Reliability policy threshold: `0.9490250173235835`, unchanged
- Frozen R02/C03 artifact hashes: verified
- Free disk at decision: {free_bytes:,} bytes

## Consequence

X05 may proceed and must list X04 as access-blocked. It must not treat missing X04 metrics as a failure, pass, zero, or independent meningioma result. Human-controlled access can support a future separately registered evaluation, but it cannot be backfilled into this completed Stage 6 run.
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    by_id["x04_meningioma_seg_class"]["status"] = "ACCESS_BLOCKED"
    write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Close X04 as controlled-access blocked

- **Decision made:** Record X04 as `ACCESS_BLOCKED` after X01-X03 completion and omit performance metrics from X05.
- **Triggering artifact:** `experiments/x04_meningioma_seg_class/artifacts/access_audit.json`, SHA256 `{sha256(audit_path)}`.
- **Rationale:** No verified human-account access grant and no local Meningioma-SEG-CLASS payload existed. Amendment 13 explicitly predeclares this outcome and permits synthesis to continue.
- **Verification:** Frozen C03/R02 hashes passed. Dataset files and images opened, BRISC sealed-test access, retraining/refitting, threshold changes, and gate relaxations were all zero.
- **Alternatives considered:** Claim or simulate access; rejected. Impute meningioma metrics; rejected. Delay synthesis indefinitely; rejected by the registered access-blocked rule.
- **Who approved:** User approved Amendment 13 and instructed X04 to be marked `ACCESS_BLOCKED` if access remained unavailable after X01-X03.
"""
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(review_entry)

    manifest_path = ARTIFACTS / "experiment_manifest.json"
    tracked = [
        amendment,
        REPORT,
        ROOT / "scripts/record_x04_access_blocked.py",
        ROOT / "scripts/validate_experiment_tracking.py",
        ROOT / "tests/test_record_x04_access_blocked.py",
        ROOT / "tests/test_foundation_controls.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        EXPERIMENT / "result.json",
        EXPERIMENT / "decision.md",
        audit_path,
    ]
    write_json(manifest_path, experiment_manifest(tracked))

    subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, check=True, capture_output=True, text=True
    )


if __name__ == "__main__":
    main()
