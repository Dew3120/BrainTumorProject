"""Consolidate and close the preregistered E07 routed specialist experiment."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e07_glioma_specialized"
ARTIFACTS = EXPERIMENT / "artifacts"
LABELS = ("glioma", "meningioma", "pituitary")


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def write_text(path, value):
    Path(path).write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def run_manifest(run):
    entries = []
    for path in sorted(item for item in run.rglob("*") if item.is_file()):
        entries.append(
            {
                "path": path.relative_to(run).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return {
        "schema_version": 1,
        "run": str(run),
        "file_count": len(entries),
        "total_bytes": sum(item["bytes"] for item in entries),
        "files": entries,
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E07 is not in its registered pre-close state")
    run = Path(
        (ROOT / "runs/segmentation_v2_glioma_specialized/latest.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    run_config = read_json(run / "config.json")
    run_result_path = run / "result.json"
    run_result = read_json(run_result_path)
    if run_config["split_sha256"] != config["split_sha256"]:
        raise ValueError("E07 split hash mismatch")
    if run_result["status"] != "ROUTED_VALIDATION_COMPLETE":
        raise ValueError("E07 routed validation is incomplete")
    if run_result["gate"]["status"] not in {"PASS", "FAIL"}:
        raise ValueError("E07 routed gate lacks a final status")
    if run_result["test_evaluated"] or run_result["test_images_accessed"]:
        raise ValueError("E07 accessed sealed test data")
    comparison = read_csv(ARTIFACTS / "per_case_comparison.csv")
    if len(comparison) != 680 or len({row["path"] for row in comparison}) != 680:
        raise ValueError("E07 comparison population mismatch")
    assignments = read_csv(run / "routing_assignments.csv")
    if len(assignments) != 680 or len({row["path"] for row in assignments}) != 680:
        raise ValueError("E07 route-assignment population mismatch")
    failure_rows = read_csv(run / "segmentation_failure_modes.csv")
    failure_counts = {
        label: sum(row["category"] == label for row in failure_rows)
        for label in LABELS
    }
    if failure_counts != {"glioma": 65, "meningioma": 9, "pituitary": 15}:
        raise ValueError(f"Unexpected E07 failure counts: {failure_counts}")
    pixel = read_json(ARTIFACTS / "segmenter_pixel_calibration.json")
    reliability = read_json(ARTIFACTS / "segmenter_case_reliability.json")
    if pixel["gate"]["status"] not in {"PASS", "FAIL"}:
        raise ValueError("E07 pixel calibration is incomplete")
    if reliability["gate"]["status"] not in {"PASS", "FAIL"}:
        raise ValueError("E07 reliability evaluation is incomplete")
    visual_review = {
        "status": "PASS_NO_SYSTEMATIC_TECHNICAL_DEFECT",
        "selected_tiles": 42,
        "finding": (
            "Best cases show close reference/prediction alignment. Worst cases show "
            "mislocalization, over-segmentation, and fragmentation without a repeated "
            "flip, stretch, offset, or image/mask pairing defect."
        ),
        "reviewer": "Codex visual inspection under the user-authorized audit protocol",
        "test_images_accessed": 0,
    }
    write_json(ARTIFACTS / "visual_review.json", visual_review)

    copied = (
        "routing.json",
        "routing_assignments.csv",
        "visual_audit_selections.csv",
        "worst_validation_by_category.png",
        "best_validation_by_category.png",
        "segmentation_failure_modes.csv",
        "segmentation_failure_modes.md",
        "segmenter_foreground_reliability_diagram.png",
        "segmenter_whole_image_reliability_diagram.png",
        "segmenter_case_reliability_diagram.png",
    )
    for name in copied:
        shutil.copy2(run / name, ARTIFACTS / name)
    output = {
        "schema_version": 1,
        "experiment_id": "e07_glioma_specialized",
        "status": run_result["gate"]["status"],
        "scope": "BRISC internal validation development result; not clinical validation",
        "training_source_git_commit": run_config["launch_source_git_commit"],
        "routed_evaluation_source_git_commit": "82d30695647bee5b88d58b4c9057def859462586",
        "run": str(run),
        "run_name": run.name,
        "training_status": read_json(run / "specialist_training_result.json")["status"],
        "epochs_completed": run_result["epochs_completed"],
        "selected_epoch": run_result["best_epoch"],
        "selected_checkpoint": run_result["top3"][0]["checkpoint"],
        "selected_checkpoint_sha256": run_result["routing"][
            "specialist_checkpoint_sha256"
        ],
        "checkpoint_reload_max_metric_difference": run_result[
            "best_checkpoint_reload_max_metric_difference"
        ],
        "classifier_recompute_max_probability_difference": run_result[
            "classifier_recompute_max_probability_difference"
        ],
        "validation_cases": 680,
        "test_images_accessed": 0,
        "routing": run_result["routing"],
        "specialist_validation": run_result["specialist_best_validation_metrics"],
        "routed_validation": run_result["best_validation_metrics"],
        "comparison": run_result["comparison"],
        "gate": config["gate"],
        "gate_checks": run_result["gate"]["checks"],
        "gate_passed": run_result["gate"]["status"] == "PASS",
        "failure_analysis": {
            "dice_below_0_50": len(failure_rows),
            "counts_by_category": failure_counts,
        },
        "visual_audit": visual_review,
        "pixel_calibration": {
            "gate_status": pixel["gate"]["status"],
            "raw_foreground_ece": pixel["metrics"]["aggregate"]
            ["lesion_centered"]["raw"]["pixel_ece"],
            "oof_foreground_ece": pixel["metrics"]["aggregate"]
            ["lesion_centered"]["oof_calibrated"]["pixel_ece"],
            "oof_hard_mask_changed_pixels": pixel["verification"]
            ["oof_changed_mask_pixels"],
        },
        "case_reliability": {
            "gate_status": reliability["gate"]["status"],
            "failure_detection_auroc": reliability["metrics"]
            ["failure_detection_auroc"],
            "reliability_ece": reliability["metrics"]["reliability"]["ece"],
            "brier_skill": reliability["metrics"]["reliability"]["brier_skill"],
            "zero_dice_capture_top20": reliability["metrics"]["top20_risk"],
        },
        "required_followup_complete": True,
        "limitations": [
            "The same internal validation partition was used for specialist checkpoint selection and routed comparison.",
            "The classifier route and segmenters were developed on the same BRISC split framework.",
            "Patient independence is unverified.",
            "No external dataset or sealed test image was evaluated.",
            "This result does not establish clinical validity or deployment safety.",
        ],
    }
    write_json(EXPERIMENT / "result.json", output)
    run_result["required_followup_complete"] = True
    run_result["visual_review"] = visual_review["status"]
    write_json(run_result_path, run_result)
    write_json(ARTIFACTS / "run_manifest.json", run_manifest(run))
    config["status"] = "COMPLETE"
    write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    item = next(
        entry
        for entry in registry["experiments"]
        if entry["id"] == "e07_glioma_specialized"
    )
    item["status"] = "COMPLETE"
    write_json(registry_path, registry)
    decision = "ACCEPTED" if output["gate_passed"] else "REJECTED"
    pixel_decision = "passed" if pixel["gate"]["status"] == "PASS" else "failed"
    reliability_decision = (
        "passed" if reliability["gate"]["status"] == "PASS" else "failed"
    )
    delta = output["comparison"]["dice_delta"]
    write_text(
        EXPERIMENT / "decision.md",
        f"""# Glioma-specialist routed segmentation decision

**Decision:** {decision}

The routed system changed glioma Dice by `{delta['glioma']:+.4f}` against the required `+0.0800`. Frozen-classifier route accuracy was `{output['routing']['routing_accuracy']:.4f}` against the required `0.9500`. Meningioma and pituitary changes were `{delta['meningioma']:+.4f}` and `{delta['pituitary']:+.4f}` against their `-0.0100` harm limits. The conjunctive gate {'passed' if output['gate_passed'] else 'failed'} and was not relaxed.

All required follow-up checks completed. Visual review found no systematic technical defect. Pixel calibration {pixel_decision} its locked gate; case reliability {reliability_decision} its locked gate.

No sealed-test image was accessed, and this internal BRISC result is not clinical validation.
""",
    )
    write_text(
        ARTIFACTS / "summary.md",
        f"""# E07 routed glioma-specialist summary

**Registered decision:** {output['status']}

- Routed glioma Dice: `{output['routed_validation']['val_dice_glioma']:.6f}` (`{delta['glioma']:+.6f}` versus v1)
- Classifier routing accuracy: `{output['routing']['routing_accuracy']:.6f}`
- Meningioma/Pituitary Dice changes: `{delta['meningioma']:+.6f}` / `{delta['pituitary']:+.6f}`
- Weak cases: `{len(failure_rows)}`
- Pixel calibration: `{pixel['gate']['status']}`; case reliability: `{reliability['gate']['status']}`
- Sealed-test images accessed: `0`
""",
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
