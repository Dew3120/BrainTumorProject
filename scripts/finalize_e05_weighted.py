"""Consolidate and close the preregistered E05 weighted-training experiment."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_smoke_test as smoke

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e05_glioma_weighted"
ARTIFACTS = EXPERIMENT / "artifacts"
BASELINE_RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
LABELS = ("glioma", "meningioma", "pituitary")
LAUNCH_GIT_COMMIT = "5402699a1900b4fc3ef8ffeebd7954070cdc1187"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_json_lf(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def write_text_lf(path, value):
    Path(path).write_bytes(value.replace("\r\n", "\n").encode("utf-8"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_by_category(rows, field):
    return {
        label: float(np.mean([row[field] for row in rows if row["category"] == label]))
        for label in LABELS
    }


def paired_group_interval(rows, label):
    members = [row for row in rows if row["category"] == label]
    groups = {}
    for row in members:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    generator = np.random.default_rng(42)
    values = []
    for _ in range(2000):
        sampled = generator.choice(keys, size=len(keys), replace=True)
        delta = [
            item["candidate_dice"] - item["baseline_dice"]
            for key in sampled
            for item in groups[key]
        ]
        values.append(float(np.mean(delta)))
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


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
    run = Path(
        (ROOT / "runs/segmentation_v2_weighted/latest_weighted.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E05 is not in its registered pre-close state")
    training_config = read_json(run / "config.json")
    training_result = read_json(run / "result.json")
    if training_config["split_sha256"] != config["split_sha256"]:
        raise ValueError("E05 split SHA256 mismatch")
    if training_config["test_evaluated"] or training_result["test_evaluated"]:
        raise ValueError("E05 accessed the sealed test partition")
    best = training_result["top3"][0]
    checkpoint = run / "top3_checkpoints" / best["checkpoint"]
    checkpoint_sha256 = sha256(checkpoint)
    selected = read_json(ARTIFACTS / "selected_checkpoint.json")
    if (
        selected["top3_checkpoints"][0]["sha256"].lower() != checkpoint_sha256
        or selected["top3_checkpoints"][0]["epoch"] != training_result["best_epoch"]
    ):
        raise ValueError("Selected checkpoint record mismatch")
    if training_result["best_checkpoint_reload_max_metric_difference"] != 0.0:
        raise ValueError("Selected checkpoint did not reload exactly")

    split_rows = smoke.read_csv(run / "split_snapshot.csv")
    validation_rows = [
        row
        for row in split_rows
        if row["draft_split"] == "validation" and row.get("mask_path", "")
    ]
    if len(validation_rows) != 680 or any(
        row["original_split"] != "train" for row in validation_rows
    ):
        raise ValueError("E05 validation population is not the frozen development set")
    metadata = {normalized_path(row["path"]): row for row in validation_rows}
    baseline = {
        normalized_path(row["path"]): row
        for row in read_csv(BASELINE_RUN / "validation_predictions.csv")
    }
    candidate = {
        normalized_path(row["path"]): row
        for row in read_csv(run / "validation_predictions.csv")
    }
    if set(baseline) != set(candidate) or set(candidate) != set(metadata):
        raise ValueError("Baseline/candidate validation membership mismatch")
    rows = []
    for key in sorted(candidate):
        source = metadata[key]
        old = baseline[key]
        new = candidate[key]
        rows.append(
            {
                "path": source["path"],
                "filename": source["filename"],
                "category": source["label"],
                "similarity_group": source["similarity_group"],
                "baseline_dice": float(old["dice"]),
                "candidate_dice": float(new["dice"]),
                "dice_delta": float(new["dice"]) - float(old["dice"]),
                "baseline_iou": float(old["iou"]),
                "candidate_iou": float(new["iou"]),
                "iou_delta": float(new["iou"]) - float(old["iou"]),
                "reference_pixels": int(new["reference_pixels"]),
                "candidate_predicted_pixels": int(new["predicted_pixels"]),
            }
        )
    baseline_mean = mean_by_category(rows, "baseline_dice")
    candidate_mean = mean_by_category(rows, "candidate_dice")
    delta = {label: candidate_mean[label] - baseline_mean[label] for label in LABELS}
    intervals = {label: paired_group_interval(rows, label) for label in LABELS}
    checks = {
        "glioma": delta["glioma"] >= config["gate"]["glioma_mean_dice_delta_minimum"],
        "meningioma": (
            delta["meningioma"]
            >= config["gate"]["meningioma_mean_dice_delta_minimum"]
        ),
        "pituitary": (
            delta["pituitary"]
            >= config["gate"]["pituitary_mean_dice_delta_minimum"]
        ),
    }
    gate_passed = all(checks.values())

    pixel = read_json(ARTIFACTS / "segmenter_pixel_calibration.json")
    reliability = read_json(ARTIFACTS / "segmenter_case_reliability.json")
    failure_rows = read_csv(run / "segmentation_failure_modes.csv")
    selections = read_csv(run / "visual_audit_selections.csv")
    if pixel["validation_images"] != 680 or reliability["validation_cases"] != 680:
        raise ValueError("E05 follow-up validation population mismatch")
    if pixel["verification"]["test_images_accessed"] != 0:
        raise ValueError("Pixel calibration accessed test images")
    if reliability["verification"]["test_images_accessed"] != 0:
        raise ValueError("Reliability analysis accessed test images")
    failure_counts = {
        label: sum(row["category"] == label for row in failure_rows) for label in LABELS
    }

    copy_names = (
        "worst_validation_by_category.png",
        "best_validation_by_category.png",
        "visual_audit_selections.csv",
        "segmentation_failure_modes.csv",
        "segmentation_failure_modes.md",
        "segmenter_foreground_reliability_diagram.png",
        "segmenter_whole_image_reliability_diagram.png",
        "segmenter_case_reliability_diagram.png",
    )
    for name in copy_names:
        shutil.copy2(run / name, ARTIFACTS / name)
    write_csv(ARTIFACTS / "per_case_comparison.csv", rows)
    visual_review = {
        "status": "PASS_NO_SYSTEMATIC_TECHNICAL_DEFECT",
        "reviewed_panels": [
            "worst_validation_by_category.png",
            "best_validation_by_category.png",
        ],
        "selected_tiles": len(selections),
        "finding": (
            "Best cases show close reference/prediction alignment. Worst cases show "
            "mislocalization, fragmentation, under-coverage, and over-coverage without "
            "a repeated flip, stretch, offset, or image/mask pairing defect."
        ),
        "reviewer": "Codex visual inspection under the user-authorized audit protocol",
        "test_images_accessed": 0,
    }
    write_json_lf(ARTIFACTS / "visual_review.json", visual_review)
    write_json_lf(ARTIFACTS / "run_manifest.json", run_manifest(run))

    figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
    positions = np.arange(len(LABELS))
    width = 0.36
    axis.bar(
        positions - width / 2,
        [baseline_mean[label] for label in LABELS],
        width,
        label="Frozen v1",
    )
    axis.bar(
        positions + width / 2,
        [candidate_mean[label] for label in LABELS],
        width,
        label="E05 weighted",
    )
    axis.set_xticks(positions, LABELS)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Validation image-mean Dice")
    axis.set_title("Frozen v1 versus glioma-weighted retraining")
    axis.legend()
    figure.savefig(ARTIFACTS / "category_dice_comparison.png", dpi=180)
    plt.close(figure)

    result = {
        "schema_version": 1,
        "experiment_id": "e05_glioma_weighted",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "BRISC internal validation development result; not clinical validation",
        "launch_source_git_commit": LAUNCH_GIT_COMMIT,
        "run": str(run),
        "run_name": run.name,
        "training_status": training_result["status"],
        "epochs_completed": training_result["epochs_completed"],
        "selected_epoch": training_result["best_epoch"],
        "selected_checkpoint": checkpoint.name,
        "selected_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_reload_max_metric_difference": training_result[
            "best_checkpoint_reload_max_metric_difference"
        ],
        "validation_cases": 680,
        "test_images_accessed": 0,
        "baseline_mean_dice": baseline_mean,
        "candidate_mean_dice": candidate_mean,
        "mean_dice_delta": delta,
        "paired_group_bootstrap_ci95_delta": intervals,
        "gate": config["gate"],
        "gate_checks": checks,
        "gate_passed": gate_passed,
        "failure_analysis": {
            "dice_below_0_50": len(failure_rows),
            "counts_by_category": failure_counts,
        },
        "visual_audit": visual_review,
        "pixel_calibration": {
            "gate_status": pixel["gate"]["status"],
            "raw_foreground_ece": pixel["metrics"]["aggregate"]["lesion_centered"]["raw"]["pixel_ece"],
            "oof_foreground_ece": pixel["metrics"]["aggregate"]["lesion_centered"]["oof_calibrated"]["pixel_ece"],
            "oof_hard_mask_changed_pixels": pixel["verification"]["oof_changed_mask_pixels"],
        },
        "case_reliability": {
            "gate_status": reliability["gate"]["status"],
            "failure_detection_auroc": reliability["metrics"]["failure_detection_auroc"],
            "reliability_ece": reliability["metrics"]["reliability"]["ece"],
            "brier_skill": reliability["metrics"]["reliability"]["brier_skill"],
            "zero_dice_capture_top20": reliability["metrics"]["top20_risk"],
        },
        "required_followup_complete": True,
        "limitations": [
            "The same internal validation partition was used for model selection and comparison.",
            "Patient independence is unverified.",
            "No external dataset or sealed test image was evaluated.",
            "This result does not establish clinical validity or deployment safety.",
        ],
    }
    write_json_lf(EXPERIMENT / "result.json", result)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    decision_text = f"""# Glioma-weighted retraining decision

**Decision:** {decision}

The selected epoch-{training_result['best_epoch']} candidate changed glioma, meningioma, and pituitary Dice by `{delta['glioma']:+.4f}`, `{delta['meningioma']:+.4f}`, and `{delta['pituitary']:+.4f}`. The fixed E05 gate required glioma `>=+0.0500` and both other categories `>=-0.0100`. The conjunctive gate {'passed' if gate_passed else 'failed and was not relaxed'}.

All required follow-up checks completed. The visual audit found no systematic technical defect. There were {len(failure_rows)} cases below Dice 0.50. Pixel calibration {'passed' if pixel['gate']['status'] == 'PASS' else 'failed'} its locked gate; case-level reliability {reliability['gate']['status'].lower()} its locked gate.

No sealed-test image was accessed, and this internal BRISC result is not clinical validation.
"""
    write_text_lf(EXPERIMENT / "decision.md", decision_text)
    summary = [
        "# Glioma-weighted retraining result",
        "",
        "| Category | v1 Dice | E05 Dice | Delta | 95% group-bootstrap CI | Gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label in LABELS:
        interval = intervals[label]
        summary.append(
            f"| {label} | {baseline_mean[label]:.4f} | {candidate_mean[label]:.4f} | "
            f"{delta[label]:+.4f} | ({interval[0]:+.4f}, {interval[1]:+.4f}) | "
            f"{'pass' if checks[label] else 'fail'} |"
        )
    summary.extend(
        [
            "",
            f"**Conjunctive decision: {'PASS' if gate_passed else 'FAIL'}.**",
            "",
            f"Failures below Dice 0.50: **{len(failure_rows)}**; pixel calibration: **{pixel['gate']['status']}**; case reliability: **{reliability['gate']['status']}**.",
            "",
        ]
    )
    write_text_lf(ARTIFACTS / "summary.md", "\n".join(summary))
    config["status"] = "COMPLETE"
    write_json_lf(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e05_glioma_weighted"
    )
    registered["status"] = "COMPLETE"
    write_json_lf(registry_path, registry)
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
