"""Compute boundary F1, consolidate follow-up evidence, and close E06."""

import json
import shutil
from pathlib import Path

import analyze_glioma_failures as analysis
import finalize_e05_weighted as common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e06_boundary_loss"
ARTIFACTS = EXPERIMENT / "artifacts"
BASELINE_RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
BASELINE_CHECKPOINT = (
    BASELINE_RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
)
LABELS = ("glioma", "meningioma", "pituitary")
STRUCTURE = np.ones((3, 3), dtype=bool)
LAUNCH_GIT_COMMIT = "7251f0592a7ba0af1f4964ff96fac9e8cd1d32a9"


def boundary_f1(prediction, reference, tolerance=2):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    predicted_boundary = prediction ^ ndimage.binary_erosion(
        prediction, structure=STRUCTURE
    )
    reference_boundary = reference ^ ndimage.binary_erosion(
        reference, structure=STRUCTURE
    )
    if not predicted_boundary.any() or not reference_boundary.any():
        return 0.0
    reference_tolerance = ndimage.binary_dilation(
        reference_boundary, structure=STRUCTURE, iterations=tolerance
    )
    predicted_tolerance = ndimage.binary_dilation(
        predicted_boundary, structure=STRUCTURE, iterations=tolerance
    )
    precision = float(reference_tolerance[predicted_boundary].mean())
    recall = float(predicted_tolerance[reference_boundary].mean())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def paired_group_interval(rows, label, field):
    members = [row for row in rows if row["category"] == label]
    groups = {}
    for row in members:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    generator = np.random.default_rng(42)
    samples = []
    for _ in range(2000):
        selected = generator.choice(keys, size=len(keys), replace=True)
        values = [item[field] for key in selected for item in groups[key]]
        samples.append(float(np.mean(values)))
    return [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def infer(checkpoint, validation_rows, device):
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    return analysis.infer_all(model, validation_rows, device)


def compact_run_manifest(run):
    excluded_cache = []
    files = []
    for path in sorted(item for item in run.rglob("*") if item.is_file()):
        relative = path.relative_to(run).as_posix()
        if relative.startswith("distance_cache/"):
            excluded_cache.append(path)
            continue
        files.append(
            {
                "path": relative,
                "sha256": common.sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    cache_manifest = run / "distance_cache_manifest.csv"
    return {
        "schema_version": 1,
        "run": str(run),
        "hashed_files": files,
        "distance_cache": {
            "derived_files_excluded_from_individual_rehash": len(excluded_cache),
            "derived_bytes": sum(path.stat().st_size for path in excluded_cache),
            "manifest_path": str(cache_manifest),
            "manifest_sha256": common.sha256(cache_manifest),
            "regeneration_source": "training_script.py and split_snapshot.csv",
        },
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    run = Path(
        (ROOT / "runs/segmentation_v2_boundary_loss/latest_boundary.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    experiment_config_path = EXPERIMENT / "config.json"
    experiment_config = common.read_json(experiment_config_path)
    if experiment_config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E06 is not in its registered pre-close state")
    training_config = common.read_json(run / "config.json")
    training_result = common.read_json(run / "result.json")
    if training_config["split_sha256"] != experiment_config["split_sha256"]:
        raise ValueError("E06 split SHA256 mismatch")
    if training_config["test_evaluated"] or training_result["test_evaluated"]:
        raise ValueError("E06 accessed the sealed test partition")
    best = training_result["top3"][0]
    checkpoint = run / "top3_checkpoints" / best["checkpoint"]
    checkpoint_sha256 = common.sha256(checkpoint)
    selected = common.read_json(ARTIFACTS / "selected_checkpoint.json")
    if selected["top3_checkpoints"][0]["sha256"].lower() != checkpoint_sha256:
        raise ValueError("E06 selected checkpoint hash mismatch")
    if training_result["best_checkpoint_reload_max_metric_difference"] != 0.0:
        raise ValueError("E06 selected checkpoint reload mismatch")

    split_rows = smoke.read_csv(run / "split_snapshot.csv")
    validation_rows = [
        row
        for row in split_rows
        if row["draft_split"] == "validation" and row.get("mask_path", "")
    ]
    if len(validation_rows) != 680 or any(
        row["original_split"] != "train" for row in validation_rows
    ):
        raise ValueError("E06 validation population mismatch")
    metadata = {analysis.normalized_path(row["path"]): row for row in validation_rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, baseline_records, baseline_masks, baseline_references, _ = infer(
        BASELINE_CHECKPOINT, validation_rows, device
    )
    _, candidate_records, candidate_masks, candidate_references, _ = infer(
        checkpoint, validation_rows, device
    )
    baseline_by_path = {
        analysis.normalized_path(row["path"]): row for row in baseline_records
    }
    candidate_by_path = {
        analysis.normalized_path(row["path"]): row for row in candidate_records
    }
    stored_baseline = {
        analysis.normalized_path(row["path"]): row
        for row in common.read_csv(BASELINE_RUN / "validation_predictions.csv")
    }
    stored_candidate = {
        analysis.normalized_path(row["path"]): row
        for row in common.read_csv(run / "validation_predictions.csv")
    }
    if not (
        set(metadata)
        == set(baseline_by_path)
        == set(candidate_by_path)
        == set(stored_baseline)
        == set(stored_candidate)
    ):
        raise ValueError("E06 comparison membership mismatch")
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    rows = []
    maximum_metric_difference = 0.0
    for key in sorted(metadata):
        old = baseline_by_path[key]
        new = candidate_by_path[key]
        differences = (
            abs(old["dice"] - float(stored_baseline[key]["dice"])),
            abs(old["iou"] - float(stored_baseline[key]["iou"])),
            abs(new["dice"] - float(stored_candidate[key]["dice"])),
            abs(new["iou"] - float(stored_candidate[key]["iou"])),
        )
        maximum_metric_difference = max(maximum_metric_difference, *differences)
        if maximum_metric_difference > tolerance:
            raise ValueError(f"E06 frozen metric mismatch: {metadata[key]['path']}")
        if not np.array_equal(baseline_references[key], candidate_references[key]):
            raise ValueError(f"E06 reference mismatch: {metadata[key]['path']}")
        baseline_boundary = boundary_f1(
            baseline_masks[key], baseline_references[key]
        )
        candidate_boundary = boundary_f1(
            candidate_masks[key], candidate_references[key]
        )
        rows.append(
            {
                "path": metadata[key]["path"],
                "filename": metadata[key]["filename"],
                "category": metadata[key]["label"],
                "similarity_group": metadata[key]["similarity_group"],
                "baseline_dice": old["dice"],
                "candidate_dice": new["dice"],
                "dice_delta": new["dice"] - old["dice"],
                "baseline_iou": old["iou"],
                "candidate_iou": new["iou"],
                "iou_delta": new["iou"] - old["iou"],
                "baseline_boundary_f1": baseline_boundary,
                "candidate_boundary_f1": candidate_boundary,
                "boundary_f1_delta": candidate_boundary - baseline_boundary,
            }
        )

    baseline_dice = common.mean_by_category(rows, "baseline_dice")
    candidate_dice = common.mean_by_category(rows, "candidate_dice")
    dice_delta = {
        label: candidate_dice[label] - baseline_dice[label] for label in LABELS
    }
    baseline_boundary = common.mean_by_category(rows, "baseline_boundary_f1")
    candidate_boundary = common.mean_by_category(rows, "candidate_boundary_f1")
    boundary_delta = {
        label: candidate_boundary[label] - baseline_boundary[label]
        for label in LABELS
    }
    dice_intervals = {
        label: paired_group_interval(rows, label, "dice_delta") for label in LABELS
    }
    boundary_intervals = {
        label: paired_group_interval(rows, label, "boundary_f1_delta")
        for label in LABELS
    }
    checks = {
        "glioma_dice": (
            dice_delta["glioma"]
            >= experiment_config["gate"]["glioma_mean_dice_delta_minimum"]
        ),
        "glioma_boundary_f1": (
            boundary_delta["glioma"]
            >= experiment_config["gate"]["glioma_boundary_f1_delta_minimum"]
        ),
    }
    gate_passed = all(checks.values())
    pixel = common.read_json(ARTIFACTS / "segmenter_pixel_calibration.json")
    reliability = common.read_json(ARTIFACTS / "segmenter_case_reliability.json")
    failure_rows = common.read_csv(run / "segmentation_failure_modes.csv")
    selections = common.read_csv(run / "visual_audit_selections.csv")
    if pixel["validation_images"] != 680 or reliability["validation_cases"] != 680:
        raise ValueError("E06 follow-up population mismatch")
    if (
        pixel["verification"]["test_images_accessed"] != 0
        or reliability["verification"]["test_images_accessed"] != 0
    ):
        raise ValueError("E06 follow-up accessed test images")

    for name in (
        "worst_validation_by_category.png",
        "best_validation_by_category.png",
        "visual_audit_selections.csv",
        "segmentation_failure_modes.csv",
        "segmentation_failure_modes.md",
        "segmenter_foreground_reliability_diagram.png",
        "segmenter_whole_image_reliability_diagram.png",
        "segmenter_case_reliability_diagram.png",
    ):
        shutil.copy2(run / name, ARTIFACTS / name)
    common.write_csv(ARTIFACTS / "per_case_comparison.csv", rows)
    visual_review = {
        "status": "PASS_NO_SYSTEMATIC_TECHNICAL_DEFECT",
        "selected_tiles": len(selections),
        "finding": (
            "Best cases show close reference/prediction alignment. Worst cases show "
            "mislocalization, fragmentation, and coverage errors without a repeated "
            "flip, stretch, offset, or image/mask pairing defect."
        ),
        "reviewer": "Codex visual inspection under the user-authorized audit protocol",
        "test_images_accessed": 0,
    }
    common.write_json_lf(ARTIFACTS / "visual_review.json", visual_review)
    common.write_json_lf(ARTIFACTS / "run_manifest.json", compact_run_manifest(run))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    positions = np.arange(len(LABELS))
    width = 0.36
    for axis, baseline, candidate, title, ylabel in (
        (axes[0], baseline_dice, candidate_dice, "Dice", "Image-mean Dice"),
        (
            axes[1],
            baseline_boundary,
            candidate_boundary,
            "Boundary F1",
            "Image-mean boundary F1",
        ),
    ):
        axis.bar(
            positions - width / 2,
            [baseline[label] for label in LABELS],
            width,
            label="Frozen v1",
        )
        axis.bar(
            positions + width / 2,
            [candidate[label] for label in LABELS],
            width,
            label="E06 boundary loss",
        )
        axis.set_xticks(positions, LABELS)
        axis.set_ylim(0, 1)
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.legend()
    figure.savefig(ARTIFACTS / "dice_and_boundary_f1_comparison.png", dpi=180)
    plt.close(figure)

    failure_counts = {
        label: sum(row["category"] == label for row in failure_rows) for label in LABELS
    }
    result = {
        "schema_version": 1,
        "experiment_id": "e06_boundary_loss",
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
        "recomputed_max_metric_difference": maximum_metric_difference,
        "validation_cases": len(rows),
        "test_images_accessed": 0,
        "dice": {
            "baseline": baseline_dice,
            "candidate": candidate_dice,
            "delta": dice_delta,
            "paired_group_bootstrap_ci95_delta": dice_intervals,
        },
        "boundary_f1": {
            "baseline": baseline_boundary,
            "candidate": candidate_boundary,
            "delta": boundary_delta,
            "paired_group_bootstrap_ci95_delta": boundary_intervals,
            "tolerance_pixels": 2,
        },
        "gate": experiment_config["gate"],
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
    common.write_json_lf(EXPERIMENT / "result.json", result)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    decision_text = f"""# Boundary-aware retraining decision

**Decision:** {decision}

The selected epoch-{training_result['best_epoch']} candidate changed glioma Dice by `{dice_delta['glioma']:+.4f}` and glioma boundary F1 by `{boundary_delta['glioma']:+.4f}`. The fixed gate required both changes to be at least `+0.0500`. The conjunctive gate {'passed' if gate_passed else 'failed and was not relaxed'}.

All required follow-up checks completed. Visual review found no systematic technical defect. There were {len(failure_rows)} cases below Dice 0.50. Pixel calibration {pixel['gate']['status'].lower()} its locked gate; case reliability {reliability['gate']['status'].lower()} its locked gate.

No sealed-test image was accessed, and this internal BRISC result is not clinical validation.
"""
    common.write_text_lf(EXPERIMENT / "decision.md", decision_text)
    summary = [
        "# Boundary-aware retraining result",
        "",
        "| Metric | v1 | E06 | Delta | 95% group-bootstrap CI | Gate |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| Glioma Dice | {baseline_dice['glioma']:.4f} | "
            f"{candidate_dice['glioma']:.4f} | {dice_delta['glioma']:+.4f} | "
            f"({dice_intervals['glioma'][0]:+.4f}, {dice_intervals['glioma'][1]:+.4f}) | "
            f"{'pass' if checks['glioma_dice'] else 'fail'} |"
        ),
        (
            f"| Glioma boundary F1 | {baseline_boundary['glioma']:.4f} | "
            f"{candidate_boundary['glioma']:.4f} | {boundary_delta['glioma']:+.4f} | "
            f"({boundary_intervals['glioma'][0]:+.4f}, {boundary_intervals['glioma'][1]:+.4f}) | "
            f"{'pass' if checks['glioma_boundary_f1'] else 'fail'} |"
        ),
        "",
        f"**Conjunctive decision: {'PASS' if gate_passed else 'FAIL'}.**",
        "",
    ]
    common.write_text_lf(ARTIFACTS / "summary.md", "\n".join(summary))
    experiment_config["status"] = "COMPLETE"
    common.write_json_lf(experiment_config_path, experiment_config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    registered = next(
        item for item in registry["experiments"] if item["id"] == "e06_boundary_loss"
    )
    registered["status"] = "COMPLETE"
    common.write_json_lf(registry_path, registry)
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
