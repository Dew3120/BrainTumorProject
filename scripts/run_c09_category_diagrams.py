"""Render the six registered per-category calibration diagrams."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from run_c03_isotonic import apply_isotonic

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c09_category_diagrams"
ARTIFACTS = EXPERIMENT / "artifacts"
FIGURES = ROOT / "audit/figures"
INDEX_JSON = ROOT / "audit/category_reliability_diagrams.json"
INDEX_MD = ROOT / "audit/category_reliability_diagrams.md"


def render_pixel(path, label, raw, calibrated):
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    for summary, name, marker in (
        (raw, "Raw", "o"),
        (calibrated, "C03 OOF isotonic", "s"),
    ):
        axis.plot(
            [item["mean_probability"] for item in summary["bins"]],
            [item["observed_frequency"] for item in summary["bins"]],
            marker=marker,
            label=f"{name} (ECE={summary['pixel_ece']:.4f})",
        )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean foreground probability",
        ylabel="Observed foreground frequency",
        title=f"{label.title()} lesion-centred pixel calibration",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_case(path, label, ece, bins):
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], "--", color="black", label="Ideal")
    axis.plot(
        [item["mean_probability"] for item in bins],
        [item["observed_frequency"] for item in bins],
        marker="o",
        color="#4c72b0",
        label=f"OOF reliability (ECE={ece:.4f})",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Predicted P(Dice >= 0.50)",
        ylabel="Observed fraction with Dice >= 0.50",
        title=f"{label.title()} case-level segmentation reliability",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C09 is not in its registered pre-run state")
    c03_result_path = ROOT / "experiments/c03_isotonic/result.json"
    mappings_path = ROOT / config["c03_mappings_path"]
    reliability_path = ROOT / config["case_reliability_oof_path"]
    if common.sha256(c03_result_path) != config["c03_result_sha256"]:
        raise ValueError("C03 result hash mismatch")
    if common.sha256(mappings_path) != config["c03_mappings_sha256"]:
        raise ValueError("C03 mapping hash mismatch")
    if common.sha256(reliability_path) != config["case_reliability_oof_sha256"]:
        raise ValueError("Reliability OOF hash mismatch")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    mappings = common.read_json(mappings_path)["fold_mappings"]
    if [int(item["fold"]) for item in mappings] != [1, 2, 3, 4, 5]:
        raise ValueError("C03 mapping folds are incomplete")
    oof_probability = np.empty_like(raw_probability)
    for fold, mapping in enumerate(mappings):
        indices = np.flatnonzero(assignment == fold)
        _logits, probability = apply_isotonic(raw_probability[indices], mapping)
        oof_probability[indices] = probability
    raw_clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    calibrated_clipped = np.clip(oof_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(raw_clipped / (1 - raw_clipped))
    calibrated_logits = np.log(calibrated_clipped / (1 - calibrated_clipped))
    regions = common.lesion_regions(targets)
    label_names = np.asarray([row["label"] for row in rows])
    reliability_source = {row["path"]: row for row in common.read_csv(reliability_path)}
    if len(reliability_source) != 680:
        raise ValueError("Reliability source does not contain 680 unique paths")
    FIGURES.mkdir(parents=True, exist_ok=True)
    category_results = {}
    output_paths = []
    for label in config["categories"]:
        indices = np.flatnonzero(label_names == label)
        raw, _, _ = common.region_metrics(
            raw_probability,
            raw_logits,
            targets,
            regions,
            indices,
            "lesion_centered",
        )
        calibrated, _, _ = common.region_metrics(
            oof_probability,
            calibrated_logits,
            targets,
            regions,
            indices,
            "lesion_centered",
        )
        pixel_path = FIGURES / f"pixel_reliability_{label}.png"
        render_pixel(pixel_path, label, raw, calibrated)
        reliability = []
        success = []
        for index in indices:
            source = reliability_source.get(rows[index]["path"])
            if source is None or source["label"] != label:
                raise ValueError(f"Reliability category mismatch: {rows[index]['path']}")
            reliability.append(float(source["oof_reliability_score"]))
            success.append(int(source["success_dice_at_least_0_50"]))
        reliability = np.asarray(reliability)
        success = np.asarray(success)
        case_ece, case_mce, case_bins = common.equal_frequency_binary(
            reliability, success, bins=15
        )
        case_path = FIGURES / f"case_reliability_{label}.png"
        render_case(case_path, label, case_ece, case_bins)
        category_results[label] = {
            "cases": len(indices),
            "pixel_lesion_centered": {
                "raw_ece": raw["pixel_ece"],
                "c03_oof_ece": calibrated["pixel_ece"],
                "raw_mce": raw["pixel_mce"],
                "c03_oof_mce": calibrated["pixel_mce"],
                "evaluated_pixels": raw["pixel_count"],
            },
            "case_reliability": {
                "ece": case_ece,
                "mce": case_mce,
                "success_cases": int(success.sum()),
                "failure_cases": int((success == 0).sum()),
                "bins": case_bins,
            },
            "figures": {
                "pixel": pixel_path.relative_to(ROOT).as_posix(),
                "case": case_path.relative_to(ROOT).as_posix(),
            },
        }
        output_paths.extend([pixel_path, case_path])
    if len(output_paths) != 6 or any(
        not path.is_file() or path.stat().st_size < 10_000 for path in output_paths
    ):
        raise ValueError("C09 did not produce six non-empty diagrams")
    index = {
        "schema_version": 1,
        "status": "VERIFIED",
        "selected_pixel_calibrator": "c03_isotonic",
        "selection_reason": config["selection_reason"],
        "binning": "15 equal-frequency bins",
        "pixel_region": config["pixel_region"],
        "categories": category_results,
        "verification": {
            "c03_result_sha256": common.sha256(c03_result_path),
            "c03_mappings_sha256": common.sha256(mappings_path),
            "case_reliability_oof_sha256": common.sha256(reliability_path),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "figures_created": 6,
            "all_pixel_outputs_out_of_fold": True,
            "all_case_outputs_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "calibrator_refitted": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Categories are true labels used only to stratify descriptive diagrams, not route calibration.",
            "Lesion-centered pixel diagrams use reference masks for evaluation-region definition only.",
            "All results use internal validation data; patient independence and external generalization are unverified.",
        ],
    }
    common.write_json(INDEX_JSON, index)
    lines = [
        "# Per-category calibration diagrams",
        "",
        "C03 isotonic calibration was selected by the fixed operational ranking. The pixel curves are lesion-centred and strictly out-of-fold. The case curves use the frozen accepted v1 reliability model's out-of-fold scores.",
        "",
        "| Category | Cases | Raw pixel ECE | C03 pixel ECE | Case reliability ECE |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, item in category_results.items():
        lines.append(
            f"| {label} | {item['cases']} | {item['pixel_lesion_centered']['raw_ece']:.6f} | {item['pixel_lesion_centered']['c03_oof_ece']:.6f} | {item['case_reliability']['ece']:.6f} |"
        )
    lines.extend(
        [
            "",
            "These are internal validation diagnostics, not clinical performance claims. True labels stratify the figures only and never route the calibrator.",
        ]
    )
    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "schema_version": 1,
        "experiment_id": "c09_category_diagrams",
        "status": "PASS",
        "decision": "VERIFIED",
        "selected_pixel_calibrator": "c03_isotonic",
        "figures_created": [path.relative_to(ROOT).as_posix() for path in output_paths],
        "category_metrics": category_results,
        "verification": index["verification"],
    }
    common.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(
        item
        for item in registry["experiments"]
        if item["id"] == "c09_category_diagrams"
    )["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    (EXPERIMENT / "decision.md").write_text(
        """# Per-category reliability diagrams decision

**Decision:** VERIFIED

All registered source hashes matched. Six non-empty diagrams were rendered with 15 equal-frequency bins: lesion-centred C03 pixel calibration and frozen case-level reliability for glioma, meningioma, and pituitary. All underlying scores are out-of-fold.

C09 is descriptive and changes no earlier calibration decision. No calibrator or neural model was fitted, no threshold changed, and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    common.write_json(
        ARTIFACTS / "figure_index.json",
        {
            "figures": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": common.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in output_paths
            ]
        },
    )
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    included.extend([INDEX_JSON, INDEX_MD, *output_paths])
    common.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": "PASS",
            "experiment_id": "c09_category_diagrams",
            "registered_decision": "VERIFIED",
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": common.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(set(included))
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "selected_pixel_calibrator": "c03_isotonic",
                "figures_created": 6,
                "category_metrics": {
                    label: {
                        "cases": item["cases"],
                        "raw_pixel_ece": item["pixel_lesion_centered"]["raw_ece"],
                        "c03_pixel_ece": item["pixel_lesion_centered"][
                            "c03_oof_ece"
                        ],
                        "case_reliability_ece": item["case_reliability"]["ece"],
                    }
                    for label, item in category_results.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
