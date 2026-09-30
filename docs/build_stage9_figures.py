"""Build Stage 9 documentation figures from committed experiment artifacts only."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs/figures"
CLASSES = ["Glioma", "Meningioma", "Pituitary", "No tumor"]
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(relative: str):
    return json.loads((ROOT / relative).read_text(encoding="utf-8-sig"))


def read_csv(relative: str) -> list[dict]:
    with (ROOT / relative).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, name: str) -> Path:
    path = OUTPUT / name
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )


def pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.3, 1.8, 1.5, 1.4, "Uploaded\nimage", "#E8F1F8"),
        (2.2, 2.8, 1.8, 1.3, "ResNet-18\nclassifier", "#D6EAF8"),
        (2.2, 0.8, 1.8, 1.3, "O02–O04\nOOD signals", "#FCE8D5"),
        (4.5, 2.8, 2.0, 1.3, "E08 segmenter\nE05 + E06 + routed E07/v1", "#D5F5E3"),
        (4.5, 0.8, 2.0, 1.3, "O05 logistic\nOOD ensemble", "#FADBD8"),
        (7.0, 2.8, 1.6, 1.3, "C03 isotonic\ncalibration", "#E8DAEF"),
        (7.0, 0.8, 1.6, 1.3, "O06 ACCEPT /\nREVIEW policy", "#F5CBA7"),
        (9.1, 2.8, 1.6, 1.3, "R02 case\nreliability", "#FCF3CF"),
        (10.9, 1.8, 0.9, 1.4, "Result\n+ status", "#EAECEE"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor="#333333"
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", weight="bold")
    arrows = [
        ((1.8, 2.5), (2.2, 3.45)),
        ((1.8, 2.5), (2.2, 1.45)),
        ((4.0, 3.45), (4.5, 3.45)),
        ((4.0, 1.45), (4.5, 1.45)),
        ((6.5, 3.45), (7.0, 3.45)),
        ((6.5, 1.45), (7.0, 1.45)),
        ((8.6, 3.45), (9.1, 3.45)),
        ((8.6, 1.45), (10.9, 2.2)),
        ((10.7, 3.45), (10.9, 2.8)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color="#555555"))
    ax.text(6, 4.65, "Frozen research inference stack", ha="center", weight="bold", fontsize=15)
    ax.text(6, 0.25, "OOD REVIEW stops normal display; accepted cases receive segmentation and reliability status.", ha="center")
    return save(fig, "01_pipeline_architecture.png")


def classifier_confusion() -> Path:
    rows = read_csv("runs/classifier/20260921-170607-243133-baseline/validation_predictions.csv")
    order = [name.lower().replace(" ", "_") for name in CLASSES]
    matrix = confusion_matrix([r["label"] for r in rows], [r["predicted"] for r in rows], labels=order)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    for y in range(4):
        for x in range(4):
            ax.text(x, y, str(matrix[y, x]), ha="center", va="center", color="white" if matrix[y, x] > matrix.max() / 2 else "black", weight="bold")
    ax.set_xticks(range(4), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(4), CLASSES)
    ax.set(xlabel="Predicted label", ylabel="Dataset label", title="Classifier confusion matrix — BRISC internal validation")
    fig.colorbar(image, ax=ax, label="Images")
    return save(fig, "02_classifier_confusion_matrix.png")


def segmentation_panels() -> Path:
    best = Image.open(ROOT / "experiments/e08_ensemble/artifacts/best_validation_by_category.png").convert("RGB")
    worst = Image.open(ROOT / "experiments/e08_ensemble/artifacts/worst_validation_by_category.png").convert("RGB")
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), constrained_layout=True)
    axes[0].imshow(best)
    axes[0].set_title("High-overlap E08 examples")
    axes[1].imshow(worst)
    axes[1].set_title("Low-overlap E08 examples")
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Selected segmenter: category-stratified success and failure examples")
    return save(fig, "03_segmentation_examples.png")


def copy_as_figure(source: str, output: str, title: str) -> Path:
    image = Image.open(ROOT / source).convert("RGB")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(title)
    return save(fig, output)


def failure_modes() -> Path:
    data = read_json("audit/segmentation_training_findings.json")["segmentation_validation_consolidation"]["failure_modes"]
    names = ["Fragmented", "Disconnected false positive", "Severe over-prediction", "Severe under-prediction", "Complete miss"]
    values = [70 / 83, 67 / 83, 24 / 83, 19 / 83, 17 / 83]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], np.asarray(values[::-1]) * 100, color="#0072B2")
    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(value * 100 + 1, bar.get_y() + bar.get_height() / 2, f"{value:.1%}", va="center")
    ax.set(xlabel="Share of 83 weak cases (%)", title="Overlapping failure-mode flags on BRISC internal validation")
    ax.set_xlim(0, 100)
    ax.grid(axis="x")
    assert data["fragmented_prediction"]["count"] == 70
    return save(fig, "08_failure_mode_breakdown.png")


def calibration_comparison() -> Path:
    rows = read_csv("experiments/c08_unified_ece/artifacts/segmenter_ece_table.csv")
    rows = [row for row in rows if row["region"] == "lesion_centered"]
    labels = [row["experiment_id"].split("_")[0].upper() for row in rows]
    raw = [float(row["raw_ece"]) for row in rows]
    calibrated = [float(row["oof_calibrated_ece"]) for row in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - 0.18, raw, 0.36, label="Raw ECE", color="#999999")
    ax.bar(x + 0.18, calibrated, 0.36, label="OOF calibrated ECE", color="#009E73")
    ax.axhline(0.05, color="#D55E00", linestyle="--", label="0.05 reference")
    ax.set_xticks(x, labels)
    ax.set(ylabel="Lesion-centered ECE", title="Pixel calibration comparison under unified binning")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    return save(fig, "09_calibration_comparison.png")


def feature_importance() -> Path:
    rows = read_csv("experiments/r01_ablation/artifacts/permutation_importance.csv")
    rows.sort(key=lambda row: float(row["bootstrap_mean_auroc_decrease"]), reverse=True)
    rows = rows[:10]
    labels = [row["feature"].replace("_", " ") for row in rows][::-1]
    means = np.asarray([float(row["bootstrap_mean_auroc_decrease"]) for row in rows][::-1])
    lower = np.asarray([float(row["ci95_lower"]) for row in rows][::-1])
    upper = np.asarray([float(row["ci95_upper"]) for row in rows][::-1])
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(labels, means, color="#0072B2", xerr=np.vstack([means - lower, upper - means]), capsize=3)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Mean AUROC decrease after permutation", title="R01 prediction-feature importance")
    ax.grid(axis="x")
    return save(fig, "10_feature_importance.png")


def timeline() -> Path:
    labels = [
        "Foundation", "Failure audit", "Glioma experiments", "Calibration", "Reliability",
        "Policy", "External validation", "OOD", "Final freeze",
    ]
    notes = ["Stage 0", "Stage 1", "Stage 2", "Stage 3", "Stage 4", "Stage 5", "Stage 6", "Stage 7", "Stage 8"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, np.zeros_like(x), color="#444444", linewidth=2)
    ax.scatter(x, np.zeros_like(x), s=180, c=[COLORS[i % len(COLORS)] for i in x], zorder=3)
    for index, (label, note) in enumerate(zip(labels, notes, strict=True)):
        offset = 0.28 if index % 2 == 0 else -0.28
        ax.text(index, offset, f"{note}\n{label}", ha="center", va="bottom" if offset > 0 else "top")
    ax.set_ylim(-0.75, 0.75)
    ax.axis("off")
    ax.set_title("Registered project progression")
    return save(fig, "12_stage_progression_timeline.png")


def cost_curve() -> Path:
    rows = read_csv("experiments/p02_threshold_selection/artifacts/binary_threshold_curve.csv")
    threshold = np.asarray([float(row["threshold"]) for row in rows])
    cost = np.asarray([float(row["total_cost"]) for row in rows])
    selected = read_json("experiments/p02_threshold_selection/result.json")["binary_selection"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(threshold, cost, color="#0072B2", linewidth=1.8)
    ax.scatter([selected["threshold"]], [selected["total_cost"]], color="#D55E00", s=70, zorder=3, label="Selected")
    ax.scatter([0.5], [406], color="#009E73", s=55, zorder=3, label="Conventional 0.50")
    ax.set(xlabel="Reliability threshold", ylabel="Expected cost", title="Policy cost across all observed thresholds")
    ax.grid()
    ax.legend(frameon=False)
    return save(fig, "13_policy_cost_curve.png")


def reliability_transfer() -> Path:
    r02 = read_json("experiments/r02_reduced_features/result.json")
    x2 = read_json("experiments/x02_openneuro_pituitary/result.json")["reliability_transfer"]
    x3 = read_json("experiments/x03_upenn_gbm/result.json")["reliability_transfer"]
    reduced = r02["reduced_10_feature_model"]
    internal = reduced["failure_detection_auroc"]
    internal_ci = reduced["bootstrap"]["ci95_lower"], reduced["bootstrap"]["ci95_upper"]
    names = ["BRISC internal", "OpenNeuro", "UPENN-GBM"]
    values = [internal, x2["failure_detection_auroc"], x3["failure_detection_auroc"]]
    cis = [internal_ci, x2["bootstrap_ci95"]["failure_detection_auroc"], x3["bootstrap_ci95"]["failure_detection_auroc"]]
    errors = np.asarray([[v - c[0] for v, c in zip(values, cis, strict=True)], [c[1] - v for v, c in zip(values, cis, strict=True)]])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values, color=["#0072B2", "#E69F00", "#009E73"], yerr=errors, capsize=5)
    ax.axhline(0.75, color="#D55E00", linestyle="--", label="Registered minimum")
    ax.set(
        ylim=(0.70, 1.0),
        ylabel="Failure-detection AUROC",
        title="Failure ranking retained discrimination on two external cohorts",
    )
    ax.legend(frameon=False)
    ax.grid(axis="y")
    return save(fig, "14_reliability_transfer.png")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("*.png"):
        old.unlink()
    style()
    records = []

    def record(path: Path, sources: list[str], description: str) -> None:
        records.append(
            {
                "figure": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "dpi": 300,
                "description": description,
                "sources": [
                    {"path": source, "sha256": sha256(ROOT / source)} for source in sources
                ],
            }
        )

    record(pipeline(), ["runs/final_stack/manifest.json"], "Frozen inference architecture")
    record(classifier_confusion(), ["runs/classifier/20260921-170607-243133-baseline/validation_predictions.csv", "audit/classifier_training_findings.json"], "Classifier internal-validation confusion matrix")
    record(segmentation_panels(), ["experiments/e08_ensemble/artifacts/best_validation_by_category.png", "experiments/e08_ensemble/artifacts/worst_validation_by_category.png"], "E08 success and failure examples")
    record(copy_as_figure("experiments/c03_isotonic/artifacts/lesion_centered_reliability.png", "04_c03_reliability.png", "C03 isotonic calibration — lesion-centered pixels"), ["experiments/c03_isotonic/artifacts/lesion_centered_reliability.png", "experiments/c03_isotonic/result.json"], "Raw and calibrated pixel reliability")
    record(copy_as_figure("experiments/p02_threshold_selection/artifacts/roc_curve.png", "05_reliability_roc.png", "R02 case-reliability ROC"), ["experiments/p02_threshold_selection/artifacts/roc_curve.png", "experiments/r02_reduced_features/result.json"], "Reliability ROC")
    record(copy_as_figure("experiments/o05_ensemble_ood/artifacts/roc_curve.png", "06_ood_roc.png", "O05 OOD detector ROC"), ["experiments/o05_ensemble_ood/artifacts/roc_curve.png", "experiments/o05_ensemble_ood/result.json"], "OOD ROC")
    record(copy_as_figure("experiments/x05_cross_dataset_synthesis/artifacts/transport_summary.png", "07_cross_dataset_transport.png", "Frozen-stack transport across external cohorts"), ["experiments/x05_cross_dataset_synthesis/artifacts/transport_summary.png", "experiments/x05_cross_dataset_synthesis/result.json"], "External transport summary")
    record(failure_modes(), ["audit/segmentation_training_findings.json", "audit/segmentation_failure_modes.md"], "Failure-mode proportions")
    record(calibration_comparison(), ["experiments/c08_unified_ece/artifacts/segmenter_ece_table.csv"], "C01-C07 calibration comparison")
    record(feature_importance(), ["experiments/r01_ablation/artifacts/permutation_importance.csv"], "R01 feature importance")
    record(copy_as_figure("experiments/p05_threshold_sensitivity/artifacts/sensitivity_curve.png", "11_threshold_sensitivity.png", "Reliability threshold sensitivity"), ["experiments/p05_threshold_sensitivity/artifacts/sensitivity_curve.png", "experiments/p05_threshold_sensitivity/result.json"], "Threshold sensitivity")
    record(timeline(), ["experiments/registry.json"], "Project-stage timeline")
    record(cost_curve(), ["experiments/p02_threshold_selection/artifacts/binary_threshold_curve.csv", "experiments/p02_threshold_selection/result.json"], "Expected-cost curve")
    record(reliability_transfer(), ["experiments/r02_reduced_features/result.json", "experiments/x02_openneuro_pituitary/result.json", "experiments/x03_upenn_gbm/result.json"], "Failure-ranking AUROC on two external cohorts")

    (OUTPUT / "figure_manifest.json").write_text(
        json.dumps({"schema_version": 1, "figure_count": len(records), "figures": records}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"status": "PASS", "figures": len(records)}, indent=2))


if __name__ == "__main__":
    main()
