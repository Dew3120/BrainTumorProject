"""Quantify failure modes for held-out Phase 4 validation cases below Dice 0.50."""

import argparse
import csv
import json
from pathlib import Path

import mlflow
import numpy as np
import phase4_overfit as overfit
import phase4_segmentation_smoke as smoke
import phase4_train as train
import phase4_visual_audit as visual
import torch
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
LABELS = train.LABELS
STRUCTURE_8_CONNECTED = np.ones((3, 3), dtype=np.uint8)


def component_statistics(prediction, reference):
    prediction = np.asarray(prediction, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    predicted_pixels = int(prediction.sum())
    reference_pixels = int(reference.sum())
    if reference_pixels == 0:
        raise ValueError("Positive-case reference mask is empty")
    overlap_pixels = int((prediction & reference).sum())
    labeled, component_count = ndimage.label(
        prediction, structure=STRUCTURE_8_CONNECTED
    )
    if component_count:
        sizes = np.bincount(labeled.ravel())[1:]
        largest_component_fraction = float(sizes.max() / predicted_pixels)
        nonoverlap_components = sum(
            not np.any(reference[labeled == component])
            for component in range(1, component_count + 1)
        )
    else:
        largest_component_fraction = 0.0
        nonoverlap_components = 0
    ratio = predicted_pixels / reference_pixels
    overlap_exists = overlap_pixels > 0
    empty_prediction = predicted_pixels == 0
    nonempty_zero_overlap = predicted_pixels > 0 and not overlap_exists
    severe_under = overlap_exists and ratio < 0.5
    severe_over = overlap_exists and ratio > 2.0
    fragmented = component_count > 1
    disconnected_false_positive = component_count > 1 and nonoverlap_components > 0
    if empty_prediction:
        primary = "empty_prediction"
    elif nonempty_zero_overlap:
        primary = "nonempty_zero_overlap"
    elif severe_under:
        primary = "severe_under_prediction"
    elif severe_over:
        primary = "severe_over_prediction"
    elif fragmented:
        primary = "fragmented_prediction"
    else:
        primary = "other_partial_overlap"
    return {
        "reference_foreground_pixels": reference_pixels,
        "prediction_foreground_pixels": predicted_pixels,
        "overlap_pixels": overlap_pixels,
        "prediction_reference_ratio": ratio,
        "overlap_exists": overlap_exists,
        "empty_prediction": empty_prediction,
        "nonempty_zero_overlap": nonempty_zero_overlap,
        "severe_under_prediction": severe_under,
        "severe_over_prediction": severe_over,
        "fragmented_prediction": fragmented,
        "prediction_connected_components": int(component_count),
        "nonoverlap_component_count": int(nonoverlap_components),
        "disconnected_false_positive": disconnected_false_positive,
        "largest_component_fraction": largest_component_fraction,
        "reference_touches_model_edge": bool(
            reference[0].any()
            or reference[-1].any()
            or reference[:, 0].any()
            or reference[:, -1].any()
        ),
        "prediction_touches_model_edge": bool(
            prediction[0].any()
            or prediction[-1].any()
            or prediction[:, 0].any()
            or prediction[:, -1].any()
        ),
        "primary_failure_mode": primary,
    }


def summarize(rows):
    groups = [("all", rows)] + [
        (label, [row for row in rows if row["category"] == label]) for label in LABELS
    ]
    flags = (
        "empty_prediction",
        "nonempty_zero_overlap",
        "severe_under_prediction",
        "severe_over_prediction",
        "fragmented_prediction",
        "disconnected_false_positive",
        "reference_touches_model_edge",
        "prediction_touches_model_edge",
    )
    result = {}
    for name, members in groups:
        count = len(members)
        if not count:
            raise ValueError(f"No failures for {name}")
        values = {"count": count}
        for flag in flags:
            flag_count = sum(bool(row[flag]) for row in members)
            values[flag] = {"count": flag_count, "fraction": flag_count / count}
        values["complete_miss"] = {
            "count": values["empty_prediction"]["count"]
            + values["nonempty_zero_overlap"]["count"],
            "fraction": (
                values["empty_prediction"]["count"]
                + values["nonempty_zero_overlap"]["count"]
            )
            / count,
        }
        primary = {}
        for row in members:
            primary[row["primary_failure_mode"]] = (
                primary.get(row["primary_failure_mode"], 0) + 1
            )
        values["primary_failure_modes"] = {
            key: {"count": value, "fraction": value / count}
            for key, value in sorted(primary.items())
        }
        result[name] = values
    return result


def percentage(value):
    return f"{100 * value:.1f}%"


def summary_table(summary):
    lines = [
        "| Group | n | Empty | Non-empty zero overlap | Complete miss | Severe under | Severe over | Fragmented | Disconnected false positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", *LABELS):
        values = summary[group]
        cells = [group, str(values["count"])]
        for flag in (
            "empty_prediction",
            "nonempty_zero_overlap",
            "complete_miss",
            "severe_under_prediction",
            "severe_over_prediction",
            "fragmented_prediction",
            "disconnected_false_positive",
        ):
            item = values[flag]
            cells.append(f"{item['count']} ({percentage(item['fraction'])})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def primary_mode_table(summary):
    modes = (
        "empty_prediction",
        "nonempty_zero_overlap",
        "severe_under_prediction",
        "severe_over_prediction",
        "fragmented_prediction",
        "other_partial_overlap",
    )
    lines = [
        "| Group | Empty | Non-empty zero overlap | Severe under | Severe over | Fragmented | Other partial overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group in ("all", *LABELS):
        values = summary[group]
        cells = [group]
        for mode in modes:
            item = values["primary_failure_modes"].get(
                mode, {"count": 0, "fraction": 0.0}
            )
            cells.append(f"{item['count']} ({percentage(item['fraction'])})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_markdown(path, summary, verification):
    glioma_miss = summary["glioma"]["complete_miss"]["fraction"]
    meningioma_over = summary["meningioma"]["severe_over_prediction"]["fraction"]
    pituitary_over = summary["pituitary"]["severe_over_prediction"]["fraction"]
    if glioma_miss >= 0.70:
        glioma_conclusion = (
            "Glioma failures have a dominant complete-miss pattern: "
            f"{percentage(glioma_miss)} are empty predictions or non-empty predictions "
            "with zero overlap. This confirms the visual-review hypothesis."
        )
    else:
        glioma_conclusion = (
            "Glioma failures are heterogeneous: only "
            f"{percentage(glioma_miss)} are complete misses. The visual panel overstated "
            "a single dominant failure mode."
        )
    other_conclusion = (
        "Severe over-prediction accounts for "
        f"{percentage(meningioma_over)} of meningioma failures and "
        f"{percentage(pituitary_over)} of pituitary failures."
    )
    edge_fraction = summary["all"]["reference_touches_model_edge"]["fraction"]
    if edge_fraction >= 0.50:
        technical = (
            "Reference masks touch the model-space edge in "
            f"{percentage(edge_fraction)} of failures. This repeated edge pattern requires "
            "technical investigation before attributing failures only to task difficulty."
        )
    else:
        technical = (
            "Only "
            f"{percentage(edge_fraction)} of failed references touch a model-space edge. "
            "Together with the passed Section A alignment review, the failure distribution "
            "does not indicate a common flip, shift, pairing, or edge-clipping defect."
        )
    fragmented = summary["all"]["fragmented_prediction"]["fraction"]
    disconnected = summary["all"]["disconnected_false_positive"]["fraction"]
    content = f"""# Phase 4 failure-mode analysis

This analysis covers the 83 held-out BRISC validation cases with hard-mask Dice below 0.50 at the unchanged probability threshold of 0.50. The epoch-56 checkpoint was reloaded, all 680 validation predictions were recomputed exactly, and no test image was accessed.

Foreground pixel counts, overlap, area ratios, edge checks, and connected components are computed on the {verification["model_size"]}x{verification["model_size"]} hard masks used by the model. `original_width` and `original_height` preserve each source image's dimensions.

## Verification

- Checkpoint SHA256: `{verification["checkpoint_sha256"]}`
- Split snapshot SHA256: `{verification["split_sha256"]}`
- Recomputed validation cases: {verification["validation_images_recomputed"]}
- Maximum stored-versus-recomputed metric difference: {verification["maximum_metric_difference"]}
- Threshold: {verification["threshold"]:.2f}
- Test images accessed: {verification["test_images_accessed"]}

## Definitions

- **Empty prediction:** predicted foreground contains zero pixels.
- **Non-empty zero overlap:** predicted foreground exists but has no intersection with the reference.
- **Complete miss:** empty prediction or non-empty zero overlap.
- **Severe under-prediction:** overlap exists and prediction/reference area ratio is below 0.5.
- **Severe over-prediction:** overlap exists and prediction/reference area ratio is above 2.0.
- **Fragmented:** prediction has more than one 8-connected component.
- **Disconnected false positive:** prediction has multiple components and at least one component has no reference overlap.

Except for the mutually exclusive empty and non-empty-zero-overlap flags, diagnostic flags may overlap and therefore do not sum to 100%.

## Breakdown

{summary_table(summary)}

The flags above describe independent properties and can overlap. The following table assigns exactly one primary mode to every case, using this precedence: empty, non-empty zero overlap, severe under-prediction, severe over-prediction, fragmented prediction, then other partial overlap.

{primary_mode_table(summary)}

## Tumor-type hypothesis

{glioma_conclusion}

{other_conclusion}

## Technical cause versus task difficulty

{technical}

The dominant cross-category topology signal is fragmentation: {percentage(fragmented)} of all weak cases contain multiple predicted components, and {percentage(disconnected)} contain at least one predicted component with no reference overlap. This is a model-output failure pattern, not evidence of a shared image/mask mapping defect.

These findings quantify model behavior on BRISC's internal validation partition. They do not establish clinical validity, cross-dataset generalization, or verified patient-independent performance.
"""
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if args.threshold != 0.5:
        raise ValueError("Phase 4 failure analysis is fixed at threshold 0.5")
    run = args.run.resolve()
    checkpoint = args.checkpoint.resolve()
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    model_size = int(config["image_size"])
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    findings = json.loads(
        (ROOT / "audit/phase4_full_training_findings.json").read_text(encoding="utf-8")
    )
    accepted = findings["top3_checkpoints"][0]
    if (
        checkpoint.name != accepted["file"]
        or visual.sha256(checkpoint) != accepted["sha256"]
    ):
        raise ValueError("Accepted checkpoint identity or SHA256 mismatch")
    split = run / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Split snapshot SHA256 mismatch")
    parts = train.partition_masked_rows(smoke.read_csv(split))
    validation_rows = parts["validation"]
    if any(row["draft_split"] != "validation" for row in validation_rows):
        raise ValueError("Non-validation row entered validation analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state"]
    )
    metrics, records, predictions = visual.inference(
        model, validation_rows, device, args.threshold
    )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_delta = max(
        abs(metrics[key] - result["best_validation_metrics"][key])
        for key in result["best_validation_metrics"]
    )
    if maximum_delta > tolerance:
        raise ValueError(f"Validation metric mismatch: {maximum_delta}")
    failures = [row for row in records if row["dice"] < 0.5]
    expected_counts = {"glioma": 60, "meningioma": 8, "pituitary": 15}
    actual_counts = {
        label: sum(row["label"] == label for row in failures) for label in LABELS
    }
    if len(failures) != 83 or actual_counts != expected_counts:
        raise ValueError(
            f"Failure-set membership changed: total={len(failures)}, {actual_counts}"
        )
    row_lookup = {row["path"]: row for row in validation_rows}
    analyzed = []
    for record in sorted(
        failures,
        key=lambda row: (LABELS.index(row["label"]), row["dice"], row["path"]),
    ):
        source = row_lookup[record["path"]]
        _, reference_tensor = overfit.SizedPairs([source], model_size)[0]
        reference = reference_tensor[0].numpy() >= 0.5
        prediction = predictions[record["path"]]
        component = component_statistics(prediction, reference)
        if (
            component["reference_foreground_pixels"] != record["reference_pixels"]
            or component["prediction_foreground_pixels"] != record["predicted_pixels"]
        ):
            raise ValueError(f"Foreground count mismatch: {record['path']}")
        analyzed.append(
            {
                "path": record["path"],
                "mask_path": source["mask_path"],
                "category": record["label"],
                "original_split": source["original_split"],
                "draft_split": source["draft_split"],
                "original_width": int(source["width"]),
                "original_height": int(source["height"]),
                "model_width": model_size,
                "model_height": model_size,
                **component,
                "dice": record["dice"],
                "iou": record["iou"],
            }
        )
    csv_path = run / "phase4_failure_modes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(analyzed[0]))
        writer.writeheader()
        writer.writerows(analyzed)
    summary = summarize(analyzed)
    markdown_path = ROOT / "audit/phase4_failure_modes.md"
    verification = {
        "checkpoint_sha256": accepted["sha256"],
        "split_sha256": config["split_sha256"],
        "validation_images_recomputed": len(records),
        "maximum_metric_difference": maximum_delta,
        "threshold": args.threshold,
        "model_size": model_size,
        "test_images_accessed": 0,
    }
    write_markdown(markdown_path, summary, verification)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(markdown_path))
    print(
        json.dumps(
            {
                "status": "FAILURE_MODE_ANALYSIS_COMPLETE",
                "device": str(device),
                "validation_images_recomputed": len(records),
                "maximum_metric_difference": maximum_delta,
                "failures_analyzed": len(analyzed),
                "failure_counts_by_type": actual_counts,
                "test_images_accessed": 0,
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
