"""Extract Phase 4 uncertainty signals without retraining or test-set access."""

import argparse
import csv
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import phase3_classifier as classifier
import phase4_classifier_segmenter_agreement as agreement
import phase4_overfit as overfit
import phase4_segmentation_smoke as smoke
import phase4_train as segmenter
import phase4_visual_audit as visual
import torch
from monai.losses import DiceLoss
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
THRESHOLD = 0.5
CLASSIFIER_RUN_NAME = agreement.CLASSIFIER_RUN_NAME
SEGMENTER_RUN_NAME = agreement.SEGMENTER_RUN_NAME
CLASSIFIER_CHECKPOINT_NAME = agreement.CLASSIFIER_CHECKPOINT_NAME
SEGMENTER_CHECKPOINT_NAME = agreement.SEGMENTER_CHECKPOINT_NAME
PROBABILITY_FIELDS = (
    "segmenter_mean_probability_inside_reference",
    "segmenter_max_probability_outside_reference",
    "segmenter_mean_probability_whole_image",
    "segmenter_max_probability_inside_reference",
)


def probability_statistics(probability, reference):
    probability = np.asarray(probability, dtype=np.float64)
    reference = np.asarray(reference, dtype=bool)
    if probability.shape != reference.shape:
        raise ValueError("Probability/reference shape mismatch")
    if not reference.any() or reference.all():
        raise ValueError("Reference must contain foreground and background")
    inside = probability[reference]
    outside = probability[~reference]
    return {
        "segmenter_mean_probability_inside_reference": float(inside.mean()),
        "segmenter_max_probability_outside_reference": float(outside.max()),
        "segmenter_mean_probability_whole_image": float(probability.mean()),
        "segmenter_max_probability_inside_reference": float(inside.max()),
    }


def segmenter_inference(model, rows, device):
    dataset = segmenter.FullPairs(rows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    dice_loss = DiceLoss(sigmoid=True)
    model.eval()
    records = []
    predictions = {}
    dice_values = []
    iou_values = []
    labels = []
    total_loss = 0.0
    with torch.inference_mode():
        for images, targets, label_indices, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(images)
            loss = segmenter.segmentation_loss(logits, targets, dice_loss)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise RuntimeError("Non-finite segmenter validation output")
            probabilities = logits.float().sigmoid()
            predicted = probabilities >= THRESHOLD
            reference = targets >= 0.5
            intersection = (predicted & reference).sum(dim=(1, 2, 3)).float()
            predicted_area = predicted.sum(dim=(1, 2, 3)).float()
            reference_area = reference.sum(dim=(1, 2, 3)).float()
            dice = 2 * intersection / (predicted_area + reference_area).clamp_min(1)
            iou = intersection / (
                predicted_area + reference_area - intersection
            ).clamp_min(1)
            total_loss += loss.item() * len(images)
            dice_values.extend(dice.cpu().tolist())
            iou_values.extend(iou.cpu().tolist())
            labels.extend(label_indices.tolist())
            for position, dataset_index in enumerate(indices.tolist()):
                row = rows[dataset_index]
                probability = probabilities[position, 0].cpu().numpy()
                reference_mask = reference[position, 0].cpu().numpy()
                hard_prediction = predicted[position, 0].cpu().numpy()
                predictions[row["path"]] = hard_prediction
                records.append(
                    {
                        "path": row["path"],
                        "filename": row["filename"],
                        "category": row["label"],
                        "label": row["label"],
                        "dice": float(dice[position].item()),
                        "iou": float(iou[position].item()),
                        "reference_pixels": int(reference_area[position].item()),
                        "predicted_pixels": int(predicted_area[position].item()),
                        **probability_statistics(probability, reference_mask),
                    }
                )
    metrics = segmenter.summarize_scores(
        np.asarray(dice_values), np.asarray(iou_values), np.asarray(labels)
    )
    metrics["val_loss"] = total_loss / len(dataset)
    return metrics, records, predictions


def select_groups(records):
    zeros = sorted(
        [row for row in records if row["dice"] == 0.0],
        key=lambda row: agreement.normalized_path(row["path"]),
    )
    best = sorted(
        records,
        key=lambda row: (-row["dice"], agreement.normalized_path(row["path"])),
    )[:20]
    if len(zeros) != 17:
        raise ValueError(f"Expected 17 zero-Dice cases, found {len(zeros)}")
    if len(best) != 20:
        raise ValueError("Expected 20 highest-Dice cases")
    if {row["path"] for row in zeros} & {row["path"] for row in best}:
        raise ValueError("Zero-Dice and highest-Dice groups overlap")
    return zeros, best


def join_classifier_signals(segmentation_records, classifier_records, row_lookup):
    classification = {
        agreement.normalized_path(row["path"]): row for row in classifier_records
    }
    cases = []
    for segmentation in segmentation_records:
        key = agreement.normalized_path(segmentation["path"])
        if key not in classification:
            raise ValueError(f"Missing classifier signal: {segmentation['path']}")
        classified = classification[key]
        if classified["category"] != segmentation["category"]:
            raise ValueError(
                f"Classifier/segmenter label mismatch: {segmentation['path']}"
            )
        predicted_class = classified["predicted_class"]
        source = row_lookup[key]
        cases.append(
            {
                "path": segmentation["path"],
                "filename": segmentation["filename"],
                "category": segmentation["category"],
                "original_split": source["original_split"],
                "draft_split": source["draft_split"],
                "segmenter_dice": segmentation["dice"],
                "segmenter_iou": segmentation["iou"],
                "classifier_predicted_class": predicted_class,
                "classifier_correct": predicted_class == segmentation["category"],
                "classifier_predicted_class_confidence": classified[
                    f"score_{predicted_class}"
                ],
                "classifier_true_class_probability": classified[
                    f"score_{segmentation['category']}"
                ],
                **{field: segmentation[field] for field in PROBABILITY_FIELDS},
            }
        )
    return cases


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("Distribution requires finite values")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "q25": float(np.quantile(array, 0.25)),
        "q75": float(np.quantile(array, 0.75)),
    }


def summarize_group(cases):
    fields = (
        "classifier_predicted_class_confidence",
        "classifier_true_class_probability",
        *PROBABILITY_FIELDS,
    )
    return {
        "count": len(cases),
        "category_counts": dict(
            sorted(Counter(case["category"] for case in cases).items())
        ),
        "classifier_correct_count": sum(case["classifier_correct"] for case in cases),
        "distributions": {
            field: distribution([case[field] for case in cases]) for field in fields
        },
    }


def safe_spearman(left, right):
    statistic = float(spearmanr(left, right).statistic)
    if not np.isfinite(statistic):
        raise ValueError("Spearman correlation is not finite")
    return statistic


def cross_model_correlations(all_cases):
    classifier_confidence = [
        case["classifier_predicted_class_confidence"] for case in all_cases
    ]
    segmenter_dice = [case["segmenter_dice"] for case in all_cases]
    segmenter_inside = [
        case["segmenter_mean_probability_inside_reference"] for case in all_cases
    ]
    return {
        "population": len(all_cases),
        "spearman_classifier_confidence_vs_segmenter_dice": safe_spearman(
            classifier_confidence, segmenter_dice
        ),
        "spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference": safe_spearman(
            classifier_confidence, segmenter_inside
        ),
        "spearman_segmenter_mean_probability_inside_reference_vs_dice": safe_spearman(
            segmenter_inside, segmenter_dice
        ),
    }


def write_manual_selection(path, groups):
    rows = []
    for group, records in groups:
        for rank, record in enumerate(records, start=1):
            rows.append(
                {
                    "group": group,
                    "rank": rank,
                    "path": record["path"],
                    "category": record["category"],
                    "dice": record["dice"],
                    "iou": record["iou"],
                }
            )
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{value:.4f}"


def group_table(group_summaries):
    zero = group_summaries["zero_dice"]
    best = group_summaries["highest_dice_20"]
    rows = [
        "| Signal (median) | 17 zero-Dice | 20 highest-Dice |",
        "|---|---:|---:|",
    ]
    labels = {
        "classifier_predicted_class_confidence": "Classifier predicted-class confidence",
        "classifier_true_class_probability": "Classifier true-class probability",
        "segmenter_mean_probability_inside_reference": "Segmenter mean probability inside reference",
        "segmenter_max_probability_inside_reference": "Segmenter max probability inside reference",
        "segmenter_max_probability_outside_reference": "Segmenter max probability outside reference",
        "segmenter_mean_probability_whole_image": "Segmenter whole-image mean probability",
    }
    for field, label in labels.items():
        rows.append(
            f"| {label} | {fmt(zero['distributions'][field]['median'])} | "
            f"{fmt(best['distributions'][field]['median'])} |"
        )
    return "\n".join(rows)


def write_markdown(path, result):
    zero = result["group_summaries"]["zero_dice"]
    best = result["group_summaries"]["highest_dice_20"]
    zero_classifier = zero["distributions"]["classifier_predicted_class_confidence"][
        "median"
    ]
    best_classifier = best["distributions"]["classifier_predicted_class_confidence"][
        "median"
    ]
    confidence_difference = best_classifier - zero_classifier
    if abs(confidence_difference) < 0.05:
        classifier_comparison = "similar"
    elif confidence_difference > 0:
        classifier_comparison = "higher in the highest-Dice group"
    else:
        classifier_comparison = "higher in the zero-Dice group"
    zero_inside = zero["distributions"]["segmenter_mean_probability_inside_reference"][
        "median"
    ]
    best_inside = best["distributions"]["segmenter_mean_probability_inside_reference"][
        "median"
    ]
    zero_outside = zero["distributions"]["segmenter_max_probability_outside_reference"][
        "median"
    ]
    correlations = result["cross_model_correlations_all_680"]
    section_c = result["section_c_context"]
    content = f"""# Phase 4 uncertainty signals

This analysis uses the accepted classifier and segmenter on the held-out BRISC validation partition. It compares all 17 zero-Dice cases with the 20 highest-Dice cases. No model was retrained, thresholds remained at 0.50, and no test image was accessed.

## Verification

- Shared split SHA256: `{result["verification"]["shared_split_sha256"]}`
- Classifier checkpoint SHA256: `{result["verification"]["classifier_checkpoint_sha256"]}`
- Segmenter checkpoint SHA256: `{result["verification"]["segmenter_checkpoint_sha256"]}`
- Classifier maximum metric/probability difference: {result["verification"]["classifier_maximum_metric_difference"]} / {result["verification"]["classifier_maximum_probability_difference"]}
- Segmenter maximum metric/per-image difference: {result["verification"]["segmenter_maximum_metric_difference"]} / {result["verification"]["segmenter_maximum_per_image_difference"]}
- Test images accessed: {result["verification"]["test_images_accessed"]}

## Group comparison

{group_table(result["group_summaries"])}

Zero-Dice category counts: {zero["category_counts"]}. Highest-Dice category counts: {best["category_counts"]}.

The groups have very different tumor-type composition: zero-Dice cases are mostly glioma, whereas the highest-Dice cases are mostly meningioma. Their descriptive confidence differences therefore combine failure severity with tumor-type mix and must not be interpreted as a controlled calibration comparison.

## Findings

### 1. Low confidence or confident-wrong?

The zero-Dice segmentations are spatially wrong rather than empty. Their median mean probability inside the true reference is {fmt(zero_inside)}, while their median maximum probability outside the reference is {fmt(zero_outside)}. Every zero-Dice case has a non-empty hard prediction outside the reference. The classifier is correct in {zero["classifier_correct_count"]} of 17 cases; the remaining case predicts no tumor with 0.999985 confidence and assigns approximately 0.00000025 probability to the true meningioma class. The pixel-level pattern combines extreme under-confidence on the true lesion with extreme over-confidence elsewhere.

### 2. Confidence in highest-Dice versus zero-Dice cases

Median classifier predicted-class confidence is {fmt(zero_classifier)} for zero-Dice cases and {fmt(best_classifier)} for the highest-Dice cases; the distributions are {classifier_comparison}. In contrast, median segmenter probability inside the reference rises from {fmt(zero_inside)} to {fmt(best_inside)}, so the segmenter's spatial signal separates the groups much more clearly than classifier confidence. Maximum probability outside the reference is near 1.0 in both groups, showing that a single outside-mask maximum is saturated and is not a useful standalone failure detector.

### 3. Classifier/segmenter decoupling

Across all 680 paired validation cases, Spearman correlation between classifier predicted-class confidence and segmenter Dice is {correlations["spearman_classifier_confidence_vs_segmenter_dice"]:.4f}; correlation with segmenter mean probability inside the reference is {correlations["spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference"]:.4f}. These are weak-to-moderate and weak relationships, respectively. By comparison, segmenter mean inside-reference probability correlates with its own Dice at {correlations["spearman_segmenter_mean_probability_inside_reference_vs_dice"]:.4f}. Section C also found that {section_c["segmenter_failures_with_correct_classifier"]} of {section_c["segmenter_failure_count"]} segmentation failures retained the correct classifier type, and {section_c["zero_dice_with_correct_classifier_type"]} of {section_c["zero_dice_count"]} zero-Dice cases retained it. Classifier confidence cannot be used as a proxy for segmentation reliability.

### 4. Phase 5 calibration hand-off

The observed problem is miscalibration and over-confidence, not global under-confidence. Phase 5 should calibrate the classifier and segmenter separately. The classifier needs class-probability calibration and an abstention/disagreement rule because its no-tumor false negative is confidently wrong. The segmenter needs pixel-level reliability analysis inside and outside reference regions, calibration curves, and case-level uncertainty or abstention signals that expose misplaced foreground. Calibration can make confidence more honest; it cannot repair incorrect localization or replace retraining and external validation.

## Manual review still required

The review panel contains the single classifier-no-tumor/segmenter-non-empty case and four deterministically sampled zero-Dice gliomas. Confirm whether the meningioma reference mask is incomplete and whether the glioma cases share a visible pattern before accepting Section D.

- Panel: `{result["manual_review"]["panel"]}`
- Selection CSV: `{result["manual_review"]["selection_csv"]}`

## Scope

These findings describe selected extremes and correlations on BRISC's internal validation partition. They do not establish clinical validity, patient-independent generalization, or sealed-test performance.
"""
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-run",
        type=Path,
        default=ROOT / "runs/classifier" / CLASSIFIER_RUN_NAME,
    )
    parser.add_argument(
        "--segmenter-run",
        type=Path,
        default=ROOT / "runs/segmentation" / SEGMENTER_RUN_NAME,
    )
    parser.add_argument("--classifier-threshold", type=float, default=THRESHOLD)
    parser.add_argument("--segmenter-threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()
    if args.classifier_threshold != THRESHOLD or args.segmenter_threshold != THRESHOLD:
        raise ValueError(
            "Accepted classifier and segmenter thresholds are fixed at 0.5"
        )
    classifier_run = args.classifier_run.resolve()
    segmenter_run = args.segmenter_run.resolve()
    if (
        classifier_run.name != CLASSIFIER_RUN_NAME
        or segmenter_run.name != SEGMENTER_RUN_NAME
    ):
        raise ValueError("Run identity mismatch")

    classifier_findings = json.loads(
        (ROOT / "audit/phase3_findings.json").read_text(encoding="utf-8")
    )
    segmenter_findings = json.loads(
        (ROOT / "audit/phase4_full_training_findings.json").read_text(encoding="utf-8")
    )
    section_c_path = ROOT / "audit/phase4_classifier_segmenter_agreement.json"
    section_c = json.loads(section_c_path.read_text(encoding="utf-8"))
    expected_context = {
        "segmenter_failure_count": 83,
        "segmenter_failures_with_correct_classifier": 78,
        "zero_dice_count": 17,
        "zero_dice_with_correct_classifier_type": 16,
    }
    if any(
        section_c["observations"][key] != value
        for key, value in expected_context.items()
    ):
        raise ValueError("Section C context mismatch")

    classifier_checkpoint = classifier_run / CLASSIFIER_CHECKPOINT_NAME
    segmenter_checkpoint = (
        segmenter_run / "top3_checkpoints" / SEGMENTER_CHECKPOINT_NAME
    )
    classifier_checkpoint_hash = classifier.sha256(classifier_checkpoint)
    segmenter_checkpoint_hash = visual.sha256(segmenter_checkpoint)
    if classifier_checkpoint_hash != classifier_findings["checkpoint_sha256"]:
        raise ValueError("Classifier checkpoint SHA256 mismatch")
    accepted_segmenter = segmenter_findings["top3_checkpoints"][0]
    if (
        accepted_segmenter["file"] != segmenter_checkpoint.name
        or accepted_segmenter["sha256"] != segmenter_checkpoint_hash
    ):
        raise ValueError("Segmenter checkpoint identity or SHA256 mismatch")

    classifier_config = json.loads(
        (classifier_run / "config.json").read_text(encoding="utf-8")
    )
    segmenter_config = json.loads(
        (segmenter_run / "config.json").read_text(encoding="utf-8")
    )
    classifier_split = classifier_run / "split_snapshot.csv"
    segmenter_split = segmenter_run / "split_snapshot.csv"
    classifier_split_hash = classifier.sha256(classifier_split)
    segmenter_split_hash = smoke.digest(segmenter_split)
    expected_split_hash = classifier_findings["split_sha256"]
    if not (
        classifier_split_hash
        == segmenter_split_hash
        == expected_split_hash
        == classifier_config["split_sha256"]
        == segmenter_config["split_sha256"]
    ):
        raise ValueError("Classifier/segmenter split SHA256 mismatch")

    split_rows = agreement.load_csv(classifier_split)
    split_counts = Counter(row["draft_split"] for row in split_rows)
    if split_counts != {
        "train": 3570,
        "validation": 893,
        "test": 1000,
        "review_excluded": 537,
    }:
        raise ValueError(f"Frozen split counts changed: {dict(split_counts)}")
    classifier_validation = [
        row for row in split_rows if row["draft_split"] == "validation"
    ]
    if any(row["original_split"] != "train" for row in classifier_validation):
        raise ValueError("Classifier validation includes original test data")
    for row in classifier_validation:
        if classifier.sha256(classifier.image_path(row)) != row["sha256"]:
            raise ValueError(
                f"Classifier validation image hash mismatch: {row['path']}"
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_state = torch.load(
        classifier_checkpoint, map_location="cpu", weights_only=True
    )
    if (
        classifier_state["epoch"] != classifier_findings["best_epoch"]
        or classifier_state["config"]["classes"] != classifier.CLASSES
        or classifier_state["config"]["script_sha256"]
        != classifier_config["script_sha256"]
    ):
        raise ValueError("Classifier checkpoint metadata mismatch")
    classifier_model = classifier.resnet18(weights=None, num_classes=4).to(device)
    classifier_model.load_state_dict(classifier_state["model_state"])
    classifier_labels, classifier_probabilities = agreement.classifier_inference(
        classifier_model, classifier_validation, device
    )
    classifier_metrics, _ = classifier.metrics_for(
        classifier_labels, classifier_probabilities
    )
    expected_classifier_metrics = json.loads(
        (classifier_run / "validation_metrics.json").read_text(encoding="utf-8")
    )["validation"]
    classifier_metric_delta = max(
        abs(classifier_metrics[key] - expected_classifier_metrics[key])
        for key in expected_classifier_metrics
    )
    classifier_tolerance = 1e-6 if device.type == "cuda" else 1e-5
    if classifier_metric_delta > classifier_tolerance:
        raise ValueError(
            f"Classifier validation metric mismatch: {classifier_metric_delta}"
        )
    classifier_records = agreement.build_classifier_records(
        classifier_validation, classifier_probabilities
    )
    stored_classifier = {
        agreement.normalized_path(row["path"]): row
        for row in agreement.load_csv(classifier_run / "validation_predictions.csv")
    }
    if set(stored_classifier) != {
        agreement.normalized_path(row["path"]) for row in classifier_records
    }:
        raise ValueError("Stored classifier prediction membership mismatch")
    classifier_probability_delta = 0.0
    for row in classifier_records:
        stored = stored_classifier[agreement.normalized_path(row["path"])]
        if (
            stored["label"] != row["category"]
            or stored["predicted"] != row["predicted_class"]
        ):
            raise ValueError(f"Stored classifier decision mismatch: {row['path']}")
        for key in (
            "tumor_score",
            "score_glioma",
            "score_meningioma",
            "score_pituitary",
            "score_no_tumor",
        ):
            classifier_probability_delta = max(
                classifier_probability_delta,
                abs(row[key] - float(stored[key])),
            )
    if classifier_probability_delta > classifier_tolerance:
        raise ValueError(
            f"Stored classifier probability mismatch: {classifier_probability_delta}"
        )

    segmentation_rows = smoke.read_csv(segmenter_split)
    segmentation_parts = segmenter.partition_masked_rows(segmentation_rows)
    segmentation_validation = segmentation_parts["validation"]
    classifier_paths = {
        agreement.normalized_path(row["path"]) for row in classifier_validation
    }
    if len(segmentation_validation) != 680 or any(
        agreement.normalized_path(row["path"]) not in classifier_paths
        for row in segmentation_validation
    ):
        raise ValueError("Segmenter validation is not an exact classifier subset")
    segmenter_model = overfit.make_diagnostic_model("linear", "compact").to(device)
    segmenter_state = torch.load(
        segmenter_checkpoint, map_location="cpu", weights_only=True
    )
    segmenter_model.load_state_dict(segmenter_state["model_state"])
    segmenter_metrics, segmenter_records, predictions = segmenter_inference(
        segmenter_model, segmentation_validation, device
    )
    expected_segmenter_metrics = json.loads(
        (segmenter_run / "result.json").read_text(encoding="utf-8")
    )["best_validation_metrics"]
    segmenter_tolerance = 1e-6 if device.type == "cuda" else 1e-5
    segmenter_metric_delta = max(
        abs(segmenter_metrics[key] - expected_segmenter_metrics[key])
        for key in expected_segmenter_metrics
    )
    if segmenter_metric_delta > segmenter_tolerance:
        raise ValueError(
            f"Segmenter validation metric mismatch: {segmenter_metric_delta}"
        )
    stored_segmenter = {
        agreement.normalized_path(row["path"]): row
        for row in smoke.read_csv(segmenter_run / "validation_predictions.csv")
    }
    if set(stored_segmenter) != {
        agreement.normalized_path(row["path"]) for row in segmenter_records
    }:
        raise ValueError("Stored segmenter prediction membership mismatch")
    segmenter_per_image_delta = 0.0
    for row in segmenter_records:
        stored = stored_segmenter[agreement.normalized_path(row["path"])]
        if (
            stored["label"] != row["category"]
            or int(stored["reference_pixels"]) != row["reference_pixels"]
            or int(stored["predicted_pixels"]) != row["predicted_pixels"]
        ):
            raise ValueError(f"Stored segmenter record mismatch: {row['path']}")
        segmenter_per_image_delta = max(
            segmenter_per_image_delta,
            abs(float(stored["dice"]) - row["dice"]),
            abs(float(stored["iou"]) - row["iou"]),
        )
    if segmenter_per_image_delta > segmenter_tolerance:
        raise ValueError(
            f"Stored segmenter per-image mismatch: {segmenter_per_image_delta}"
        )

    zero_records, best_records = select_groups(segmenter_records)
    source_lookup = {
        agreement.normalized_path(row["path"]): row for row in segmentation_validation
    }
    zero_cases = join_classifier_signals(
        zero_records, classifier_records, source_lookup
    )
    best_cases = join_classifier_signals(
        best_records, classifier_records, source_lookup
    )
    all_cases = join_classifier_signals(
        segmenter_records, classifier_records, source_lookup
    )
    if any(
        case["draft_split"] != "validation" or case["original_split"] != "train"
        for case in (*zero_cases, *best_cases)
    ):
        raise ValueError("Selected uncertainty case has non-validation provenance")

    section_c_special = section_c["special_cases"][
        "flag_classifier_no_tumor_segmenter_nonempty"
    ]["cases"]
    if len(section_c_special) != 1:
        raise ValueError("Expected one Section C no-tumor/non-empty special case")
    record_lookup = {
        agreement.normalized_path(row["path"]): row for row in segmenter_records
    }
    special_record = record_lookup[
        agreement.normalized_path(section_c_special[0]["path"])
    ]
    zero_glioma = sorted(
        [row for row in zero_records if row["category"] == "glioma"],
        key=lambda row: agreement.normalized_path(row["path"]),
    )
    if len(zero_glioma) != 12:
        raise ValueError(f"Expected 12 zero-Dice gliomas, found {len(zero_glioma)}")
    sampled_glioma = [
        zero_glioma[0],
        zero_glioma[len(zero_glioma) // 3],
        zero_glioma[2 * len(zero_glioma) // 3],
        zero_glioma[-1],
    ]
    manual_groups = [
        ("Classifier no-tumor / segmenter non-empty", [special_record]),
        ("Sampled zero-Dice gliomas", sampled_glioma),
    ]
    panel_path = segmenter_run / "section_d_manual_review.png"
    selection_path = segmenter_run / "section_d_manual_review_selections.csv"
    row_lookup = {row["path"]: row for row in segmentation_validation}
    visual.render_panel(
        panel_path,
        "Phase 4 Section D manual review",
        manual_groups,
        row_lookup,
        predictions,
    )
    write_manual_selection(selection_path, manual_groups)

    result = {
        "schema_version": 1,
        "status": "SECTION_D_SIGNALS_COMPLETE_USER_VISUAL_REVIEW_PENDING",
        "scope": "Experimental BRISC internal validation uncertainty signals; not clinical validation",
        "group_definitions": {
            "zero_dice": "all 17 validation cases with hard-mask Dice exactly 0 at threshold 0.50",
            "highest_dice_20": "20 validation cases with highest hard-mask Dice; normalized path breaks ties",
        },
        "groups": {
            "zero_dice": {"count": len(zero_cases), "cases": zero_cases},
            "highest_dice_20": {"count": len(best_cases), "cases": best_cases},
        },
        "group_summaries": {
            "zero_dice": summarize_group(zero_cases),
            "highest_dice_20": summarize_group(best_cases),
        },
        "cross_model_correlations_all_680": cross_model_correlations(all_cases),
        "section_c_context": {
            **expected_context,
            "source_sha256": classifier.sha256(section_c_path),
        },
        "manual_review": {
            "status": "USER_REVIEW_PENDING",
            "panel": str(panel_path),
            "selection_csv": str(selection_path),
            "special_case_count": 1,
            "sampled_zero_dice_glioma_count": len(sampled_glioma),
        },
        "verification": {
            "device": str(device),
            "shared_split_sha256": expected_split_hash,
            "classifier_checkpoint_sha256": classifier_checkpoint_hash,
            "segmenter_checkpoint_sha256": segmenter_checkpoint_hash,
            "classifier_validation_images_recomputed": len(classifier_records),
            "classifier_maximum_metric_difference": classifier_metric_delta,
            "classifier_maximum_probability_difference": classifier_probability_delta,
            "segmenter_validation_images_recomputed": len(segmenter_records),
            "segmenter_maximum_metric_difference": segmenter_metric_delta,
            "segmenter_maximum_per_image_difference": segmenter_per_image_delta,
            "test_images_accessed": 0,
        },
        "limitations": [
            "The two groups are deliberately selected extremes, not random samples.",
            "BRISC patient independence is not verified.",
            "The sealed test set was not evaluated.",
            "Calibration can quantify or improve score reliability but cannot correct spatial localization errors.",
        ],
    }
    json_path = ROOT / "audit/phase4_uncertainty_signals.json"
    markdown_path = ROOT / "audit/phase4_uncertainty_signals.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(markdown_path, result)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=segmenter_config["mlflow_run_id"]):
        for artifact in (
            json_path,
            markdown_path,
            panel_path,
            selection_path,
        ):
            mlflow.log_artifact(str(artifact))
    print(
        json.dumps(
            {
                "status": result["status"],
                "group_summaries": result["group_summaries"],
                "cross_model_correlations_all_680": result[
                    "cross_model_correlations_all_680"
                ],
                "manual_review": result["manual_review"],
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
