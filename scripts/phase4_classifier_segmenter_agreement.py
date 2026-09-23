"""Audit classifier/segmenter agreement on the shared held-out validation subset."""

import argparse
import csv
import json
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import phase3_classifier as classifier
import phase4_overfit as overfit
import phase4_segmentation_smoke as smoke
import phase4_train as segmenter
import phase4_visual_audit as visual
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
CLASSIFIER_RUN_NAME = "20260921-170607-243133-baseline"
SEGMENTER_RUN_NAME = "20260923-072439-397263-full"
CLASSIFIER_CHECKPOINT_NAME = "best_model.pt"
SEGMENTER_CHECKPOINT_NAME = "epoch_056_macro_dice_0.762662.pt"
THRESHOLD = 0.5


def normalized_path(value):
    return value.replace("\\", "/").casefold()


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def classifier_inference(model, rows, device):
    loader = DataLoader(
        classifier.BrainImages(rows, training=False),
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    labels = []
    probabilities = []
    model.eval()
    with torch.inference_mode():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(images)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite classifier output")
            probabilities.append(logits.float().softmax(dim=1).cpu().numpy())
            labels.extend(batch_labels.tolist())
    return np.asarray(labels), np.concatenate(probabilities)


def build_classifier_records(rows, probabilities):
    records = []
    for row, probability in zip(rows, probabilities, strict=True):
        tumor_score = float(probability[:3].sum())
        records.append(
            {
                "path": row["path"],
                "category": row["label"],
                "original_split": row["original_split"],
                "draft_split": row["draft_split"],
                "predicted_class": classifier.CLASSES[int(probability.argmax())],
                "binary_prediction": (
                    "tumor" if tumor_score >= THRESHOLD else "no_tumor"
                ),
                "tumor_score": tumor_score,
                **{
                    f"score_{name}": float(probability[index])
                    for index, name in enumerate(classifier.CLASSES)
                },
            }
        )
    return records


def build_agreement(classifier_records, segmenter_records):
    classifier_lookup = {
        normalized_path(row["path"]): row for row in classifier_records
    }
    if len(classifier_lookup) != len(classifier_records):
        raise ValueError("Duplicate classifier paths")
    rows = []
    for segmentation in sorted(
        segmenter_records, key=lambda row: normalized_path(row["path"])
    ):
        key = normalized_path(segmentation["path"])
        if key not in classifier_lookup:
            raise ValueError(f"Segmenter path absent from classifier validation: {key}")
        classification = classifier_lookup[key]
        if classification["category"] != segmentation["label"]:
            raise ValueError(f"Classifier/segmenter label mismatch: {key}")
        if (
            classification["original_split"] != "train"
            or classification["draft_split"] != "validation"
        ):
            raise ValueError(f"Non-validation provenance in agreement set: {key}")
        classifier_correct = (
            classification["predicted_class"] == classification["category"]
        )
        segmenter_success = segmentation["dice"] >= THRESHOLD
        segmenter_nonempty = segmentation["predicted_pixels"] > 0
        classifier_no_tumor = classification["binary_prediction"] == "no_tumor"
        agreement = {
            "path": segmentation["path"],
            "category": segmentation["label"],
            "original_split": classification["original_split"],
            "draft_split": classification["draft_split"],
            "classifier_predicted_class": classification["predicted_class"],
            "classifier_binary_prediction": classification["binary_prediction"],
            "classifier_correct": classifier_correct,
            "classifier_tumor_score": classification["tumor_score"],
            "classifier_score_glioma": classification["score_glioma"],
            "classifier_score_meningioma": classification["score_meningioma"],
            "classifier_score_pituitary": classification["score_pituitary"],
            "classifier_score_no_tumor": classification["score_no_tumor"],
            "segmenter_dice": segmentation["dice"],
            "segmenter_iou": segmentation["iou"],
            "segmenter_reference_pixels": segmentation["reference_pixels"],
            "segmenter_predicted_pixels": segmentation["predicted_pixels"],
            "segmenter_dice_at_least_0_50": segmenter_success,
            "segmenter_prediction_nonempty": segmenter_nonempty,
            "flag_classifier_no_tumor_segmenter_nonempty": (
                classifier_no_tumor and segmenter_nonempty
            ),
            "flag_classifier_tumor_segmenter_empty": (
                not classifier_no_tumor and not segmenter_nonempty
            ),
            "flag_classifier_correct_type_segmenter_zero_dice": (
                classifier_correct and segmentation["dice"] == 0.0
            ),
        }
        rows.append(agreement)
    if len({normalized_path(row["path"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate agreement paths")
    return rows


def summarize_agreement(rows):
    def count(classifier_correct, segmenter_success):
        return sum(
            row["classifier_correct"] == classifier_correct
            and row["segmenter_dice_at_least_0_50"] == segmenter_success
            for row in rows
        )

    table = {
        "classifier_correct": {
            "segmenter_dice_at_least_0_50": count(True, True),
            "segmenter_dice_below_0_50": count(True, False),
        },
        "classifier_incorrect": {
            "segmenter_dice_at_least_0_50": count(False, True),
            "segmenter_dice_below_0_50": count(False, False),
        },
    }
    for classifier_state in table.values():
        classifier_state["total"] = (
            classifier_state["segmenter_dice_at_least_0_50"]
            + classifier_state["segmenter_dice_below_0_50"]
        )
    if sum(state["total"] for state in table.values()) != len(rows):
        raise ValueError("Agreement table does not cover every row")
    flag_names = (
        "flag_classifier_no_tumor_segmenter_nonempty",
        "flag_classifier_tumor_segmenter_empty",
        "flag_classifier_correct_type_segmenter_zero_dice",
    )
    special_cases = {}
    for flag in flag_names:
        members = [row for row in rows if row[flag]]
        special_cases[flag] = {
            "count": len(members),
            "cases": [
                {
                    "path": row["path"],
                    "category": row["category"],
                    "classifier_predicted_class": row["classifier_predicted_class"],
                    "classifier_binary_prediction": row["classifier_binary_prediction"],
                    "classifier_tumor_score": row["classifier_tumor_score"],
                    "segmenter_dice": row["segmenter_dice"],
                    "segmenter_predicted_pixels": row["segmenter_predicted_pixels"],
                }
                for row in members
            ],
        }
    return table, special_cases


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percent(count, total):
    return f"{100 * count / total:.1f}%"


def special_case_markdown(title, value):
    lines = [f"### {title}", "", f"Count: **{value['count']}**", ""]
    if not value["cases"]:
        lines.append("No cases.")
        return "\n".join(lines)
    lines.extend(
        [
            "| Path | True type | Classifier class | Binary decision | Tumor score | Segmenter Dice | Predicted pixels |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for case in value["cases"]:
        lines.append(
            "| {path} | {category} | {classifier_predicted_class} | "
            "{classifier_binary_prediction} | {classifier_tumor_score:.6f} | "
            "{segmenter_dice:.6f} | {segmenter_predicted_pixels} |".format(**case)
        )
    return "\n".join(lines)


def write_markdown(path, result):
    table = result["agreement_2x2"]
    total = result["validation_images_joined"]
    cc = table["classifier_correct"]
    ci = table["classifier_incorrect"]
    special = result["special_cases"]
    observations = result["observations"]
    content = f"""# Phase 4 classifier/segmenter agreement

This verification joins the accepted Phase 3 classifier and accepted Phase 4 segmenter on the same 680 held-out BRISC validation images that have tumor masks. No test image was accessed. Four-class classifier correctness is evaluated against the tumor type. Segmenter success means hard-mask Dice at or above 0.50.

## Verification

- Shared split SHA256: `{result["verification"]["shared_split_sha256"]}`
- Classifier checkpoint SHA256: `{result["verification"]["classifier_checkpoint_sha256"]}`
- Segmenter checkpoint SHA256: `{result["verification"]["segmenter_checkpoint_sha256"]}`
- Classifier validation images recomputed: {result["verification"]["classifier_validation_images_recomputed"]}
- Classifier maximum metric difference: {result["verification"]["classifier_maximum_metric_difference"]}
- Classifier maximum stored-probability difference: {result["verification"]["classifier_maximum_probability_difference"]}
- Segmenter validation images recomputed: {result["verification"]["segmenter_validation_images_recomputed"]}
- Segmenter maximum metric difference: {result["verification"]["segmenter_maximum_metric_difference"]}
- Segmenter maximum per-image metric difference: {result["verification"]["segmenter_maximum_per_image_difference"]}
- Test images accessed: {result["verification"]["test_images_accessed"]}

## Agreement table

| Classifier result | Segmenter Dice >= 0.50 | Segmenter Dice < 0.50 | Total |
|---|---:|---:|---:|
| Correct tumor type | {cc["segmenter_dice_at_least_0_50"]} ({percent(cc["segmenter_dice_at_least_0_50"], total)}) | {cc["segmenter_dice_below_0_50"]} ({percent(cc["segmenter_dice_below_0_50"], total)}) | {cc["total"]} |
| Incorrect tumor type / no tumor | {ci["segmenter_dice_at_least_0_50"]} ({percent(ci["segmenter_dice_at_least_0_50"], total)}) | {ci["segmenter_dice_below_0_50"]} ({percent(ci["segmenter_dice_below_0_50"], total)}) | {ci["total"]} |
| Total | {cc["segmenter_dice_at_least_0_50"] + ci["segmenter_dice_at_least_0_50"]} | {cc["segmenter_dice_below_0_50"] + ci["segmenter_dice_below_0_50"]} | {total} |

Percentages in the table are shares of all 680 joined cases.

## Interpretation

- The classifier predicted the correct tumor type in {observations["classifier_correct_count"]} of 680 cases ({percent(observations["classifier_correct_count"], total)}).
- The segmenter reached Dice >= 0.50 in {observations["segmenter_success_count"]} of 680 cases ({percent(observations["segmenter_success_count"], total)}).
- The classifier remained correct in {observations["segmenter_failures_with_correct_classifier"]} of 83 segmentation failures ({percent(observations["segmenter_failures_with_correct_classifier"], observations["segmenter_failure_count"])}). Correct type recognition therefore does not guarantee correct localization.
- The segmenter succeeded in {observations["classifier_errors_with_successful_segmenter"]} of 19 classifier errors ({percent(observations["classifier_errors_with_successful_segmenter"], observations["classifier_error_count"])}).
- Both components failed their respective criteria in {observations["both_failed_count"]} cases.
- Of the {observations["zero_dice_count"]} zero-Dice cases, {observations["zero_dice_with_correct_classifier_type"]} still had the correct classifier type.

The binary special-case flags below use the classifier's unchanged tumor-score threshold of 0.50. The four-class prediction is retained separately in every row of the CSV.

## Special cases

{special_case_markdown("Classifier says no tumor; segmenter predicts a non-empty region", special["flag_classifier_no_tumor_segmenter_nonempty"])}

{special_case_markdown("Classifier says tumor; segmenter predicts an empty region", special["flag_classifier_tumor_segmenter_empty"])}

{special_case_markdown("Classifier has the correct tumor type; segmenter Dice is zero", special["flag_classifier_correct_type_segmenter_zero_dice"])}

## Scope

These are paired behaviors on BRISC's internal validation partition. They do not measure clinical validity, patient-independent generalization, or performance on the sealed test set.
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
    if classifier_findings["baseline_run"] != classifier_run.name:
        raise ValueError("Accepted classifier run mismatch")
    if segmenter_findings["accepted_run"] != segmenter_run.name:
        raise ValueError("Accepted segmenter run mismatch")

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

    split_rows = load_csv(classifier_split)
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
    if len({normalized_path(row["path"]) for row in classifier_validation}) != 893:
        raise ValueError("Classifier validation paths are not unique")
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
    classifier_labels, classifier_probabilities = classifier_inference(
        classifier_model, classifier_validation, device
    )
    classifier_metrics, _ = classifier.metrics_for(
        classifier_labels, classifier_probabilities
    )
    expected_classifier_result = json.loads(
        (classifier_run / "validation_metrics.json").read_text(encoding="utf-8")
    )["validation"]
    classifier_metric_delta = max(
        abs(classifier_metrics[key] - expected_classifier_result[key])
        for key in expected_classifier_result
    )
    classifier_tolerance = 1e-6 if device.type == "cuda" else 1e-5
    if classifier_metric_delta > classifier_tolerance:
        raise ValueError(
            f"Classifier validation metric mismatch: {classifier_metric_delta}"
        )
    classifier_records = build_classifier_records(
        classifier_validation, classifier_probabilities
    )
    stored_classifier = {
        normalized_path(row["path"]): row
        for row in load_csv(classifier_run / "validation_predictions.csv")
    }
    if set(stored_classifier) != {
        normalized_path(row["path"]) for row in classifier_records
    }:
        raise ValueError("Stored classifier prediction membership mismatch")
    probability_delta = 0.0
    for row in classifier_records:
        stored = stored_classifier[normalized_path(row["path"])]
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
            probability_delta = max(
                probability_delta, abs(row[key] - float(stored[key]))
            )
    if probability_delta > classifier_tolerance:
        raise ValueError(f"Stored classifier probability mismatch: {probability_delta}")

    segmentation_rows = smoke.read_csv(segmenter_split)
    segmentation_parts = segmenter.partition_masked_rows(segmentation_rows)
    segmentation_validation = segmentation_parts["validation"]
    classifier_validation_paths = {
        normalized_path(row["path"]) for row in classifier_validation
    }
    if len(segmentation_validation) != 680 or any(
        normalized_path(row["path"]) not in classifier_validation_paths
        for row in segmentation_validation
    ):
        raise ValueError("Segmenter validation is not an exact classifier subset")
    segmenter_model = overfit.make_diagnostic_model("linear", "compact").to(device)
    segmenter_state = torch.load(
        segmenter_checkpoint, map_location="cpu", weights_only=True
    )
    segmenter_model.load_state_dict(segmenter_state["model_state"])
    segmenter_metrics, segmenter_records, _ = visual.inference(
        segmenter_model, segmentation_validation, device, THRESHOLD
    )
    expected_segmenter_result = json.loads(
        (segmenter_run / "result.json").read_text(encoding="utf-8")
    )["best_validation_metrics"]
    segmenter_tolerance = 1e-6 if device.type == "cuda" else 1e-5
    segmenter_metric_delta = max(
        abs(segmenter_metrics[key] - expected_segmenter_result[key])
        for key in expected_segmenter_result
    )
    if segmenter_metric_delta > segmenter_tolerance:
        raise ValueError(
            f"Segmenter validation metric mismatch: {segmenter_metric_delta}"
        )
    stored_segmenter = {
        normalized_path(row["path"]): row
        for row in smoke.read_csv(segmenter_run / "validation_predictions.csv")
    }
    if set(stored_segmenter) != {
        normalized_path(row["path"]) for row in segmenter_records
    }:
        raise ValueError("Stored segmenter prediction membership mismatch")
    per_image_delta = 0.0
    for row in segmenter_records:
        stored = stored_segmenter[normalized_path(row["path"])]
        if (
            stored["label"] != row["label"]
            or int(stored["reference_pixels"]) != row["reference_pixels"]
            or int(stored["predicted_pixels"]) != row["predicted_pixels"]
        ):
            raise ValueError(f"Stored segmenter record mismatch: {row['path']}")
        per_image_delta = max(
            per_image_delta,
            abs(float(stored["dice"]) - row["dice"]),
            abs(float(stored["iou"]) - row["iou"]),
        )
    if per_image_delta > segmenter_tolerance:
        raise ValueError(f"Stored segmenter per-image mismatch: {per_image_delta}")

    agreement_rows = build_agreement(classifier_records, segmenter_records)
    if len(agreement_rows) != 680:
        raise ValueError(f"Unexpected joined validation count: {len(agreement_rows)}")
    agreement_table, special_cases = summarize_agreement(agreement_rows)
    classifier_correct_count = agreement_table["classifier_correct"]["total"]
    classifier_error_count = agreement_table["classifier_incorrect"]["total"]
    segmenter_success_count = sum(
        state["segmenter_dice_at_least_0_50"] for state in agreement_table.values()
    )
    segmenter_failure_count = len(agreement_rows) - segmenter_success_count
    zero_dice_count = sum(row["segmenter_dice"] == 0.0 for row in agreement_rows)
    observations = {
        "classifier_correct_count": classifier_correct_count,
        "classifier_error_count": classifier_error_count,
        "segmenter_success_count": segmenter_success_count,
        "segmenter_failure_count": segmenter_failure_count,
        "segmenter_failures_with_correct_classifier": agreement_table[
            "classifier_correct"
        ]["segmenter_dice_below_0_50"],
        "classifier_errors_with_successful_segmenter": agreement_table[
            "classifier_incorrect"
        ]["segmenter_dice_at_least_0_50"],
        "both_failed_count": agreement_table["classifier_incorrect"][
            "segmenter_dice_below_0_50"
        ],
        "zero_dice_count": zero_dice_count,
        "zero_dice_with_correct_classifier_type": special_cases[
            "flag_classifier_correct_type_segmenter_zero_dice"
        ]["count"],
    }
    result = {
        "schema_version": 1,
        "status": "SECTION_C_CLASSIFIER_SEGMENTER_AGREEMENT_COMPLETE",
        "scope": "Experimental BRISC internal validation agreement; not clinical validation",
        "validation_images_joined": len(agreement_rows),
        "validation_counts": dict(
            sorted(Counter(row["category"] for row in agreement_rows).items())
        ),
        "definitions": {
            "classifier_correct": "four-class prediction equals the dataset tumor type",
            "segmenter_success": "hard-mask Dice is at least 0.50",
            "classifier_binary_prediction": "tumor score threshold fixed at 0.50",
            "segmenter_nonempty": "hard mask at threshold 0.50 contains at least one foreground pixel",
        },
        "agreement_2x2": agreement_table,
        "observations": observations,
        "special_cases": special_cases,
        "verification": {
            "device": str(device),
            "shared_split_sha256": expected_split_hash,
            "classifier_checkpoint_sha256": classifier_checkpoint_hash,
            "segmenter_checkpoint_sha256": segmenter_checkpoint_hash,
            "classifier_validation_images_recomputed": len(classifier_records),
            "classifier_maximum_metric_difference": classifier_metric_delta,
            "classifier_maximum_probability_difference": probability_delta,
            "segmenter_validation_images_recomputed": len(segmenter_records),
            "segmenter_maximum_metric_difference": segmenter_metric_delta,
            "segmenter_maximum_per_image_difference": per_image_delta,
            "test_images_accessed": 0,
        },
        "limitations": [
            "BRISC patient independence is not verified.",
            "The sealed test set was not evaluated.",
            "Agreement on an internal validation split does not establish clinical validity or cross-dataset generalization.",
        ],
    }
    csv_path = segmenter_run / "phase4_classifier_segmenter_agreement.csv"
    json_path = ROOT / "audit/phase4_classifier_segmenter_agreement.json"
    markdown_path = ROOT / "audit/phase4_classifier_segmenter_agreement.md"
    write_csv(csv_path, agreement_rows)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_markdown(markdown_path, result)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=segmenter_config["mlflow_run_id"]):
        mlflow.log_artifact(str(csv_path))
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(markdown_path))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
