"""Build and evaluate the preregistered classifier-routed E07 segmenter."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import classifier_segmenter_agreement as agreement
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_smoke_test as smoke
import segmentation_system as system
import torch
import train_classifier as classifier
import train_segmenter as general_training
from torchvision.models import resnet18

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e07_glioma_specialized"
ARTIFACTS = EXPERIMENT / "artifacts"
BASELINE_RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
BASELINE_CHECKPOINT = (
    BASELINE_RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
)
CLASSIFIER_RUN = ROOT / "runs/classifier/20260921-170607-243133-baseline"
CLASSIFIER_CHECKPOINT = CLASSIFIER_RUN / "best_model.pt"
LABELS = general_training.LABELS


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    Path(path).write_bytes(payload.encode("utf-8"))


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


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
        values.append(
            float(
                np.mean(
                    [
                        case["routed_dice"] - case["baseline_dice"]
                        for key in sampled
                        for case in groups[key]
                    ]
                )
            )
        )
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def mean_by_category(rows, field):
    return {
        label: float(np.mean([row[field] for row in rows if row["category"] == label]))
        for label in LABELS
    }


def load_and_verify_classifier(validation_rows, config, device):
    if sha256(CLASSIFIER_CHECKPOINT) != config["classifier_checkpoint_sha256"]:
        raise ValueError("E07 classifier checkpoint hash mismatch")
    stored_path = CLASSIFIER_RUN / "validation_predictions.csv"
    if sha256(stored_path) != config["classifier_validation_predictions_sha256"]:
        raise ValueError("E07 classifier validation-prediction hash mismatch")
    state = torch.load(CLASSIFIER_CHECKPOINT, map_location="cpu", weights_only=True)
    model = resnet18(weights=None, num_classes=4).to(device)
    model.load_state_dict(state["model_state"])
    labels, probabilities = agreement.classifier_inference(
        model, validation_rows, device
    )
    expected_labels = np.asarray(
        [classifier.CLASSES.index(row["label"]) for row in validation_rows]
    )
    if not np.array_equal(labels, expected_labels):
        raise ValueError("Classifier inference label order mismatch")
    stored = {
        normalized_path(row["path"]): row for row in read_csv(stored_path)
    }
    maximum_probability_difference = 0.0
    assignments = []
    for row, probability in zip(validation_rows, probabilities, strict=True):
        key = normalized_path(row["path"])
        old = stored.get(key)
        if old is None:
            raise ValueError(f"Missing stored classifier prediction: {row['path']}")
        predicted = classifier.CLASSES[int(probability.argmax())]
        if old["label"] != row["label"] or old["predicted"] != predicted:
            raise ValueError(f"Frozen classifier decision mismatch: {row['path']}")
        for index, label in enumerate(classifier.CLASSES):
            maximum_probability_difference = max(
                maximum_probability_difference,
                abs(float(probability[index]) - float(old[f"score_{label}"])),
            )
        assignments.append(
            {
                "path": row["path"],
                "true_label": row["label"],
                "classifier_predicted_class": predicted,
                "classifier_correct": predicted == row["label"],
                "selected_segmenter": (
                    "specialist" if predicted == "glioma" else "general_v1"
                ),
                **{
                    f"classifier_score_{label}": float(probability[index])
                    for index, label in enumerate(classifier.CLASSES)
                },
            }
        )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    if maximum_probability_difference > tolerance:
        raise ValueError(
            "Frozen classifier probability mismatch: "
            f"{maximum_probability_difference}"
        )
    return assignments, maximum_probability_difference


def render_comparison(path, baseline, routed):
    x = np.arange(len(LABELS))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5.4))
    axis.bar(x - width / 2, [baseline[label] for label in LABELS], width, label="Frozen v1")
    axis.bar(x + width / 2, [routed[label] for label in LABELS], width, label="E07 routed")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set(
        xticks=x,
        xticklabels=LABELS,
        ylabel="Image-mean hard Dice",
        ylim=(0, 1),
        title="Frozen v1 versus classifier-routed glioma specialist",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    experiment_config = read_json(EXPERIMENT / "config.json")
    if experiment_config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E07 is not in its registered pre-close state")
    run = Path(
        (ROOT / "runs/segmentation_v2_glioma_specialized/latest.txt")
        .read_text(encoding="utf-8")
        .strip()
    ).resolve()
    training_config = read_json(run / "config.json")
    training_result_path = run / "result.json"
    training_result = read_json(training_result_path)
    if training_config["experiment_id"] != "e07_glioma_specialized":
        raise ValueError("E07 training run identity mismatch")
    if training_config["split_sha256"] != experiment_config["split_sha256"]:
        raise ValueError("E07 split hash mismatch")
    if training_config["test_evaluated"] or training_result["test_evaluated"]:
        raise ValueError("E07 accessed the sealed test partition")
    if training_result["best_checkpoint_reload_max_metric_difference"] != 0.0:
        raise ValueError("E07 specialist checkpoint did not reload exactly")
    best = training_result["top3"][0]
    checkpoint = run / "top3_checkpoints" / best["checkpoint"]
    checkpoint_hash = sha256(checkpoint)
    if sha256(BASELINE_CHECKPOINT) != experiment_config[
        "baseline_segmenter_checkpoint_sha256"
    ]:
        raise ValueError("E07 fallback v1 checkpoint hash mismatch")

    split_rows = smoke.read_csv(run / "split_snapshot.csv")
    parts = general_training.partition_masked_rows(split_rows)
    validation_rows = parts["validation"]
    if len(validation_rows) != 680:
        raise ValueError("E07 routed validation population mismatch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assignments, classifier_difference = load_and_verify_classifier(
        validation_rows, experiment_config, device
    )
    assignment_path = run / "routing_assignments.csv"
    write_csv(assignment_path, assignments)
    routing_accuracy = float(np.mean([row["classifier_correct"] for row in assignments]))
    routing = {
        "schema_version": 1,
        "kind": "classifier_predicted_glioma_specialist",
        "specialist_checkpoint": str(checkpoint),
        "specialist_checkpoint_sha256": checkpoint_hash,
        "fallback_checkpoint": str(BASELINE_CHECKPOINT),
        "fallback_checkpoint_sha256": sha256(BASELINE_CHECKPOINT),
        "classifier_checkpoint": str(CLASSIFIER_CHECKPOINT),
        "classifier_checkpoint_sha256": sha256(CLASSIFIER_CHECKPOINT),
        "classifier_validation_predictions": str(
            CLASSIFIER_RUN / "validation_predictions.csv"
        ),
        "classifier_validation_predictions_sha256": sha256(
            CLASSIFIER_RUN / "validation_predictions.csv"
        ),
        "assignments_file": assignment_path.name,
        "assignments_sha256": sha256(assignment_path),
        "routing_accuracy": routing_accuracy,
        "specialist_cases": sum(
            row["selected_segmenter"] == "specialist" for row in assignments
        ),
        "fallback_cases": sum(
            row["selected_segmenter"] == "general_v1" for row in assignments
        ),
        "threshold": 0.5,
        "test_images_accessed": 0,
    }
    write_json(run / "routing.json", routing)

    specialist_predictions = run / "specialist_validation_predictions.csv"
    specialist_result = run / "specialist_training_result.json"
    if not specialist_predictions.exists():
        shutil.copy2(run / "validation_predictions.csv", specialist_predictions)
    if not specialist_result.exists():
        shutil.copy2(training_result_path, specialist_result)

    routed_system = system.load_selected_system(
        run, checkpoint, device, checkpoint_hash
    )
    import calibrate_segmenter_pixels as pixel

    logits, targets, dice, iou, _label_indices, metrics = pixel.infer(
        routed_system, validation_rows, device
    )
    baseline_rows = {
        normalized_path(row["path"]): row
        for row in read_csv(BASELINE_RUN / "validation_predictions.csv")
    }
    assignment_lookup = {
        normalized_path(row["path"]): row for row in assignments
    }
    comparison = []
    routed_predictions = []
    for index, row in enumerate(validation_rows):
        key = normalized_path(row["path"])
        old = baseline_rows[key]
        route = assignment_lookup[key]
        comparison.append(
            {
                "path": row["path"],
                "category": row["label"],
                "similarity_group": row["similarity_group"],
                "classifier_predicted_class": route["classifier_predicted_class"],
                "selected_segmenter": route["selected_segmenter"],
                "baseline_dice": float(old["dice"]),
                "routed_dice": float(dice[index]),
                "dice_delta": float(dice[index]) - float(old["dice"]),
                "baseline_iou": float(old["iou"]),
                "routed_iou": float(iou[index]),
                "iou_delta": float(iou[index]) - float(old["iou"]),
            }
        )
        routed_predictions.append(
            {
                "path": row["path"],
                "filename": row["filename"],
                "label": row["label"],
                "dice": float(dice[index]),
                "iou": float(iou[index]),
                "reference_pixels": int(targets[index].sum()),
                "predicted_pixels": int((logits[index] >= 0).sum()),
                "classifier_predicted_class": route["classifier_predicted_class"],
                "selected_segmenter": route["selected_segmenter"],
            }
        )
    if set(baseline_rows) != {normalized_path(row["path"]) for row in comparison}:
        raise ValueError("E07 baseline/routed membership mismatch")
    write_csv(run / "validation_predictions.csv", routed_predictions)
    write_csv(ARTIFACTS / "per_case_comparison.csv", comparison)

    baseline_dice = mean_by_category(comparison, "baseline_dice")
    routed_dice = mean_by_category(comparison, "routed_dice")
    delta = {
        label: routed_dice[label] - baseline_dice[label] for label in LABELS
    }
    intervals = {
        label: paired_group_interval(comparison, label) for label in LABELS
    }
    checks = {
        "routed_glioma_dice_delta_at_least_0_08": delta["glioma"] >= 0.08,
        "classifier_routing_accuracy_at_least_0_95": routing_accuracy >= 0.95,
        "meningioma_dice_delta_at_least_minus_0_01": delta["meningioma"] >= -0.01,
        "pituitary_dice_delta_at_least_minus_0_01": delta["pituitary"] >= -0.01,
    }
    routed_result = {
        **training_result,
        "status": "ROUTED_VALIDATION_COMPLETE",
        "specialist_best_validation_metrics": training_result[
            "best_validation_metrics"
        ],
        "best_validation_metrics": metrics,
        "routing": routing,
        "classifier_recompute_max_probability_difference": classifier_difference,
        "gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "rule": "All four registered checks must pass.",
        },
        "comparison": {
            "baseline_dice": baseline_dice,
            "routed_dice": routed_dice,
            "dice_delta": delta,
            "paired_similarity_group_bootstrap_ci95_delta": intervals,
        },
        "required_followup_complete": False,
        "test_evaluated": False,
        "test_images_accessed": 0,
    }
    write_json(training_result_path, routed_result)
    selected = {
        "experiment_id": "e07_glioma_specialized",
        "run": str(run),
        "top3_checkpoints": [
            {
                "rank": 1,
                "epoch": training_result["best_epoch"],
                "file": checkpoint.name,
                "sha256": checkpoint_hash.upper(),
            }
        ],
    }
    write_json(ARTIFACTS / "selected_checkpoint.json", selected)
    render_comparison(
        ARTIFACTS / "routed_dice_comparison.png", baseline_dice, routed_dice
    )
    print(json.dumps(routed_result, indent=2))


if __name__ == "__main__":
    main()
