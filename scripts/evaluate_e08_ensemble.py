"""Evaluate the preregistered equal-weight E05-E07 probability ensemble."""

import csv
import hashlib
import json
import subprocess
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_smoke_test as smoke
import segmentation_system as system
import segmentation_visual_audit as visual
import torch
import train_segmenter as train
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/e08_ensemble"
ARTIFACTS = EXPERIMENT / "artifacts"
BASELINE_RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
COMPONENTS = {
    "e05_glioma_weighted": ROOT
    / "runs/segmentation_v2_weighted/20260924-055049-892543-weighted",
    "e06_boundary_loss": ROOT
    / "runs/segmentation_v2_boundary_loss/20260924-063051-647842-boundary",
    "e07_glioma_specialized": ROOT
    / "runs/segmentation_v2_glioma_specialized/20260924-105229-241365-specialist",
}
LABELS = train.LABELS


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


def scores(hard, reference):
    intersection = (hard & reference).sum(dim=(1, 2, 3)).float()
    predicted = hard.sum(dim=(1, 2, 3)).float()
    target = reference.sum(dim=(1, 2, 3)).float()
    dice = 2 * intersection / (predicted + target).clamp_min(1)
    iou = intersection / (predicted + target - intersection).clamp_min(1)
    return dice, iou, predicted, target


def paired_group_interval(rows, label, candidate_field, reference_field):
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
                        item[candidate_field] - item[reference_field]
                        for key in sampled
                        for item in groups[key]
                    ]
                )
            )
        )
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def category_mean(rows, field):
    return {
        label: float(np.mean([row[field] for row in rows if row["category"] == label]))
        for label in LABELS
    }


def render_comparison(path, baseline, comparator, ensemble):
    x = np.arange(len(LABELS))
    width = 0.25
    figure, axis = plt.subplots(figsize=(9.5, 5.4))
    axis.bar(x - width, [baseline[label] for label in LABELS], width, label="Frozen v1")
    axis.bar(
        x,
        [comparator[label] for label in LABELS],
        width,
        label="E07 best eligible single",
    )
    axis.bar(
        x + width,
        [ensemble[label] for label in LABELS],
        width,
        label="E08 ensemble",
    )
    axis.set(
        xticks=x,
        xticklabels=LABELS,
        ylabel="Image-mean hard Dice",
        ylim=(0, 1),
        title="Frozen v1, best eligible single, and fixed ensemble",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main():
    ARTIFACTS.mkdir(exist_ok=True)
    config = read_json(EXPERIMENT / "config.json")
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E08 is not in its registered pre-evaluation state")
    component_config = {item["experiment_id"]: item for item in config["components"]}
    if set(component_config) != set(COMPONENTS):
        raise ValueError("E08 component identity changed")
    weights = [component_config[name]["weight"] for name in COMPONENTS]
    if weights != [1 / 3, 1 / 3, 1 / 3] or not np.isclose(sum(weights), 1.0):
        raise ValueError("E08 weights differ from the registered equal average")

    split = BASELINE_RUN / "split_snapshot.csv"
    if sha256(split) != config["split_sha256"]:
        raise ValueError("E08 split hash mismatch")
    parts = train.partition_masked_rows(smoke.read_csv(split))
    rows = parts["validation"]
    if len(rows) != 680:
        raise ValueError("E08 validation population mismatch")
    row_keys = [normalized_path(row["path"]) for row in rows]
    if len(set(row_keys)) != 680:
        raise ValueError("E08 validation path duplication")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    stored = {}
    for name, run in COMPONENTS.items():
        result = read_json(run / "result.json")
        run_config = read_json(run / "config.json")
        if run_config["split_sha256"] != config["split_sha256"]:
            raise ValueError(f"{name} split hash mismatch")
        selected = result["top3"][0]
        checkpoint = run / "top3_checkpoints" / selected["checkpoint"]
        expected_hash = component_config[name]["checkpoint_sha256"]
        models[name] = system.load_selected_system(
            run, checkpoint, device, expected_hash
        )
        stored[name] = {
            normalized_path(item["path"]): item
            for item in read_csv(run / "validation_predictions.csv")
        }
        if set(stored[name]) != set(row_keys):
            raise ValueError(f"{name} stored prediction membership mismatch")
    e07_routing = COMPONENTS["e07_glioma_specialized"] / "routing.json"
    if sha256(e07_routing) != component_config["e07_glioma_specialized"][
        "routing_sha256"
    ]:
        raise ValueError("E08 E07 routing hash mismatch")

    dataset = train.FullPairs(rows)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output_rows = [None] * len(rows)
    predictions = {}
    component_maximum_difference = {name: 0.0 for name in COMPONENTS}
    with torch.inference_mode():
        for images, targets, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            batch_rows = [rows[index] for index in indices.tolist()]
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            component_probabilities = {}
            component_scores = {}
            with context:
                for name, model in models.items():
                    logits = system.logits_for_rows(model, images, batch_rows)
                    probability = logits.float().sigmoid()
                    component_probabilities[name] = probability
                    component_scores[name] = scores(probability >= 0.5, targets >= 0.5)
            ensemble_probability = sum(component_probabilities.values()) / 3.0
            hard = ensemble_probability >= config["threshold"]
            reference = targets >= 0.5
            dice, iou, predicted_pixels, reference_pixels = scores(hard, reference)
            for position, dataset_index in enumerate(indices.tolist()):
                row = rows[dataset_index]
                key = normalized_path(row["path"])
                component_values = {}
                for name, values in component_scores.items():
                    component_dice = float(values[0][position].item())
                    component_iou = float(values[1][position].item())
                    old = stored[name][key]
                    difference = max(
                        abs(component_dice - float(old["dice"])),
                        abs(component_iou - float(old["iou"])),
                    )
                    component_maximum_difference[name] = max(
                        component_maximum_difference[name], difference
                    )
                    component_values[f"{name}_dice"] = component_dice
                    component_values[f"{name}_iou"] = component_iou
                output_rows[dataset_index] = {
                    "path": row["path"],
                    "filename": row["filename"],
                    "category": row["label"],
                    "similarity_group": row["similarity_group"],
                    **component_values,
                    "ensemble_dice": float(dice[position].item()),
                    "ensemble_iou": float(iou[position].item()),
                    "reference_pixels": int(reference_pixels[position].item()),
                    "predicted_pixels": int(predicted_pixels[position].item()),
                }
                predictions[row["path"]] = hard[position, 0].cpu().numpy()
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    if any(value > tolerance for value in component_maximum_difference.values()):
        raise ValueError(
            f"E08 component prediction verification failed: {component_maximum_difference}"
        )
    if any(row is None for row in output_rows):
        raise ValueError("E08 inference did not populate every validation row")

    baseline = {
        normalized_path(row["path"]): row
        for row in read_csv(BASELINE_RUN / "validation_predictions.csv")
    }
    for row in output_rows:
        old = baseline[normalized_path(row["path"])]
        row["baseline_dice"] = float(old["dice"])
        row["baseline_iou"] = float(old["iou"])
        row["ensemble_minus_v1_dice"] = row["ensemble_dice"] - row["baseline_dice"]
        row["ensemble_minus_e07_dice"] = (
            row["ensemble_dice"] - row["e07_glioma_specialized_dice"]
        )
    write_csv(ARTIFACTS / "per_case_comparison.csv", output_rows)

    baseline_mean = category_mean(output_rows, "baseline_dice")
    comparator_mean = category_mean(output_rows, "e07_glioma_specialized_dice")
    ensemble_mean = category_mean(output_rows, "ensemble_dice")
    delta_vs_v1 = {
        label: ensemble_mean[label] - baseline_mean[label] for label in LABELS
    }
    delta_vs_comparator = {
        label: ensemble_mean[label] - comparator_mean[label] for label in LABELS
    }
    intervals = {
        "glioma_vs_best_eligible_single": paired_group_interval(
            output_rows,
            "glioma",
            "ensemble_dice",
            "e07_glioma_specialized_dice",
        ),
        "meningioma_vs_v1": paired_group_interval(
            output_rows, "meningioma", "ensemble_dice", "baseline_dice"
        ),
        "pituitary_vs_v1": paired_group_interval(
            output_rows, "pituitary", "ensemble_dice", "baseline_dice"
        ),
    }
    checks = {
        "glioma_delta_over_best_eligible_single_at_least_0_02": (
            delta_vs_comparator["glioma"] >= 0.02
        ),
        "meningioma_delta_from_v1_at_least_minus_0_01": (
            delta_vs_v1["meningioma"] >= -0.01
        ),
        "pituitary_delta_from_v1_at_least_minus_0_01": (
            delta_vs_v1["pituitary"] >= -0.01
        ),
    }
    render_comparison(
        ARTIFACTS / "dice_comparison.png",
        baseline_mean,
        comparator_mean,
        ensemble_mean,
    )
    visual_records = [
        {
            "path": row["path"],
            "filename": row["filename"],
            "label": row["category"],
            "dice": row["ensemble_dice"],
            "iou": row["ensemble_iou"],
            "reference_pixels": row["reference_pixels"],
            "predicted_pixels": row["predicted_pixels"],
        }
        for row in output_rows
    ]
    zero_count = min(6, sum(row["dice"] == 0 for row in visual_records))
    if zero_count == 0:
        raise ValueError("E08 visual audit requires at least one zero-Dice case")
    groups = visual.select_cases(
        visual_records, worst_per_category=6, zero_count=zero_count, best_per_category=6
    )
    worst = [
        ("Worst glioma", groups["worst_glioma"]),
        ("Worst meningioma", groups["worst_meningioma"]),
        ("Worst pituitary", groups["worst_pituitary"]),
        ("Zero-Dice validation cases", groups["zero_dice"]),
    ]
    best = [
        ("Best glioma", groups["best_glioma"]),
        ("Best meningioma", groups["best_meningioma"]),
        ("Best pituitary", groups["best_pituitary"]),
    ]
    row_lookup = {row["path"]: row for row in rows}
    visual.render_panel(
        ARTIFACTS / "worst_validation_by_category.png",
        "Worst fixed-ensemble validation cases",
        worst,
        row_lookup,
        predictions,
    )
    visual.render_panel(
        ARTIFACTS / "best_validation_by_category.png",
        "Best fixed-ensemble validation cases",
        best,
        row_lookup,
        predictions,
    )
    visual.write_selection(
        ARTIFACTS / "visual_audit_selections.csv",
        {"worst": worst, "best": best},
    )
    result = {
        "schema_version": 1,
        "experiment_id": "e08_ensemble",
        "status": "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING",
        "scope": "BRISC internal validation development result; not clinical validation",
        "evaluation_source_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "evaluation_script_sha256": sha256(Path(__file__)),
        "validation_cases": 680,
        "test_images_accessed": 0,
        "component_maximum_recomputed_metric_difference": component_maximum_difference,
        "mean_dice": {
            "baseline_v1": baseline_mean,
            "best_eligible_single_e07": comparator_mean,
            "ensemble": ensemble_mean,
            "ensemble_delta_vs_v1": delta_vs_v1,
            "ensemble_delta_vs_best_eligible_single": delta_vs_comparator,
        },
        "paired_similarity_group_bootstrap_ci95": intervals,
        "weak_cases_dice_below_0_50": {
            label: sum(
                row["category"] == label and row["ensemble_dice"] < 0.5
                for row in output_rows
            )
            for label in LABELS
        },
        "gate": {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "rule": "All three registered checks must pass.",
        },
        "visual_review": "PENDING",
        "test_evaluated": False,
    }
    write_json(EXPERIMENT / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
