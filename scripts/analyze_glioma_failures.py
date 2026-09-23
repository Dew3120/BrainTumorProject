"""Audit frozen v1 glioma segmentation failures without test-set access."""

import argparse
import csv
import json
import math
import re
from contextlib import nullcontext
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import segmentation_visual_audit as visual
import torch
import train_segmenter as segmenter
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from skimage.filters import sobel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/segmentation/20260923-072439-397263-full"
CHECKPOINT = RUN / "top3_checkpoints/epoch_056_macro_dice_0.762662.pt"
OUTPUT_DECOMPOSITION = ROOT / "audit/glioma_failure_decomposition.csv"
OUTPUT_SHARPNESS = ROOT / "audit/glioma_boundary_sharpness.csv"
OUTPUT_CASE_STUDY = ROOT / "audit/glioma_case_study.md"
OUTPUT_CASE_FIGURE = ROOT / "audit/glioma_case_study_pairs.png"
OUTPUT_SUBGROUP = ROOT / "audit/glioma_subgroup_analysis.md"
OUTPUT_VERIFICATION = ROOT / "audit/glioma_failure_analysis_verification.json"
REVIEW_ANNOTATIONS = ROOT / "audit/glioma_visual_review_annotations.csv"
REVIEW_PANEL_PREFIX = ROOT / "audit/glioma_failure_review"

MODEL_SIZE = 128
THRESHOLD = 0.50
EXPECTED_CHECKPOINT_SHA256 = (
    "16DC8745993EEF959D0F2924C9147A45320673F6FC0B465AB08F8AD7B2F723DE"
)
EXPECTED_SPLIT_SHA256 = (
    "d97ef6d8fed52c7c972dbc33f5de4ac4bb0be994e937a95dffa505615cb8ab28"
)
STRUCTURE_8 = np.ones((3, 3), dtype=np.uint8)
SHARP_EDGE_THRESHOLD = 0.10
BOOTSTRAP_REPETITIONS = 2000
BOOTSTRAP_SEED = 42


def normalized_path(value):
    return str(value).replace("\\", "/").casefold()


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, allow_nan=False), encoding="utf-8"
    )


def infer_all(model, rows, device):
    dataset = segmenter.FullPairs(rows, size=MODEL_SIZE)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    model.eval()
    records = []
    masks = {}
    images = {}
    references = {}
    dice_values = []
    iou_values = []
    label_values = []
    with torch.inference_mode():
        for inputs, targets, label_indices, indices in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite frozen-model logits")
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
            dice_values.extend(dice.cpu().tolist())
            iou_values.extend(iou.cpu().tolist())
            label_values.extend(label_indices.tolist())
            for position, dataset_index in enumerate(indices.tolist()):
                row = rows[dataset_index]
                key = normalized_path(row["path"])
                masks[key] = predicted[position, 0].cpu().numpy().astype(bool)
                references[key] = reference[position, 0].cpu().numpy().astype(bool)
                images[key] = inputs[position, 0].float().cpu().numpy()
                records.append(
                    {
                        "path": row["path"],
                        "filename": row["filename"],
                        "label": row["label"],
                        "dice": float(dice[position].item()),
                        "iou": float(iou[position].item()),
                        "reference_pixels": int(reference_area[position].item()),
                        "predicted_pixels": int(predicted_area[position].item()),
                    }
                )
    metrics = segmenter.summarize_scores(
        np.asarray(dice_values),
        np.asarray(iou_values),
        np.asarray(label_values),
    )
    return metrics, records, masks, references, images


def component_summary(mask):
    mask = np.asarray(mask, dtype=bool)
    labeled, raw_count = ndimage.label(mask, structure=STRUCTURE_8)
    if raw_count:
        sizes = np.bincount(labeled.ravel())[1:]
        retained = sizes >= 4
        count_ge4 = int(retained.sum())
        largest_fraction = float(sizes.max() / mask.sum())
    else:
        count_ge4 = 0
        largest_fraction = math.nan
    return int(raw_count), count_ge4, largest_fraction


def centroid(mask):
    points = np.argwhere(mask)
    if not len(points):
        return math.nan, math.nan
    y, x = points.mean(axis=0)
    return float(x), float(y)


def bbox(mask):
    points = np.argwhere(mask)
    if not len(points):
        return None
    y0, x0 = points.min(axis=0)
    y1, x1 = points.max(axis=0) + 1
    return int(x0), int(y0), int(x1), int(y1)


def bbox_iou(left, right):
    if left is None or right is None:
        return math.nan
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return float(intersection / union) if union else math.nan


def touches_edge(mask):
    return bool(
        mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
    )


def orientation(filename):
    tokens = re.split(r"[_\-.]", filename.casefold())
    mapping = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
    matches = [mapping[token] for token in tokens if token in mapping]
    return matches[0] if len(matches) == 1 else "unknown"


def boundary_statistics(image, reference):
    eroded = ndimage.binary_erosion(reference, structure=STRUCTURE_8)
    boundary = reference ^ eroded
    values = sobel(np.asarray(image, dtype=np.float32))[boundary]
    if not len(values):
        raise ValueError("Empty reference boundary")
    median = float(np.median(values))
    sharp_fraction = float(np.mean(values >= SHARP_EDGE_THRESHOLD))
    return {
        "boundary_pixels": len(values),
        "gradient_mean": float(np.mean(values)),
        "gradient_median": median,
        "gradient_max": float(np.max(values)),
        "gradient_p10": float(np.percentile(values, 10)),
        "sharp_pixel_fraction": sharp_fraction,
        "boundary_class": (
            "sharp"
            if median >= SHARP_EDGE_THRESHOLD and sharp_fraction >= 0.50
            else "diffuse"
        ),
    }


def size_group(area):
    if area < 100:
        return "small"
    if area <= 500:
        return "medium"
    return "large"


def location_group(reference_centroid_x, reference_centroid_y):
    centre = (MODEL_SIZE - 1) / 2
    distance = math.hypot(reference_centroid_x - centre, reference_centroid_y - centre)
    normalized = distance / math.hypot(centre, centre)
    return ("central" if normalized <= 0.35 else "peripheral"), normalized


def focality(component_count_ge4):
    if component_count_ge4 == 1:
        return "single_focal"
    if component_count_ge4 > 1:
        return "multi_focal"
    return "indeterminate"


def load_or_create_annotations(failures):
    if REVIEW_ANNOTATIONS.exists():
        rows = smoke.read_csv(REVIEW_ANNOTATIONS)
        lookup = {normalized_path(row["path"]): row for row in rows}
        expected = {normalized_path(row["path"]) for row in failures}
        if set(lookup) != expected:
            raise ValueError("Visual annotation membership mismatch")
        return lookup
    rows = []
    for index, row in enumerate(failures, start=1):
        rows.append(
            {
                "review_index": index,
                "path": row["path"],
                "edema_visual": "not_assessable_unverified_single_slice",
                "contrast_enhancement_visual": (
                    "not_assessable_unverified_sequence_and_contrast"
                ),
                "ventricle_proximity_visual": "indeterminate",
                "multifocal_visual": row["focality_from_reference_mask"],
                "visual_review_status": "PENDING",
                "visual_review_notes": "",
            }
        )
    write_csv(REVIEW_ANNOTATIONS, rows)
    return {normalized_path(row["path"]): row for row in rows}


def render_review_panels(failures, images, references, predictions):
    font = ImageFont.load_default()
    for page, start in enumerate(range(0, len(failures), 10), start=1):
        page_rows = failures[start : start + 10]
        tile_width, tile_height = 414, 182
        canvas = Image.new("RGB", (tile_width * 2, tile_height * 5), "#17191c")
        draw = ImageDraw.Draw(canvas)
        for local_index, row in enumerate(page_rows):
            column = local_index % 2
            line = local_index // 2
            x0, y0 = column * tile_width, line * tile_height
            key = normalized_path(row["path"])
            image = np.clip(images[key] * 255, 0, 255).astype(np.uint8)
            rgb = np.repeat(image[..., None], 3, axis=2)
            reference_overlay = rgb.copy()
            prediction_overlay = rgb.copy()
            reference_overlay[references[key]] = (
                0.42 * reference_overlay[references[key]]
                + 0.58 * np.array([235, 64, 52])
            ).astype(np.uint8)
            prediction_overlay[predictions[key]] = (
                0.42 * prediction_overlay[predictions[key]]
                + 0.58 * np.array([0, 210, 220])
            ).astype(np.uint8)
            panels = (rgb, reference_overlay, prediction_overlay)
            for panel_index, panel in enumerate(panels):
                picture = Image.fromarray(panel).resize((128, 128), Image.Resampling.NEAREST)
                canvas.paste(picture, (x0 + 4 + panel_index * 134, y0 + 35))
            title = (
                f"#{start + local_index + 1:02d} {row['filename']} "
                f"D={row['dice']:.3f} {row['orientation']}"
            )
            draw.text((x0 + 4, y0 + 4), title, fill="white", font=font)
            draw.text(
                (x0 + 4, y0 + 18),
                "input | reference red | prediction cyan",
                fill="#cfd3d7",
                font=font,
            )
        destination = Path(f"{REVIEW_PANEL_PREFIX}_{page:02d}.png")
        canvas.save(destination)


def match_pairs(cases):
    failures = [row for row in cases if row["dice"] < 0.50]
    successes = [row for row in cases if row["dice"] >= 0.80]
    candidates = []
    for failure in failures:
        for success in successes:
            ratio = success["reference_pixels"] / failure["reference_pixels"]
            if 0.80 <= ratio <= 1.20:
                candidates.append(
                    (
                        failure["orientation"] != success["orientation"],
                        abs(math.log(ratio)),
                        normalized_path(failure["path"]),
                        normalized_path(success["path"]),
                        failure,
                        success,
                    )
                )
    candidates.sort(key=lambda item: item[:4])
    pairs = []
    used_failures = set()
    used_successes = set()
    for mismatch, area_distance, _, _, failure, success in candidates:
        failure_key = normalized_path(failure["path"])
        success_key = normalized_path(success["path"])
        if failure_key in used_failures or success_key in used_successes:
            continue
        pairs.append(
            {
                "failure": failure,
                "success": success,
                "orientation_match": not mismatch,
                "absolute_log_area_ratio": area_distance,
            }
        )
        used_failures.add(failure_key)
        used_successes.add(success_key)
        if len(pairs) == 10:
            break
    if len(pairs) != 10:
        raise ValueError(f"Only {len(pairs)} valid matched pairs were found")
    return pairs


def render_case_study(pairs, images, references, predictions):
    figure, axes = plt.subplots(10, 4, figsize=(13, 25), constrained_layout=True)
    for pair_index, pair in enumerate(pairs, start=1):
        for side_index, key_name in enumerate(("failure", "success")):
            row = pair[key_name]
            key = normalized_path(row["path"])
            image = images[key]
            ref = references[key]
            pred = predictions[key]
            overlay = np.repeat(image[..., None], 3, axis=2)
            overlay[ref] = 0.42 * overlay[ref] + 0.58 * np.array([1.0, 0.18, 0.12])
            overlay[pred] = 0.42 * overlay[pred] + 0.58 * np.array([0.0, 0.90, 1.0])
            image_axis = axes[pair_index - 1, side_index * 2]
            overlay_axis = axes[pair_index - 1, side_index * 2 + 1]
            image_axis.imshow(image, cmap="gray", vmin=0, vmax=1)
            overlay_axis.imshow(np.clip(overlay, 0, 1))
            prefix = "Failure" if key_name == "failure" else "Success"
            image_axis.set_title(f"Pair {pair_index} {prefix}: {row['filename']}", fontsize=8)
            overlay_axis.set_title(
                f"Ref red / pred cyan — Dice {row['dice']:.3f}", fontsize=8
            )
            image_axis.axis("off")
            overlay_axis.axis("off")
    figure.savefig(OUTPUT_CASE_FIGURE, dpi=150)
    plt.close(figure)


def write_case_study(pairs):
    lines = [
        "# Matched glioma case study",
        "",
        "Ten failure/success pairs were selected without replacement before visual interpretation. Reference areas differ by at most 20%; identical orientation was preferred, followed by minimum absolute log area ratio and normalized-path tie-breaking. Red is reference and cyan is prediction.",
        "",
        f"![Matched glioma case pairs]({OUTPUT_CASE_FIGURE.name})",
        "",
    ]
    for index, pair in enumerate(pairs, start=1):
        failure = pair["failure"]
        success = pair["success"]
        difference = failure["centroid_distance"] - success["centroid_distance"]
        difference_text = (
            f"{abs(difference):.1f} pixels larger"
            if difference >= 0
            else f"{abs(difference):.1f} pixels smaller"
        )
        lines.extend(
            [
                f"## Pair {index}",
                "",
                (
                    f"The failure (`{failure['filename']}`, Dice {failure['dice']:.3f}) "
                    f"and success (`{success['filename']}`, Dice {success['dice']:.3f}) "
                    f"have reference areas of {failure['reference_pixels']} and "
                    f"{success['reference_pixels']} pixels. The failure prediction has "
                    f"{failure['prediction_component_count_raw']} components and a "
                    f"prediction/reference area ratio of {failure['area_ratio']:.2f}; the "
                    f"success has {success['prediction_component_count_raw']} components "
                    f"and ratio {success['area_ratio']:.2f}. Its centroid error is "
                    f"{difference_text} than the matched success. This pair supports a "
                    "localization/topology explanation only; it does not establish a "
                    "clinical imaging cause."
                ),
                "",
            ]
        )
    OUTPUT_CASE_STUDY.write_text("\n".join(lines), encoding="utf-8")


def bootstrap_mean(rows, rng, value_key="dice"):
    groups = {}
    for row in rows:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    estimates = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        values = [item[value_key] for key in sampled for item in groups[key]]
        estimates.append(float(np.mean(values)))
    return np.percentile(estimates, [2.5, 97.5]).tolist()


def bootstrap_risk_difference(rows, group_key, group_value, rng):
    groups = {}
    for row in rows:
        groups.setdefault(row["similarity_group"], []).append(row)
    keys = sorted(groups)
    estimates = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = rng.choice(keys, size=len(keys), replace=True)
        sample = [item for key in sampled for item in groups[key]]
        inside = [row["failure"] for row in sample if row[group_key] == group_value]
        outside = [row["failure"] for row in sample if row[group_key] != group_value]
        if inside and outside:
            estimates.append(float(np.mean(inside) - np.mean(outside)))
    if not estimates:
        return [math.nan, math.nan]
    return np.percentile(estimates, [2.5, 97.5]).tolist()


def odds_ratio(rows, group_key, group_value):
    a = sum(row[group_key] == group_value and row["failure"] for row in rows)
    b = sum(row[group_key] == group_value and not row["failure"] for row in rows)
    c = sum(row[group_key] != group_value and row["failure"] for row in rows)
    d = sum(row[group_key] != group_value and not row["failure"] for row in rows)
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def subgroup_rows(cases, key, values):
    output = []
    for value in values:
        members = [row for row in cases if row[key] == value]
        if not members:
            continue
        complement = [row for row in cases if row[key] != value]
        failed = [row for row in members if row["failure"]]
        succeeded = [row for row in members if not row["failure"]]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        mean_ci = bootstrap_mean(members, rng)
        failure_rate = float(np.mean([row["failure"] for row in members]))
        complement_rate = float(np.mean([row["failure"] for row in complement]))
        risk_difference = failure_rate - complement_rate
        risk_ci = bootstrap_risk_difference(
            cases, key, value, np.random.default_rng(BOOTSTRAP_SEED)
        )
        output.append(
            {
                "value": value,
                "n": len(members),
                "mean_dice": float(np.mean([row["dice"] for row in members])),
                "mean_ci": mean_ci,
                "failure_n": len(failed),
                "failure_mean_dice": (
                    float(np.mean([row["dice"] for row in failed]))
                    if failed
                    else math.nan
                ),
                "success_n": len(succeeded),
                "success_mean_dice": (
                    float(np.mean([row["dice"] for row in succeeded]))
                    if succeeded
                    else math.nan
                ),
                "failure_rate": failure_rate,
                "risk_difference": risk_difference,
                "risk_ci": risk_ci,
                "odds_ratio": odds_ratio(cases, key, value),
                "associated": risk_ci[0] > 0 or risk_ci[1] < 0,
            }
        )
    return output


def write_subgroup_analysis(cases):
    analyses = [
        ("Reference size", "size_group", ("small", "medium", "large")),
        ("Boundary class", "boundary_class", ("sharp", "diffuse")),
        (
            "Reference focality",
            "focality_from_reference_mask",
            ("single_focal", "multi_focal", "indeterminate"),
        ),
        ("Reference location", "location_group", ("central", "peripheral")),
    ]
    lines = [
        "# Glioma subgroup analysis",
        "",
        "This analysis covers all 204 frozen BRISC glioma validation cases. Failure is Dice below 0.50. Confidence intervals use 2,000 similarity-group bootstrap resamples with seed 42. The success/failure mean-Dice columns are descriptive because those labels are defined by Dice itself. Association is judged by the failure-rate risk-difference CI against the complementary group.",
        "",
    ]
    associated = []
    for title, key, values in analyses:
        results = subgroup_rows(cases, key, values)
        lines.extend(
            [
                f"## {title}",
                "",
                "| Group | n | Mean Dice (95% CI) | Failures | Failure mean | Success mean | Failure rate | Risk difference vs others (95% CI) | Odds ratio | Associated? |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in results:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["value"],
                        str(row["n"]),
                        f"{row['mean_dice']:.3f} ({row['mean_ci'][0]:.3f}, {row['mean_ci'][1]:.3f})",
                        f"{row['failure_n']}/{row['n']}",
                        (
                            f"{row['failure_mean_dice']:.3f}"
                            if math.isfinite(row["failure_mean_dice"])
                            else "NA"
                        ),
                        (
                            f"{row['success_mean_dice']:.3f}"
                            if math.isfinite(row["success_mean_dice"])
                            else "NA"
                        ),
                        f"{row['failure_rate']:.3f}",
                        f"{row['risk_difference']:+.3f} ({row['risk_ci'][0]:+.3f}, {row['risk_ci'][1]:+.3f})",
                        f"{row['odds_ratio']:.2f}",
                        "yes" if row["associated"] else "no",
                    ]
                )
                + " |"
            )
            if row["associated"]:
                direction = "higher" if row["risk_difference"] > 0 else "lower"
                associated.append(f"{title.lower()}={row['value']} ({direction} failure risk)")
        lines.append("")
    lines.extend(
        [
            "## Evidence statement",
            "",
            (
                "Characteristics with group-bootstrap risk-difference intervals excluding zero: "
                + "; ".join(associated)
                + "."
                if associated
                else "No predeclared characteristic has a group-bootstrap risk-difference interval that excludes zero."
            ),
            "",
            "These are internal associations on a reused development validation set. They do not prove biological causes, clinical subgroup performance, or patient-independent generalization.",
        ]
    )
    OUTPUT_SUBGROUP.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize-visual-review",
        action="store_true",
        help="Require every visual review annotation to be marked REVIEWED.",
    )
    args = parser.parse_args()

    if visual.sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("Frozen v1 checkpoint SHA256 mismatch")
    split = RUN / "split_snapshot.csv"
    if smoke.digest(split) != EXPECTED_SPLIT_SHA256:
        raise ValueError("Frozen split SHA256 mismatch")
    rows = smoke.read_csv(split)
    parts = segmenter.partition_masked_rows(rows)
    validation_rows = parts["validation"]
    if any(row["original_split"] != "train" for row in validation_rows):
        raise ValueError("Original test data entered the development analysis")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location="cpu", weights_only=True)["model_state"]
    )
    metrics, records, predictions, references, images = infer_all(
        model, validation_rows, device
    )
    prior = {
        normalized_path(row["path"]): row
        for row in smoke.read_csv(RUN / "validation_predictions.csv")
    }
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_metric_difference = 0.0
    for record in records:
        old = prior[normalized_path(record["path"])]
        for metric in ("dice", "iou"):
            difference = abs(record[metric] - float(old[metric]))
            maximum_metric_difference = max(maximum_metric_difference, difference)
            if difference > tolerance:
                raise ValueError(f"Stored {metric} mismatch: {record['path']}")
        if (
            record["reference_pixels"] != int(old["reference_pixels"])
            or record["predicted_pixels"] != int(old["predicted_pixels"])
        ):
            raise ValueError(f"Stored area mismatch: {record['path']}")

    source_lookup = {normalized_path(row["path"]): row for row in validation_rows}
    glioma_records = [row for row in records if row["label"] == "glioma"]
    if len(records) != 680 or len(glioma_records) != 204:
        raise ValueError("Validation or glioma case count changed")
    analyzed = []
    sharpness_rows = []
    for record in sorted(glioma_records, key=lambda row: normalized_path(row["path"])):
        key = normalized_path(record["path"])
        source = source_lookup[key]
        reference = references[key]
        prediction = predictions[key]
        overlap = int((reference & prediction).sum())
        ref_raw, ref_ge4, _ = component_summary(reference)
        pred_raw, pred_ge4, largest_fraction = component_summary(prediction)
        ref_x, ref_y = centroid(reference)
        pred_x, pred_y = centroid(prediction)
        centroid_distance = (
            math.hypot(ref_x - pred_x, ref_y - pred_y)
            if math.isfinite(pred_x)
            else math.nan
        )
        ref_bbox = bbox(reference)
        pred_bbox = bbox(prediction)
        location, normalized_location = location_group(ref_x, ref_y)
        boundary = boundary_statistics(images[key], reference)
        focality_value = focality(ref_ge4)
        case = {
            **record,
            "mask_path": source["mask_path"],
            "similarity_group": source["similarity_group"],
            "original_width": int(source["width"]),
            "original_height": int(source["height"]),
            "model_width": MODEL_SIZE,
            "model_height": MODEL_SIZE,
            "orientation": orientation(record["filename"]),
            "area_ratio": record["predicted_pixels"] / record["reference_pixels"],
            "overlap_pixels": overlap,
            "reference_component_count_raw": ref_raw,
            "reference_component_count_ge4": ref_ge4,
            "prediction_component_count_raw": pred_raw,
            "prediction_component_count_ge4": pred_ge4,
            "prediction_largest_component_fraction": largest_fraction,
            "reference_centroid_x": ref_x,
            "reference_centroid_y": ref_y,
            "prediction_centroid_x": pred_x,
            "prediction_centroid_y": pred_y,
            "centroid_distance": centroid_distance,
            "reference_bbox_width": ref_bbox[2] - ref_bbox[0],
            "reference_bbox_height": ref_bbox[3] - ref_bbox[1],
            "prediction_bbox_width": (
                pred_bbox[2] - pred_bbox[0] if pred_bbox is not None else math.nan
            ),
            "prediction_bbox_height": (
                pred_bbox[3] - pred_bbox[1] if pred_bbox is not None else math.nan
            ),
            "bbox_overlap_iou": bbox_iou(ref_bbox, pred_bbox),
            "reference_touches_image_edge": touches_edge(reference),
            "prediction_touches_image_edge": touches_edge(prediction),
            "focality_from_reference_mask": focality_value,
            "size_group": size_group(record["reference_pixels"]),
            "location_group": location,
            "normalized_reference_centroid_distance": normalized_location,
            "failure": record["dice"] < 0.50,
            **boundary,
        }
        analyzed.append(case)
        sharpness_rows.append(
            {
                "path": record["path"],
                "filename": record["filename"],
                "similarity_group": source["similarity_group"],
                "dice": record["dice"],
                "failure": record["dice"] < 0.50,
                "reference_pixels": record["reference_pixels"],
                "orientation": case["orientation"],
                **boundary,
            }
        )

    failures = [row for row in analyzed if row["failure"]]
    if len(failures) != 60:
        raise ValueError(f"Expected 60 glioma failures, found {len(failures)}")
    annotations = load_or_create_annotations(failures)
    if args.finalize_visual_review and any(
        row["visual_review_status"] != "REVIEWED" for row in annotations.values()
    ):
        raise ValueError("Visual review is not complete for every glioma failure")
    decomposition = []
    for row in failures:
        annotation = annotations[normalized_path(row["path"])]
        decomposition.append(
            {
                **{key: value for key, value in row.items() if key != "failure"},
                "edema_visual": annotation["edema_visual"],
                "contrast_enhancement_visual": annotation[
                    "contrast_enhancement_visual"
                ],
                "ventricle_proximity_visual": annotation[
                    "ventricle_proximity_visual"
                ],
                "multifocal_visual": annotation["multifocal_visual"],
                "visual_review_status": annotation["visual_review_status"],
                "visual_review_notes": annotation["visual_review_notes"],
            }
        )

    write_csv(OUTPUT_DECOMPOSITION, decomposition)
    write_csv(OUTPUT_SHARPNESS, sharpness_rows)
    render_review_panels(failures, images, references, predictions)
    pairs = match_pairs(analyzed)
    render_case_study(pairs, images, references, predictions)
    write_case_study(pairs)
    write_subgroup_analysis(analyzed)
    save_json(
        OUTPUT_VERIFICATION,
        {
            "status": (
                "COMPLETE" if args.finalize_visual_review else "AWAITING_VISUAL_REVIEW"
            ),
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256.lower(),
            "split_sha256": EXPECTED_SPLIT_SHA256,
            "device": str(device),
            "threshold": THRESHOLD,
            "validation_images_recomputed": len(records),
            "glioma_images_analyzed": len(analyzed),
            "glioma_failures_analyzed": len(failures),
            "maximum_stored_metric_difference": maximum_metric_difference,
            "recomputed_glioma_mean_dice": metrics["val_dice_glioma"],
            "visual_review_complete": args.finalize_visual_review,
            "test_images_accessed": 0,
        },
    )
    print(OUTPUT_VERIFICATION.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
