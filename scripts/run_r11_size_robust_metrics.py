"""Run Amendment 21 R11 size-robust glioma comparison."""

from __future__ import annotations

import csv
import hashlib
import json
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import torch
import train_segmenter as train
from scipy.ndimage import binary_erosion, distance_transform_edt, label
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/r11_size_robust_metrics"
ART = EXP / "artifacts"
FINAL = ROOT / "runs/final_stack"
E08 = ROOT / "experiments/e08_ensemble"
BASELINE = ROOT / "runs/segmentation/20260923-072439-397263-full"


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def normalized(value: str) -> str:
    return value.replace("\\", "/").casefold()


def overlap_scores(prediction: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(prediction & reference))
    denominator = int(prediction.sum() + reference.sum())
    union = int(np.count_nonzero(prediction | reference))
    # Match the registered PyTorch float32 accumulator exactly.
    return float(np.float32(2 * intersection) / np.float32(denominator)), float(np.float32(intersection) / np.float32(union))


def surfaces(mask: np.ndarray) -> np.ndarray:
    return mask ^ binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def surface_metrics(prediction: np.ndarray, reference: np.ndarray, tolerance: float, diagonal: float) -> tuple[float, float]:
    ps, rs = surfaces(prediction), surfaces(reference)
    if not rs.any():
        raise ValueError("Empty glioma reference")
    if not ps.any():
        return 0.0, diagonal
    distance_to_ref = distance_transform_edt(~rs)
    distance_to_pred = distance_transform_edt(~ps)
    pred_dist = distance_to_ref[ps]
    ref_dist = distance_to_pred[rs]
    surface_dice = (np.count_nonzero(pred_dist <= tolerance) + np.count_nonzero(ref_dist <= tolerance)) / (len(pred_dist) + len(ref_dist))
    hd95 = float(np.percentile(np.concatenate([pred_dist, ref_dist]), 95))
    return float(surface_dice), hd95


def lesion_counts(prediction: np.ndarray, reference: np.ndarray, threshold: float) -> tuple[int, int, int]:
    structure = np.ones((3, 3), dtype=np.uint8)
    pred_labels, pred_count = label(prediction, structure=structure)
    ref_labels, ref_count = label(reference, structure=structure)
    if ref_count == 0:
        raise ValueError("Empty glioma reference")
    if pred_count == 0:
        return ref_count, 0, 0
    scores = np.zeros((ref_count, pred_count), dtype=np.float64)
    for i in range(ref_count):
        ref = ref_labels == i + 1
        for j in range(pred_count):
            pred = pred_labels == j + 1
            denominator = ref.sum() + pred.sum()
            scores[i, j] = 2 * np.count_nonzero(ref & pred) / denominator
    # Stable infinitesimal tie-break prefers lower row/column indices.
    tie = (np.arange(ref_count)[:, None] * pred_count + np.arange(pred_count)[None, :]) * 1e-15
    rows, cols = linear_sum_assignment(-(scores - tie))
    detected = int(np.count_nonzero(scores[rows, cols] >= threshold))
    return ref_count, detected, pred_count


def bootstrap(rows: list[dict], repetitions: int, seed: int) -> dict:
    groups = sorted({row["similarity_group"] for row in rows})
    by_group = {group: [row for row in rows if row["similarity_group"] == group] for group in groups}
    rng = np.random.default_rng(seed)
    metrics = {name: [] for name in ("dice", "surface_dice", "hd95", "lesion_detection")}
    for _ in range(repetitions):
        sample = [row for group in rng.choice(groups, len(groups), replace=True) for row in by_group[group]]
        metrics["dice"].append(float(np.mean([row["e08_dice"] - row["v1_dice"] for row in sample])))
        metrics["surface_dice"].append(float(np.mean([row["e08_surface_dice_2px"] - row["v1_surface_dice_2px"] for row in sample])))
        metrics["hd95"].append(float(np.mean([row["v1_hd95"] - row["e08_hd95"] for row in sample])))
        e08_rate = sum(row["e08_detected_lesions"] for row in sample) / sum(row["reference_lesions"] for row in sample)
        v1_rate = sum(row["v1_detected_lesions"] for row in sample) / sum(row["reference_lesions"] for row in sample)
        metrics["lesion_detection"].append(float(e08_rate - v1_rate))
    return {name: {"ci95_lower": float(np.percentile(values, 2.5)), "ci95_upper": float(np.percentile(values, 97.5))} for name, values in metrics.items()}


def main() -> None:
    config = read_json(EXP / "config.json")
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise RuntimeError("R11 is not registered-not-started")
    if sha256(ROOT / config["amendment_path"]) != config["amendment_sha256"]:
        raise RuntimeError("Amendment 21 hash mismatch")
    sources = {
        "final_stack_manifest": FINAL / "manifest.json",
        "e08_per_case_comparison": E08 / "artifacts/per_case_comparison.csv",
        "e08_experiment_manifest": E08 / "artifacts/experiment_manifest.json",
        "split": BASELINE / "split_snapshot.csv",
    }
    for key, path in sources.items():
        if sha256(path) != config["source_hashes"][key]:
            raise RuntimeError(f"R11 source hash mismatch: {key}")
    final_tree_before = sha256(FINAL / "manifest.json")
    source_rows = {normalized(row["path"]): row for row in read_csv(sources["e08_per_case_comparison"])}
    validation = train.partition_masked_rows(smoke.read_csv(sources["split"]))["validation"]
    glioma = [row for row in validation if row["label"] == "glioma"]
    if len(glioma) != 204 or len(source_rows) != 680:
        raise RuntimeError("R11 population mismatch")
    latest = Path((ROOT / "runs/segmentation_v2_ensemble/latest.txt").read_text(encoding="utf-8").strip())
    cache_manifest = read_json(latest / "manifest.json")
    if cache_manifest["test_images_accessed"] != 0:
        raise RuntimeError("E08 cache test-access mismatch")
    order = read_csv(latest / "validation_order.csv")
    with np.load(latest / "ensemble_probabilities_targets.npz") as archive:
        e08_probability = archive["probabilities"]
        targets = archive["targets"].astype(bool)
    order_index = {normalized(row["path"]): i for i, row in enumerate(order)}
    if set(order_index) != {normalized(row["path"]) for row in validation}:
        raise RuntimeError("E08 cache membership mismatch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = FINAL / "models/segmenter_v1_fallback.pt"
    manifest = read_json(FINAL / "manifest.json")
    artifact = next(item for item in manifest["artifacts"] if item["path"] == "models/segmenter_v1_fallback.pt")
    if sha256(checkpoint) != artifact["sha256"]:
        raise RuntimeError("Frozen v1 checkpoint hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source_result = read_json(BASELINE / "result.json")
    source_config = read_json(BASELINE / "config.json")
    if state["epoch"] != source_result["best_epoch"] or state["config"] != source_config:
        raise RuntimeError("Frozen v1 checkpoint provenance mismatch")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    model.load_state_dict(state["model_state"]); model.eval()
    # Preserve the v1/E08 audit batch geometry because fp16 kernels can be
    # batch-shape sensitive at the 0.50 hard-mask boundary.
    loader = DataLoader(train.FullPairs(glioma), batch_size=8, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    v1_probability = np.empty((204, 128, 128), dtype=np.float32)
    seen_targets = np.empty_like(v1_probability, dtype=bool)
    with torch.inference_mode():
        for images, batch_targets, _labels, indices in loader:
            images = images.to(device, non_blocking=True)
            context = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                values = model(images).float().sigmoid()[:, 0].cpu().numpy()
            v1_probability[indices.numpy()] = values
            seen_targets[indices.numpy()] = batch_targets[:, 0].numpy() >= 0.5

    detailed = []
    diagonal = float(config["metrics"]["empty_prediction_hd95"])
    for i, row in enumerate(glioma):
        key = normalized(row["path"]); cache_i = order_index[key]
        reference = targets[cache_i]
        if not np.array_equal(reference, seen_targets[i]):
            raise RuntimeError(f"R11 target mismatch: {row['path']}")
        current = {"path": row["path"], "similarity_group": row["similarity_group"]}
        current["reference_pixels"] = int(reference.sum())
        ref_lesions = None
        for name, probability in (("v1", v1_probability[i]), ("e08", e08_probability[cache_i])):
            hard = probability >= 0.5
            dice, iou = overlap_scores(hard, reference)
            surface_dice, hd95 = surface_metrics(hard, reference, 2.0, diagonal)
            lesions, detected, predicted_lesions = lesion_counts(hard, reference, 0.5)
            ref_lesions = lesions if ref_lesions is None else ref_lesions
            if lesions != ref_lesions:
                raise RuntimeError("Reference component count changed")
            current.update({f"{name}_dice": dice, f"{name}_iou": iou, f"{name}_surface_dice_2px": surface_dice, f"{name}_hd95": hd95, f"{name}_predicted_pixels": int(hard.sum()), f"{name}_predicted_lesions": predicted_lesions, f"{name}_detected_lesions": detected})
        current["reference_lesions"] = ref_lesions
        source = source_rows[key]
        current["v1_source_dice"] = float(source["baseline_dice"])
        current["e08_source_dice"] = float(source["ensemble_dice"])
        detailed.append(current)
    maximum_difference = max(max(abs(row["v1_dice"] - row["v1_source_dice"]), abs(row["e08_dice"] - row["e08_source_dice"])) for row in detailed)
    if maximum_difference > 1e-12:
        raise RuntimeError(f"R11 ordinary Dice reproduction mismatch: {maximum_difference}")

    def summary(prefix: str) -> dict:
        ref = sum(row["reference_lesions"] for row in detailed)
        return {"mean_dice": float(np.mean([row[f"{prefix}_dice"] for row in detailed])), "mean_surface_dice_2px": float(np.mean([row[f"{prefix}_surface_dice_2px"] for row in detailed])), "mean_hd95_pixels": float(np.mean([row[f"{prefix}_hd95"] for row in detailed])), "lesions_detected": int(sum(row[f"{prefix}_detected_lesions"] for row in detailed)), "reference_lesions": int(ref), "lesion_detection_rate": float(sum(row[f"{prefix}_detected_lesions"] for row in detailed) / ref)}
    v1, e08 = summary("v1"), summary("e08")
    differences = {"ordinary_dice_e08_minus_v1": e08["mean_dice"] - v1["mean_dice"], "surface_dice_e08_minus_v1": e08["mean_surface_dice_2px"] - v1["mean_surface_dice_2px"], "hd95_v1_minus_e08": v1["mean_hd95_pixels"] - e08["mean_hd95_pixels"], "lesion_detection_e08_minus_v1": e08["lesion_detection_rate"] - v1["lesion_detection_rate"]}
    ci = bootstrap(detailed, 2000, 42)
    passed = differences["surface_dice_e08_minus_v1"] > 0
    write_csv(ART / "per_case_size_robust_metrics.csv", detailed)
    bootstrap_rows = [{"metric": name, **value} for name, value in ci.items()]
    write_csv(ART / "paired_bootstrap_intervals.csv", bootstrap_rows)
    result = {"schema_version": 1, "experiment_id": "r11_size_robust_metrics", "status": "PASS" if passed else "FAIL", "decision": "PASS" if passed else "FAIL", "exploratory_post_review": True, "population": {"cases": 204, "category": "glioma", "similarity_groups": len({row['similarity_group'] for row in detailed})}, "v1": v1, "e08": e08, "differences": differences, "paired_similarity_group_bootstrap_ci95": ci, "gate": {**config["gate"], "observed": differences["surface_dice_e08_minus_v1"], "status": "PASS" if passed else "FAIL"}, "verification": {"ordinary_dice_max_abs_difference_from_source": maximum_difference, "tolerance": 1e-12, "source_hashes_verified": True, "final_stack_manifest_unchanged": final_tree_before == sha256(FINAL / "manifest.json"), "test_images_accessed": 0, "neural_retraining": 0, "threshold_changes": 0, "segmentation_threshold": 0.5}}
    write_json(EXP / "result.json", result)
    config["status"] = "COMPLETE"; write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — R11 size-robust metrics\n\n**Status:** {result['status']}\n\nE08 mean 2-pixel surface Dice was `{e08['mean_surface_dice_2px']:.6f}` versus v1 `{v1['mean_surface_dice_2px']:.6f}` (difference `{differences['surface_dice_e08_minus_v1']:+.6f}`, paired group-bootstrap 95% CI `{ci['surface_dice']['ci95_lower']:.6f}` to `{ci['surface_dice']['ci95_upper']:.6f}`). The preregistered strict directional gate therefore **{result['status']}**. This exploratory internal result does not establish causality, patient independence, external transport, or clinical validity.\n", encoding="utf-8", newline="\n")
    registry_path = ROOT / "experiments/registry.json"; registry = read_json(registry_path); next(item for item in registry["experiments"] if item["id"] == EXP.name)["status"] = "COMPLETE"; write_json(registry_path, registry)
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "per_case_size_robust_metrics.csv", ART / "paired_bootstrap_intervals.csv"]
    write_json(ART / "experiment_manifest.json", {"schema_version": 1, "experiment_id": EXP.name, "files": [{"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in paths]})
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Complete R11 size-robust metrics\n\n- **Decision made:** Record H10/R11 as `{result['status']}` without changing the gate.\n- **Triggering artifact:** `experiments/r11_size_robust_metrics/result.json`; E08-minus-v1 surface Dice `{differences['surface_dice_e08_minus_v1']:+.6f}`.\n- **Rationale:** The fixed 204-case glioma comparison was recomputed under Amendment 21.\n- **Verification:** Source hashes and ordinary Dice reproduction passed; zero sealed-test access, retraining, or threshold changes.\n- **Who approved:** User authorized completion of all Amendment 21 items.\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
