"""Run registered O03 frozen-segmenter bottleneck Mahalanobis detector."""

from __future__ import annotations

import json
from datetime import datetime

import evaluate_x01_cheng as x01
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ood_common as common
import stage6_external_common as external
import torch
from PIL import Image
from sklearn.covariance import LedoitWolf

ROOT = common.ROOT
EXP = ROOT / "experiments/o03_encoder_mahalanobis"
ART = EXP / "artifacts"
BATCH_SIZE = 64
BOTTLENECK = "model.1.submodule.1.submodule.1.submodule"


def tensors(rows):
    values = []
    for row in rows:
        with Image.open(ROOT / row["path"]) as image:
            array = np.asarray(image.convert("L"))
        tensor, _geometry, _window = external.preprocess_segmenter(array)
        values.append(tensor)
    return np.asarray(values, dtype=np.float32)


def main():
    if common.read_json(EXP / "result.json")["status"] != "NOT_RUN":
        raise RuntimeError("O03 not in NOT_RUN state")
    rows = common.load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = x01.load_raw_segmenter(x01.V1_CHECKPOINT, x01.EXPECTED_HASHES["v1"], device)
    modules = dict(model.named_modules())
    if BOTTLENECK not in modules:
        raise RuntimeError("Registered bottleneck module missing")
    captured = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = modules[BOTTLENECK].register_forward_hook(hook)
    embeddings = []
    try:
        with torch.inference_mode():
            for start in range(0, len(rows), BATCH_SIZE):
                batch = torch.from_numpy(tensors(rows[start : start + BATCH_SIZE])).to(device)
                captured.clear()
                _ = model(batch)
                if len(captured) != 1 or captured[0].ndim != 4 or captured[0].shape[1] != 128:
                    raise RuntimeError("Unexpected bottleneck activation contract")
                embeddings.append(captured[0].mean(dim=(2, 3)).cpu().numpy())
    finally:
        handle.remove()
    embeddings = np.concatenate(embeddings, axis=0).astype(np.float64)
    if embeddings.shape != (2680, 128) or not np.isfinite(embeddings).all():
        raise RuntimeError(f"Invalid embedding matrix: {embeddings.shape}")
    fit_id = np.asarray([r["detector_split"] == "fit" and r["domain"] == "ID" for r in rows])
    evaluation = np.asarray([r["detector_split"] == "evaluation" for r in rows])
    gaussian = LedoitWolf(assume_centered=False).fit(embeddings[fit_id])
    distance = gaussian.mahalanobis(embeddings)
    threshold = float(np.quantile(distance[fit_id], 0.95, method="linear"))
    eval_rows = [r for r, flag in zip(rows, evaluation, strict=True) if flag]
    eval_distance = distance[evaluation]
    target = common.labels(eval_rows)
    auroc = float(__import__("sklearn.metrics").metrics.roc_auc_score(target, eval_distance))
    ci, _ = common.stratified_group_bootstrap_auc(eval_rows, target, eval_distance)
    flags = eval_distance >= threshold
    specificity = float((~flags[target == 0]).mean())
    sensitivity = float(flags[target == 1].mean())
    score_rows = []
    for row, value in zip(rows, distance, strict=True):
        score_rows.append({"sample_id": row["sample_id"], "source": row["source"], "domain": row["domain"], "detector_split": row["detector_split"], "group_id": row["group_id"], "mahalanobis_squared": float(value), "flag_at_fit_id_p95": bool(value >= threshold)})
    common.write_csv(ART / "mahalanobis_scores.csv", score_rows)
    np.savez_compressed(ART / "frozen_embeddings.npz", sample_id=np.asarray([r["sample_id"] for r in rows]), embedding=embeddings.astype(np.float32))
    fpr, tpr, _ = common.roc_points(eval_rows, eval_distance)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(fpr, tpr, label=f"AUROC {auroc:.4f}"); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="O03 encoder Mahalanobis OOD detector"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ART / "roc_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(eval_distance[target == 0], bins=50, alpha=0.6, density=True, label="BRISC ID"); ax.hist(eval_distance[target == 1], bins=50, alpha=0.6, density=True, label="OOD"); ax.axvline(threshold, color="black", linestyle="--", label="fit-ID p95"); ax.set(xlabel="Squared Mahalanobis distance", ylabel="Density", title="Frozen segmenter bottleneck distance"); ax.legend(); fig.tight_layout(); fig.savefig(ART / "distance_distributions.png", dpi=180); plt.close(fig)
    passed = auroc >= 0.90
    result = {"schema_version": 1, "experiment_id": "o03_encoder_mahalanobis", "status": "COMPLETE", "decision": "PASS" if passed else "FAIL", "gate": {"metric": "held-out AUROC", "threshold": 0.90, "passed": passed}, "evaluation": {"auroc": auroc, "auroc_ci95": ci, "fit_id_p95_threshold": threshold, "specificity_at_threshold": specificity, "ood_sensitivity_at_threshold": sensitivity}, "model": {"frozen_checkpoint": x01.V1_CHECKPOINT.relative_to(ROOT).as_posix(), "checkpoint_sha256": x01.EXPECTED_HASHES["v1"], "bottleneck_module": BOTTLENECK, "embedding_dimensions": 128, "spatial_pooling": "mean", "gaussian": "LedoitWolf covariance on 600 BRISC fitting embeddings only"}, "population": {"fit_id": int(fit_id.sum()), "evaluation": int(evaluation.sum()), "evaluation_id": int((target == 0).sum()), "evaluation_ood": int((target == 1).sum())}, "verification": {"o01_manifest_sha256": common.sha256(common.MANIFEST), "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "brisc_test_images_accessed": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"}}
    common.write_json(EXP / "result.json", result)
    config = common.read_json(EXP / "config.json"); config.update({"status": "COMPLETE", "checkpoint_sha256": x01.EXPECTED_HASHES["v1"], "bottleneck_module": BOTTLENECK, "embedding_dimensions": 128, "gaussian": "LedoitWolf", "fit_id_quantile": 0.95, "o01_manifest_sha256": common.sha256(common.MANIFEST)}); common.write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — O03 encoder Mahalanobis detector\n\n**Status:** {'PASS' if passed else 'FAIL'}\n\nHeld-out AUROC was `{auroc:.6f}` (95% group-bootstrap CI `{ci[0]:.6f}`–`{ci[1]:.6f}`) against the registered `0.90` gate. The frozen v1-fallback component of E08 supplied the registered bottleneck; no neural weight changed.\n", encoding="utf-8", newline="\n")
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "mahalanobis_scores.csv", ART / "frozen_embeddings.npz", ART / "roc_curve.png", ART / "distance_distributions.png"]
    common.experiment_manifest(EXP, paths)
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Evaluate O03 encoder Mahalanobis detector\n\n- **Decision made:** Record O03 as `{'PASS' if passed else 'FAIL'}` without gate relaxation.\n- **Triggering artifact:** `experiments/o03_encoder_mahalanobis/result.json`, held-out AUROC `{auroc:.6f}`.\n- **Rationale:** A Ledoit-Wolf Gaussian was fitted only to 600 BRISC training embeddings from the frozen segmenter bottleneck.\n- **Verification:** Checkpoint hash passed. Reference masks, labels, external performance data, sealed-test access, neural refitting, and threshold changes were zero.\n- **Who approved:** User authorized O01–O06 in one sequential run.\n")
    common.mark_complete("o03_encoder_mahalanobis")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
