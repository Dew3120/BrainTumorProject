"""Run registered O04 frozen classifier/segmenter confidence OOD detectors."""

from __future__ import annotations

import json
from contextlib import nullcontext
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
from sklearn.metrics import roc_auc_score

ROOT = common.ROOT
EXP = ROOT / "experiments/o04_model_confidence_ood"
ART = EXP / "artifacts"
BATCH_SIZE = 32
EPS = 1e-7


def preprocess(rows):
    segmenter = []
    classifier = []
    for row in rows:
        with Image.open(ROOT / row["path"]) as image:
            array = np.asarray(image.convert("L"))
        seg, _g, _w = external.preprocess_segmenter(array)
        cls, _g2, _w2 = external.preprocess_classifier(array)
        segmenter.append(seg); classifier.append(cls)
    return np.asarray(segmenter, dtype=np.float32), np.asarray(classifier, dtype=np.float32)


def main():
    if common.read_json(EXP / "result.json")["status"] != "NOT_RUN":
        raise RuntimeError("O04 not in NOT_RUN state")
    rows = common.load_manifest()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, models = x01.load_models(device)
    class_score = np.empty(len(rows), dtype=np.float64)
    segmenter_score = np.empty(len(rows), dtype=np.float64)
    routes = []
    with torch.inference_mode():
        for start in range(0, len(rows), BATCH_SIZE):
            batch_rows = rows[start : start + BATCH_SIZE]
            seg_np, cls_np = preprocess(batch_rows)
            seg = torch.from_numpy(seg_np).to(device)
            cls = torch.from_numpy(cls_np).to(device)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
            with context:
                logits = classifier(cls)
                e05 = models["e05"](seg)
                e06 = models["e06"](seg)
                specialist = models["e07_specialist"](seg)
                fallback = models["v1_fallback"](seg)
            probabilities = logits.float().softmax(dim=1)
            predicted = probabilities.argmax(dim=1)
            route_mask = predicted == x01.LABEL_INDEX["glioma"]
            routed = fallback.float().sigmoid()
            routed[route_mask] = specialist.float().sigmoid()[route_mask]
            ensemble = (e05.float().sigmoid() + e06.float().sigmoid() + routed) / 3.0
            p = ensemble[:, 0].clamp(EPS, 1 - EPS)
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)).mean(dim=(1, 2))
            stop = start + len(batch_rows)
            class_score[start:stop] = (1 - probabilities.max(dim=1).values).cpu().numpy()
            segmenter_score[start:stop] = entropy.cpu().numpy()
            routes.extend(["glioma_specialist" if value else "v1_fallback" for value in route_mask.cpu().tolist()])
    if not np.isfinite(class_score).all() or not np.isfinite(segmenter_score).all():
        raise RuntimeError("Non-finite O04 score")
    evaluation = np.asarray([r["detector_split"] == "evaluation" for r in rows])
    eval_rows = [r for r, flag in zip(rows, evaluation, strict=True) if flag]
    target = common.labels(eval_rows)
    reports = {}
    for name, all_scores in (("classifier_1_minus_max_softmax", class_score), ("segmenter_mean_entropy", segmenter_score)):
        values = all_scores[evaluation]
        auc = float(roc_auc_score(target, values)); ci, _ = common.stratified_group_bootstrap_auc(eval_rows, target, values)
        reports[name] = {"auroc": auc, "auroc_ci95": ci, "gate_threshold": 0.85, "passed": auc >= 0.85}
    score_rows = [{"sample_id": row["sample_id"], "source": row["source"], "domain": row["domain"], "detector_split": row["detector_split"], "group_id": row["group_id"], "classifier_ood_score": float(class_score[i]), "segmenter_entropy_ood_score": float(segmenter_score[i]), "e07_route": routes[i]} for i, row in enumerate(rows)]
    common.write_csv(ART / "model_confidence_scores.csv", score_rows)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, scores, color in (("Classifier", class_score[evaluation], "tab:blue"), ("Segmenter", segmenter_score[evaluation], "tab:orange")):
        fpr, tpr, _ = common.roc_points(eval_rows, scores); ax.plot(fpr, tpr, color=color, label=f"{label} AUROC {roc_auc_score(target, scores):.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="O04 frozen-model OOD confidence"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ART / "roc_curves.png", dpi=180); plt.close(fig)
    passed = all(report["passed"] for report in reports.values())
    result = {"schema_version": 1, "experiment_id": "o04_model_confidence_ood", "status": "COMPLETE", "decision": "PASS" if passed else "FAIL", "gate": {"rule": "both detectors AUROC >= 0.85", "passed": passed}, "detectors": reports, "population": {"all_scores": len(rows), "evaluation": len(eval_rows), "evaluation_id": int((target == 0).sum()), "evaluation_ood": int((target == 1).sum())}, "model": {"classifier_checkpoint_sha256": x01.EXPECTED_HASHES["classifier"], "segmenter_components": {key: x01.EXPECTED_HASHES[key] for key in ("e05", "e06", "e07", "v1")}, "segmenter_score_probability": "raw E08 arithmetic-mean foreground probability; C03 not applied"}, "verification": {"o01_manifest_sha256": common.sha256(common.MANIFEST), "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "brisc_test_images_accessed": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"}}
    common.write_json(EXP / "result.json", result)
    config = common.read_json(EXP / "config.json"); config.update({"status": "COMPLETE", "classifier_score": "1 - max softmax", "segmenter_score": "whole-image mean Bernoulli entropy of raw E08 probability", "o01_manifest_sha256": common.sha256(common.MANIFEST)}); common.write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — O04 model-confidence OOD detectors\n\n**Status:** {'PASS' if passed else 'FAIL'}\n\nClassifier AUROC: `{reports['classifier_1_minus_max_softmax']['auroc']:.6f}`. Segmenter entropy AUROC: `{reports['segmenter_mean_entropy']['auroc']:.6f}`. Each was tested against the registered `0.85` gate; neither gate was relaxed.\n", encoding="utf-8", newline="\n")
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "model_confidence_scores.csv", ART / "roc_curves.png"]
    common.experiment_manifest(EXP, paths)
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Evaluate O04 frozen-model confidence detectors\n\n- **Decision made:** Record O04 as `{'PASS' if passed else 'FAIL'}`; classifier AUROC `{reports['classifier_1_minus_max_softmax']['auroc']:.6f}`, segmenter AUROC `{reports['segmenter_mean_entropy']['auroc']:.6f}`.\n- **Triggering artifact:** `experiments/o04_model_confidence_ood/result.json`.\n- **Rationale:** Both preregistered frozen-model scores were evaluated independently.\n- **Verification:** Frozen hashes passed; masks, labels, external performance values, sealed-test access, neural refitting, and threshold changes were zero.\n- **Who approved:** User authorized O01–O06 in one sequential run.\n")
    common.mark_complete("o04_model_confidence_ood")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
