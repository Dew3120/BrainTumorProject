"""Run registered O02 image-statistics OOD detector."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ood_common as common
from PIL import Image
from scipy.stats import kurtosis, skew
from skimage.feature import canny, corner_harris, corner_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = common.ROOT
EXP = ROOT / "experiments/o02_feature_ood"
ART = EXP / "artifacts"
FEATURES = ["histogram_entropy", "intensity_mean", "intensity_sd", "intensity_skew", "intensity_kurtosis", "fraction_q1", "fraction_q2", "fraction_q3", "fraction_q4", "aspect_ratio", "edge_density", "harris_corner_count", "file_bytes"]


def feature_row(row):
    path = ROOT / row["path"]
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float64) / 255.0
    hist = np.histogram(gray, bins=256, range=(0, 1))[0].astype(np.float64)
    p = hist[hist > 0] / hist.sum()
    flat = gray.ravel()
    edges = canny(gray, sigma=1.0)
    corners = corner_peaks(corner_harris(gray), min_distance=2, threshold_rel=0.01)
    values = {
        "histogram_entropy": float(-(p * np.log2(p)).sum()),
        "intensity_mean": float(flat.mean()), "intensity_sd": float(flat.std()),
        "intensity_skew": float(np.nan_to_num(skew(flat), nan=0.0, posinf=0.0, neginf=0.0)),
        "intensity_kurtosis": float(np.nan_to_num(kurtosis(flat), nan=0.0, posinf=0.0, neginf=0.0)),
        "fraction_q1": float((flat < 0.25).mean()), "fraction_q2": float(((flat >= 0.25) & (flat < 0.5)).mean()),
        "fraction_q3": float(((flat >= 0.5) & (flat < 0.75)).mean()), "fraction_q4": float((flat >= 0.75).mean()),
        "aspect_ratio": float(gray.shape[1] / gray.shape[0]), "edge_density": float(edges.mean()),
        "harris_corner_count": int(len(corners)), "file_bytes": int(path.stat().st_size),
    }
    return values


def main():
    if common.read_json(EXP / "result.json")["status"] != "NOT_RUN":
        raise RuntimeError("O02 not in NOT_RUN state")
    rows = common.load_manifest()
    extracted = []
    matrix = []
    for row in rows:
        values = feature_row(row)
        matrix.append([values[name] for name in FEATURES])
        extracted.append({**{key: row[key] for key in ("sample_id", "source", "domain", "detector_split", "group_id")}, **values})
    matrix = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite O02 feature")
    fit = np.asarray([row["detector_split"] == "fit" for row in rows])
    evaluation = ~fit
    target = common.labels(rows)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42))
    model.fit(matrix[fit], target[fit])
    probability = model.predict_proba(matrix)[:, 1]
    for item, value in zip(extracted, probability, strict=True):
        item["ood_probability"] = float(value)
    eval_rows = [row for row, flag in zip(rows, evaluation, strict=True) if flag]
    eval_probability = probability[evaluation]
    report = common.metrics(eval_rows, eval_probability)
    gate_pass = report["auroc"] >= 0.85
    scaler = model.named_steps["standardscaler"]
    logistic = model.named_steps["logisticregression"]
    coefficients = [{"feature": name, "standardized_coefficient": float(value), "sign": "positive" if value > 0 else "negative" if value < 0 else "zero"} for name, value in zip(FEATURES, logistic.coef_[0], strict=True)]
    common.write_csv(ART / "image_features_and_scores.csv", extracted)
    common.write_csv(ART / "feature_coefficients.csv", coefficients)
    fpr, tpr, _ = common.roc_points(eval_rows, eval_probability)
    fig, ax = plt.subplots(figsize=(6, 6)); ax.plot(fpr, tpr, label=f"O02 AUROC {report['auroc']:.4f}"); ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="O02 feature OOD detector"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ART / "roc_curve.png", dpi=180); plt.close(fig)
    result = {"schema_version": 1, "experiment_id": "o02_feature_ood", "status": "COMPLETE", "decision": "PASS" if gate_pass else "FAIL", "gate": {"metric": "held-out AUROC", "threshold": 0.85, "equality_passes": True, "passed": gate_pass}, "evaluation": report, "population": {"fit": int(fit.sum()), "evaluation": int(evaluation.sum()), "evaluation_id": int((target[evaluation] == 0).sum()), "evaluation_ood": int((target[evaluation] == 1).sum())}, "model": {"type": "StandardScaler + L2 logistic regression", "C": 1.0, "solver": "lbfgs", "intercept": float(logistic.intercept_[0]), "features": FEATURES, "coefficients": coefficients, "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}, "verification": {"o01_manifest_sha256": common.sha256(common.MANIFEST), "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "brisc_test_images_accessed": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"}}
    common.write_json(EXP / "result.json", result)
    config = common.read_json(EXP / "config.json"); config.update({"status": "COMPLETE", "features": FEATURES, "model": {"type": "logistic_regression", "C": 1.0, "solver": "lbfgs", "max_iter": 2000}, "o01_manifest_sha256": common.sha256(common.MANIFEST)}); common.write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — O02 feature OOD detector\n\n**Status:** {'PASS' if gate_pass else 'FAIL'}\n\nHeld-out AUROC was `{report['auroc']:.6f}` (95% group-bootstrap CI `{report['auroc_ci95'][0]:.6f}`–`{report['auroc_ci95'][1]:.6f}`) against the registered `0.85` gate. The gate was not relaxed.\n", encoding="utf-8", newline="\n")
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "image_features_and_scores.csv", ART / "feature_coefficients.csv", ART / "roc_curve.png"]
    common.experiment_manifest(EXP, paths)
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Evaluate O02 feature OOD detector\n\n- **Decision made:** Record O02 as `{'PASS' if gate_pass else 'FAIL'}` without gate relaxation.\n- **Triggering artifact:** `experiments/o02_feature_ood/result.json`, held-out AUROC `{report['auroc']:.6f}`.\n- **Rationale:** Registered inference-time image features and held-out O01 evaluation membership were used.\n- **Verification:** No reference masks, diagnostic labels, external performance data, sealed-test pixels, neural refitting, or threshold changes were used.\n- **Who approved:** User authorized O01–O06 in one sequential run.\n")
    common.mark_complete("o02_feature_ood")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
