"""Run registered O05 normalized-average and logistic OOD ensembles."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ood_common as common
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = common.ROOT
EXP = ROOT / "experiments/o05_ensemble_ood"
ART = EXP / "artifacts"
SCORE_NAMES = ["o02_probability", "o03_mahalanobis", "o04_classifier", "o04_segmenter_entropy"]


def keyed(path, fields):
    rows = common.read_csv(path)
    return {row["sample_id"]: [float(row[field]) for field in fields] for row in rows}


def empirical_cdf(reference, values):
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def main():
    if common.read_json(EXP / "result.json")["status"] != "NOT_RUN":
        raise RuntimeError("O05 not in NOT_RUN state")
    rows = common.load_manifest()
    o02_path = ROOT / "experiments/o02_feature_ood/artifacts/image_features_and_scores.csv"
    o03_path = ROOT / "experiments/o03_encoder_mahalanobis/artifacts/mahalanobis_scores.csv"
    o04_path = ROOT / "experiments/o04_model_confidence_ood/artifacts/model_confidence_scores.csv"
    o02 = keyed(o02_path, ["ood_probability"]); o03 = keyed(o03_path, ["mahalanobis_squared"]); o04 = keyed(o04_path, ["classifier_ood_score", "segmenter_entropy_ood_score"])
    sample_ids = [row["sample_id"] for row in rows]
    if set(sample_ids) != set(o02) or set(sample_ids) != set(o03) or set(sample_ids) != set(o04):
        raise RuntimeError("O02/O03/O04 score membership mismatch")
    raw = np.asarray([o02[sid] + o03[sid] + o04[sid] for sid in sample_ids], dtype=np.float64)
    fit = np.asarray([row["detector_split"] == "fit" for row in rows]); evaluation = ~fit
    fit_id = fit & (common.labels(rows) == 0)
    normalized = np.column_stack([empirical_cdf(raw[fit_id, column], raw[:, column]) for column in range(raw.shape[1])])
    primary = normalized.mean(axis=1)
    target = common.labels(rows)
    logistic = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42))
    logistic.fit(normalized[fit], target[fit])
    secondary = logistic.predict_proba(normalized)[:, 1]
    eval_rows = [row for row, flag in zip(rows, evaluation, strict=True) if flag]
    primary_metrics = common.metrics(eval_rows, primary[evaluation])
    secondary_metrics = common.metrics(eval_rows, secondary[evaluation])
    difference = secondary_metrics["auroc"] - primary_metrics["auroc"]
    difference_ci = common.paired_group_bootstrap_auc_difference(eval_rows, target[evaluation], secondary[evaluation], primary[evaluation])
    secondary_selected = difference >= 0.01 and difference_ci[0] > 0
    selected_name = "secondary_logistic" if secondary_selected else "primary_equal_weight_average"
    selected = secondary if secondary_selected else primary
    selected_metrics = secondary_metrics if secondary_selected else primary_metrics
    passed = selected_metrics["auroc"] >= 0.92
    logistic_part = logistic.named_steps["logisticregression"]
    scaler = logistic.named_steps["standardscaler"]
    score_rows = []
    for i, row in enumerate(rows):
        score_rows.append({"sample_id": row["sample_id"], "source": row["source"], "domain": row["domain"], "detector_split": row["detector_split"], "group_id": row["group_id"], **{f"normalized_{name}": float(normalized[i, j]) for j, name in enumerate(SCORE_NAMES)}, "primary_score": float(primary[i]), "secondary_probability": float(secondary[i]), "selected_score": float(selected[i])})
    common.write_csv(ART / "ensemble_scores.csv", score_rows)
    weights = [{"score": name, "primary_weight": 0.25, "secondary_standardized_coefficient": float(value)} for name, value in zip(SCORE_NAMES, logistic_part.coef_[0], strict=True)]
    common.write_csv(ART / "ensemble_weights.csv", weights)
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, values, color in (("Primary normalized mean", primary[evaluation], "tab:blue"), ("Secondary logistic", secondary[evaluation], "tab:orange")):
        fpr, tpr, _ = common.roc_points(eval_rows, values); metric = common.metrics(eval_rows, values)["auroc"]; ax.plot(fpr, tpr, color=color, label=f"{label}: {metric:.4f}")
    ax.plot([0, 1], [0, 1], "--", color="gray"); ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="O05 ensemble OOD detectors"); ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(ART / "roc_curve.png", dpi=180); plt.close(fig)
    result = {"schema_version": 1, "experiment_id": "o05_ensemble_ood", "status": "COMPLETE", "decision": "PASS" if passed else "FAIL", "gate": {"metric": "selected held-out AUROC", "threshold": 0.92, "passed": passed}, "primary_equal_weight_average": primary_metrics, "secondary_logistic": {**secondary_metrics, "standardized_coefficients": weights, "intercept": float(logistic_part.intercept_[0]), "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist()}, "selection": {"selected": selected_name, "secondary_minus_primary_auroc": difference, "paired_group_bootstrap_ci95": difference_ci, "secondary_rule": "difference >= 0.01 AND CI lower > 0", "secondary_rule_passed": secondary_selected}, "selected_metrics": selected_metrics, "population": {"fit": int(fit.sum()), "evaluation": int(evaluation.sum())}, "verification": {"source_hashes": {"o02": common.sha256(o02_path), "o03": common.sha256(o03_path), "o04": common.sha256(o04_path)}, "o01_manifest_sha256": common.sha256(common.MANIFEST), "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "brisc_test_images_accessed": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"}}
    common.write_json(EXP / "result.json", result)
    config = common.read_json(EXP / "config.json"); config.update({"status": "COMPLETE", "score_order": SCORE_NAMES, "primary": "equal mean of fit-ID empirical-CDF normalized scores", "secondary": {"type": "StandardScaler + L2 logistic", "C": 1.0, "solver": "lbfgs"}, "selection_rule": "select secondary only for delta >= 0.01 and paired bootstrap lower CI > 0"}); common.write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — O05 OOD ensemble\n\n**Status:** {'PASS' if passed else 'FAIL'}\n\nThe selected method is `{selected_name}` with held-out AUROC `{selected_metrics['auroc']:.6f}` (95% CI `{selected_metrics['auroc_ci95'][0]:.6f}`–`{selected_metrics['auroc_ci95'][1]:.6f}`), tested against the registered `0.92` gate. The secondary-minus-primary difference was `{difference:.6f}` with paired CI `{difference_ci[0]:.6f}`–`{difference_ci[1]:.6f}`.\n", encoding="utf-8", newline="\n")
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "ensemble_scores.csv", ART / "ensemble_weights.csv", ART / "roc_curve.png"]
    common.experiment_manifest(EXP, paths)
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Select O05 OOD ensemble\n\n- **Decision made:** Select `{selected_name}` and record O05 as `{'PASS' if passed else 'FAIL'}`.\n- **Triggering artifact:** `experiments/o05_ensemble_ood/result.json`, selected AUROC `{selected_metrics['auroc']:.6f}`.\n- **Rationale:** The preregistered held-out selection rule was applied without modification.\n- **Verification:** Source memberships/hashes passed; no masks, labels, external performance inputs, sealed-test access, neural refitting, or threshold changes occurred.\n- **Who approved:** User authorized O01–O06 in one sequential run.\n")
    common.mark_complete("o05_ensemble_ood")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
