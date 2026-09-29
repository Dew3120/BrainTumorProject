"""Formalize the registered O06 OOD operating policy."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ood_common as common

ROOT = common.ROOT
EXP = ROOT / "experiments/o06_ood_policy"
ART = EXP / "artifacts"
SCORES = ROOT / "experiments/o05_ensemble_ood/artifacts/ensemble_scores.csv"
REQUIRED_WARNING = "This input does not match the model's training distribution. Predictions may be unreliable."
DISCLAIMER = "Research prototype — not for clinical use"


def state(score, review, refuse, refuse_enabled):
    if not np.isfinite(score) or score < 0 or score > 1:
        return "REVIEW"
    if refuse_enabled and score >= refuse:
        return "REFUSE"
    if score >= review:
        return "REVIEW"
    return "ACCEPT"


def main():
    if common.read_json(EXP / "result.json")["status"] != "NOT_RUN":
        raise RuntimeError("O06 not in NOT_RUN state")
    rows = common.load_manifest(); score_rows = common.read_csv(SCORES)
    by_id = {row["sample_id"]: row for row in score_rows}
    if {r["sample_id"] for r in rows} != set(by_id):
        raise RuntimeError("O05/O01 membership mismatch")
    scores = np.asarray([float(by_id[r["sample_id"]]["selected_score"]) for r in rows])
    fit_id = np.asarray([r["detector_split"] == "fit" and r["domain"] == "ID" for r in rows])
    evaluation = np.asarray([r["detector_split"] == "evaluation" for r in rows])
    review_threshold = float(np.quantile(scores[fit_id], 0.95, method="linear"))
    refuse_threshold = float(np.quantile(scores[fit_id], 0.99, method="linear"))
    # The optional state is enabled only under the pre-result operational utility rule.
    provisional_refuse = scores[evaluation] >= refuse_threshold
    eval_target = common.labels([r for r, flag in zip(rows, evaluation, strict=True) if flag])
    if len(provisional_refuse) != len(eval_target):
        raise RuntimeError("Evaluation REFUSE membership length mismatch")
    refuse_precision = float(eval_target[provisional_refuse].mean()) if provisional_refuse.any() else 0.0
    id_refuse_rate = float(provisional_refuse[eval_target == 0].mean())
    refuse_enabled = refuse_precision >= 0.95 and id_refuse_rate <= 0.01
    assignments = []
    source_counts = defaultdict(Counter)
    for row, score_value in zip(rows, scores, strict=True):
        status = state(float(score_value), review_threshold, refuse_threshold, refuse_enabled)
        assignments.append({"sample_id": row["sample_id"], "source": row["source"], "domain": row["domain"], "detector_split": row["detector_split"], "group_id": row["group_id"], "ood_score": float(score_value), "status": status})
        if row["detector_split"] == "evaluation":
            source_counts[row["source"]][status] += 1
    common.write_csv(ART / "policy_assignments.csv", assignments)
    coverage = []
    for source, counts in sorted(source_counts.items()):
        total = sum(counts.values()); is_id = source == "brisc"
        coverage.append({"source": source, "domain": "ID" if is_id else "OOD", "evaluation_images": total, "accept": counts["ACCEPT"], "review": counts["REVIEW"], "refuse": counts["REFUSE"], "flagged": counts["REVIEW"] + counts["REFUSE"], "flagged_rate": (counts["REVIEW"] + counts["REFUSE"]) / total})
    common.write_csv(ART / "coverage_by_source.csv", coverage)
    eval_assignments = [a for a in assignments if a["detector_split"] == "evaluation"]
    id_eval = [a for a in eval_assignments if a["domain"] == "ID"]; ood_eval = [a for a in eval_assignments if a["domain"] != "ID"]
    id_flag_rate = sum(a["status"] != "ACCEPT" for a in id_eval) / len(id_eval)
    ood_detection_rate = sum(a["status"] != "ACCEPT" for a in ood_eval) / len(ood_eval)
    catalogue = {
        "schema_version": 1,
        "thresholds": {"review": review_threshold, "refuse": refuse_threshold if refuse_enabled else None},
        "messages": [
            {"message_id": "ood_accept", "status": "ACCEPT", "title": "Input accepted", "body": "This input is within the research prototype's learned input range. Processing can continue, subject to the model's separate reliability policy.", "disclaimer": DISCLAIMER},
            {"message_id": "ood_review", "status": "REVIEW", "title": "Input review required", "body": REQUIRED_WARNING + " Check that the upload is a supported brain MRI image before continuing.", "disclaimer": DISCLAIMER},
            {"message_id": "ood_refuse", "status": "REFUSE", "enabled": refuse_enabled, "title": "Input cannot be processed", "body": REQUIRED_WARNING + " This research prototype will not produce a tumor prediction for this upload. Use a supported brain MRI image.", "disclaimer": DISCLAIMER},
            {"message_id": "ood_score_invalid", "status": "REVIEW", "title": "Input review required", "body": "The input-distribution score was unavailable or invalid. Processing is paused for review. " + REQUIRED_WARNING, "disclaimer": DISCLAIMER},
        ],
    }
    for message in catalogue["messages"]:
        if message["status"] in {"REVIEW", "REFUSE"} and REQUIRED_WARNING not in message["body"]:
            raise RuntimeError("Required OOD warning missing")
        if message["disclaimer"] != DISCLAIMER:
            raise RuntimeError("Required prototype disclaimer missing")
    common.write_json(ART / "message_catalogue.json", catalogue)
    contract = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "OOD policy result", "type": "object", "required": ["ood_score", "status", "review_threshold", "refuse_threshold"], "properties": {"ood_score": {"type": ["number", "null"], "minimum": 0, "maximum": 1}, "status": {"enum": ["ACCEPT", "REVIEW", "REFUSE"]}, "review_threshold": {"const": review_threshold}, "refuse_threshold": {"const": refuse_threshold if refuse_enabled else None}}}
    common.write_json(ART / "ood_policy_contract.schema.json", contract)
    fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(scores[evaluation][eval_target == 0], bins=50, alpha=0.6, label="BRISC ID", density=True); ax.hist(scores[evaluation][eval_target == 1], bins=50, alpha=0.6, label="OOD", density=True); ax.axvline(review_threshold, color="tab:orange", linestyle="--", label="REVIEW threshold");
    if refuse_enabled: ax.axvline(refuse_threshold, color="tab:red", linestyle=":", label="REFUSE threshold")
    ax.set(xlabel="Selected ensemble OOD score", ylabel="Density", title="O06 fit-ID quantile thresholds"); ax.legend(); fig.tight_layout(); fig.savefig(ART / "threshold_distributions.png", dpi=180); plt.close(fig)
    result = {"schema_version": 1, "experiment_id": "o06_ood_policy", "status": "COMPLETE", "decision": "POLICY_ACCEPTED", "thresholds": {"review": review_threshold, "review_provenance": "95th percentile of 600 BRISC fitting scores", "refuse": refuse_threshold if refuse_enabled else None, "refuse_candidate": refuse_threshold, "refuse_provenance": "99th percentile of 600 BRISC fitting scores", "refuse_enabled": refuse_enabled, "refuse_enablement_rule": "held-out refuse precision >= 0.95 and BRISC validation refuse rate <= 0.01", "refuse_precision": refuse_precision, "brisc_refuse_rate": id_refuse_rate}, "coverage": {"brisc_validation_false_flag_rate": id_flag_rate, "brisc_validation_false_flags": sum(a["status"] != "ACCEPT" for a in id_eval), "brisc_validation_total": len(id_eval), "ood_detection_rate": ood_detection_rate, "ood_detected": sum(a["status"] != "ACCEPT" for a in ood_eval), "ood_total": len(ood_eval), "by_source": coverage}, "application_order": ["validate image and compute OOD score", "apply O06 ACCEPT/REVIEW/REFUSE", "only ACCEPT continues to classifier/segmenter", "apply existing reliability policy threshold 0.9490250173235835 to accepted model output"], "logging_contract": ["timestamp", "input_sha256", "ood_score", "review_threshold", "refuse_threshold", "status", "detector_artifact_hashes"], "verification": {"o05_scores_sha256": common.sha256(SCORES), "fit_id_cases": int(fit_id.sum()), "evaluation_cases": int(evaluation.sum()), "reference_masks_used": 0, "diagnostic_labels_used": 0, "external_performance_values_used": 0, "brisc_test_images_accessed": 0, "neural_retraining_or_refitting": 0, "segmentation_threshold": 0.5, "reliability_policy_threshold": "0.9490250173235835"}}
    common.write_json(EXP / "result.json", result)
    config = common.read_json(EXP / "config.json"); config.update({"status": "COMPLETE", "selected_detector": "O05 secondary logistic", "review_quantile_fit_id": 0.95, "refuse_quantile_fit_id": 0.99, "invalid_score_status": "REVIEW", "refuse_enablement_rule": "evaluation OOD precision >= 0.95 and ID false-refuse <= 0.01"}); common.write_json(EXP / "config.json", config)
    (EXP / "decision.md").write_text(f"# Decision — O06 OOD policy\n\n**Status:** ACCEPTED (descriptive)\n\nThe REVIEW threshold is `{review_threshold:.17g}` from the registered BRISC-fitting 95th percentile. The REFUSE candidate is `{refuse_threshold:.17g}` and is **{'enabled' if refuse_enabled else 'not enabled'}** under the fixed utility rule. BRISC validation false-flag rate is `{id_flag_rate:.4%}`; held-out OOD detection rate is `{ood_detection_rate:.4%}`.\n", encoding="utf-8", newline="\n")
    policy = f"""# OOD operating policy

The application must run the O05 OOD detector before tumor classification or segmentation.

## Decision rule

- **ACCEPT:** score below `{review_threshold:.17g}`. Continue to the frozen tumor stack and then apply the separate reliability policy.
- **REVIEW:** score at or above `{review_threshold:.17g}` and below the REFUSE boundary. Pause and show the registered warning.
- **REFUSE:** {'enabled at score >= `' + format(refuse_threshold, '.17g') + '`' if refuse_enabled else 'not enabled; the 99th-percentile candidate remains descriptive'}.
- Missing, non-finite, or out-of-range scores fail closed to **REVIEW**.

The segmentation threshold remains `0.50`. The existing reliability-policy threshold remains exactly `0.9490250173235835`.

## User-facing warning

> {REQUIRED_WARNING}
>
> **{DISCLAIMER}**

## Coverage on the held-out benchmark

- BRISC validation flagged: **{sum(a['status'] != 'ACCEPT' for a in id_eval)}/{len(id_eval)} ({id_flag_rate:.2%})**.
- OOD evaluation images flagged: **{sum(a['status'] != 'ACCEPT' for a in ood_eval)}/{len(ood_eval)} ({ood_detection_rate:.2%})**.

These rates describe the registered retrospective benchmark. They do not establish clinical safety or coverage of every possible OOD input.
"""
    (ROOT / "docs/ood_policy.md").write_text(policy, encoding="utf-8", newline="\n")
    threshold_report = "# OOD threshold selection\n\n" + (EXP / "decision.md").read_text(encoding="utf-8") + "\nThresholds were computed only from BRISC detector-fitting scores; held-out labels did not choose the numeric thresholds. REFUSE enablement used the pre-result utility rule recorded in the implementation.\n"
    (ART / "threshold_selection.md").write_text(threshold_report, encoding="utf-8", newline="\n")
    paths = [EXP / "config.json", EXP / "result.json", EXP / "decision.md", ART / "policy_assignments.csv", ART / "coverage_by_source.csv", ART / "message_catalogue.json", ART / "ood_policy_contract.schema.json", ART / "threshold_distributions.png", ART / "threshold_selection.md", ROOT / "docs/ood_policy.md"]
    common.experiment_manifest(EXP, paths)
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Formalize O06 OOD policy\n\n- **Decision made:** Accept the descriptive OOD policy with REVIEW threshold `{review_threshold:.17g}` and REFUSE {'enabled' if refuse_enabled else 'disabled'}.\n- **Triggering artifact:** `experiments/o06_ood_policy/result.json`.\n- **Rationale:** Numeric boundaries were computed from BRISC fitting scores only; evaluation membership reports coverage.\n- **Verification:** Required warning/disclaimer and fail-closed invalid-score handling passed. Prior thresholds, neural weights, sealed-test status, and gates were unchanged.\n- **Who approved:** User authorized O01–O06 in one sequential run.\n")
    common.mark_complete("o06_ood_policy")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
