"""Independently verify and finalize the completed X03 external evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import evaluate_x01_cheng as x01
import evaluate_x03_upenn_gbm as x03
import numpy as np
import stage6_external_common as external

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x03_upenn_gbm"
ARTIFACTS = EXPERIMENT / "artifacts"
REPORT = ROOT / "audit/x03_upenn_gbm_evaluation.md"
EXPECTED_LAUNCH_COMMIT = "a786ad39f93bce7bf679ae8c6c61b1ebed74229c"


def as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"Invalid serialized boolean: {value!r}")


def close(name: str, observed, expected, differences: list[float]) -> None:
    difference = abs(float(observed) - float(expected))
    differences.append(difference)
    if difference > 1e-12:
        raise ValueError(f"Independent recompute mismatch for {name}: {difference}")


def subject_mean(values, rows) -> float:
    grouped = defaultdict(list)
    for value, row in zip(values, rows, strict=True):
        grouped[row["patient_id"]].append(float(value))
    return float(np.mean([np.mean(grouped[key]) for key in sorted(grouped)]))


def verify_target_losses(rows: list[dict]) -> None:
    normalized = [
        {
            "patient_id": row["patient_id"],
            "subject_id": row["subject_id"],
            "axial_index_ras": int(row["axial_index_ras"]),
            "case_id": row["case_id"],
            "native_target_pixels": int(row["native_target_pixels"]),
            "post_resize_target_pixels": int(row["post_resize_target_pixels"]),
            "volume_edge": as_bool(row["volume_edge"]),
            "orientation": row["orientation"],
        }
        for row in rows
    ]
    observed_hash = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if (
        len(normalized) != 82
        or len({row["patient_id"] for row in normalized}) != 54
        or min(row["native_target_pixels"] for row in normalized) != 1
        or max(row["native_target_pixels"] for row in normalized) != 15
        or any(row["post_resize_target_pixels"] != 0 for row in normalized)
        or any(row["volume_edge"] for row in normalized)
        or {row["orientation"] for row in normalized} != {"axial"}
        or observed_hash != x03.EXPECTED_TARGET_LOSS_SHA256
    ):
        raise ValueError(
            "Target-not-representable artifact changed: "
            f"count={len(normalized)}, hash={observed_hash}"
        )


def recompute(result: dict) -> dict:
    per_case = x03.read_csv(ARTIFACTS / "per_case_predictions.csv")
    target_loss = x03.read_csv(ARTIFACTS / "target_not_representable.csv")
    whole_subject = x03.read_csv(ARTIFACTS / "whole_volume_subject_metrics.csv")
    whole_slice = x03.read_csv(ARTIFACTS / "whole_volume_slice_counts.csv")
    failures = x03.read_csv(ARTIFACTS / "failure_taxonomy.csv")
    verify_target_losses(target_loss)
    if len(per_case) != 9464 or len(whole_subject) != 147 or len(whole_slice) != 22785:
        raise ValueError("Saved X03 evaluation membership changed")
    if len({row["case_id"] for row in per_case}) != 9464:
        raise ValueError("Duplicate per-case prediction row")

    rows = [
        {**row, "patient_id": row["subject_id"], "label": "glioma"}
        for row in per_case
    ]
    dice = np.asarray([float(row["calibrated_dice"]) for row in rows])
    iou = np.asarray([float(row["calibrated_iou"]) for row in rows])
    boundary = np.asarray([float(row["boundary_f1_tolerance_2"]) for row in rows])
    representable = np.asarray(
        [not as_bool(row["target_not_representable_at_128"]) for row in rows]
    )
    probabilities = np.asarray(
        [
            [float(row[f"classifier_probability_{name}"]) for name in x03.CLASS_NAMES]
            for row in rows
        ]
    )
    reliability = np.asarray([float(row["r02_reliability"]) for row in rows])
    success = dice >= 0.5
    differences: list[float] = []

    classifier = x03.subject_classifier_evaluation(rows, probabilities)
    for key in ("accuracy", "glioma_recall", "nll", "brier", "top_label_ece", "top_label_mce"):
        close(f"classifier.{key}", classifier[key], result["classifier"][key], differences)
        for endpoint, (observed, expected) in enumerate(
            zip(
                classifier["bootstrap_ci95"][key],
                result["classifier"]["bootstrap_ci95"][key],
                strict=True,
            )
        ):
            close(f"classifier.{key}.ci{endpoint}", observed, expected, differences)

    segmentation = result["segmentation"]
    close("slice dice", dice.mean(), segmentation["slice_macro"]["dice"], differences)
    close("slice iou", iou.mean(), segmentation["slice_macro"]["iou"], differences)
    close(
        "strict subject dice",
        subject_mean(dice, rows),
        segmentation["subject_macro"]["dice"]["subject_macro_mean"],
        differences,
    )
    close(
        "strict subject iou",
        subject_mean(iou, rows),
        segmentation["subject_macro"]["iou"]["subject_macro_mean"],
        differences,
    )
    representable_rows = [row for row, keep in zip(rows, representable, strict=True) if keep]
    close(
        "representable subject dice",
        subject_mean(dice[representable], representable_rows),
        segmentation["model_space"]["dice"]["subject_macro_mean"],
        differences,
    )
    close(
        "representable subject iou",
        subject_mean(iou[representable], representable_rows),
        segmentation["model_space"]["iou"]["subject_macro_mean"],
        differences,
    )
    close(
        "strict subject boundary",
        subject_mean(boundary, rows),
        segmentation["subject_macro"]["boundary_f1_tolerance_2"]["subject_macro_mean"],
        differences,
    )

    source_positive = [row for row in whole_slice if as_bool(row["reference_has_tumor"])]
    if [row["case_id"] for row in source_positive] != [row["case_id"] for row in rows]:
        raise ValueError("Source-coordinate membership/order changed")
    source_dice = np.asarray([float(row["dice"]) for row in source_positive])
    source_iou = np.asarray([float(row["iou"]) for row in source_positive])
    close(
        "source subject dice",
        subject_mean(source_dice, rows),
        segmentation["source_coordinate_all_9464"]["dice"]["subject_macro_mean"],
        differences,
    )
    close(
        "source subject iou",
        subject_mean(source_iou, rows),
        segmentation["source_coordinate_all_9464"]["iou"]["subject_macro_mean"],
        differences,
    )

    whole_dice = np.asarray([float(row["dice"]) for row in whole_subject])
    whole_iou = np.asarray([float(row["iou"]) for row in whole_subject])
    close(
        "whole-volume subject dice",
        whole_dice.mean(),
        segmentation["exploratory_whole_volume"]["subject_macro_dice"],
        differences,
    )
    close(
        "whole-volume subject iou",
        whole_iou.mean(),
        segmentation["exploratory_whole_volume"]["subject_macro_iou"],
        differences,
    )
    total_intersection = sum(int(row["intersection_pixels"]) for row in whole_subject)
    total_prediction = sum(int(row["prediction_pixels"]) for row in whole_subject)
    total_reference = sum(int(row["reference_pixels"]) for row in whole_subject)
    close(
        "whole-volume global dice",
        2 * total_intersection / (total_prediction + total_reference),
        segmentation["exploratory_whole_volume"]["cohort_global_dice"],
        differences,
    )

    reliability_result = x01.reliability_metrics(reliability, success, rows)
    for key in ("failure_detection_auroc", "ece", "mce", "brier", "brier_skill"):
        close(
            f"reliability.{key}",
            reliability_result[key],
            result["reliability_transfer"][key],
            differences,
        )
        for endpoint, (observed, expected) in enumerate(
            zip(
                reliability_result["bootstrap_ci95"][key],
                result["reliability_transfer"]["bootstrap_ci95"][key],
                strict=True,
            )
        ):
            close(f"reliability.{key}.ci{endpoint}", observed, expected, differences)
    top_count = math.ceil(0.2 * len(rows))
    top = np.argsort(reliability)[:top_count]
    if int((~success[top]).sum()) != result["reliability_transfer"]["top20_risk"]["failures_captured"]:
        raise ValueError("Top-20% failure capture mismatch")
    review = reliability < external.RELIABILITY_POLICY_THRESHOLD
    policy = result["reliability_transfer"]["policy"]
    expected_policy = {
        "review_required": int(review.sum()),
        "standard": int((~review).sum()),
        "failures_caught": int((review & ~success).sum()),
        "failures_missed": int((~review & ~success).sum()),
        "acceptable_reviewed": int((review & success).sum()),
    }
    for key, value in expected_policy.items():
        if value != policy[key]:
            raise ValueError(f"Policy count mismatch: {key}")

    normalized_failures = []
    boolean_fields = (
        "empty_prediction",
        "nonempty_zero_overlap",
        "severe_under_prediction",
        "severe_over_prediction",
        "fragmented_prediction",
        "disconnected_false_positive",
        "reference_touches_model_edge",
        "prediction_touches_model_edge",
    )
    for row in failures:
        normalized_failures.append(
            {**row, **{field: as_bool(row[field]) for field in boolean_fields}}
        )
    taxonomy = x03.summarize_failure_modes(normalized_failures)
    if taxonomy != result["failure_taxonomy"]:
        raise ValueError("Failure-taxonomy independent recompute mismatch")

    lost = [row for row in rows if as_bool(row["target_not_representable_at_128"])]
    lost_scores = np.asarray([float(row["r02_reliability"]) for row in lost])
    representable_scores = reliability[representable]
    lost_in_top20 = sum(int(index) in set(top.tolist()) for index in np.where(~representable)[0])
    recomputed_gates = {
        "classifier_glioma_recall_at_least_0_85": classifier["glioma_recall"] >= 0.85,
        "classifier_top_label_ece_at_most_0_05": classifier["top_label_ece"] <= 0.05,
        "glioma_subject_macro_dice_at_least_0_55": subject_mean(dice, rows) >= 0.55,
        "reliability_auroc_at_least_0_75": reliability_result["failure_detection_auroc"] >= 0.75,
        "reliability_auroc_ci_lower_at_least_0_70": reliability_result["bootstrap_ci95"]["failure_detection_auroc"][0] >= 0.70,
        "reliability_ece_at_most_0_05": reliability_result["ece"] <= 0.05,
        "reliability_brier_skill_positive": reliability_result["brier_skill"] > 0,
        "reliability_brier_skill_ci_lower_positive": reliability_result["bootstrap_ci95"]["brier_skill"][0] > 0,
        "top20_failure_capture_at_least_half": int((~success[top]).sum())
        / int((~success).sum())
        >= 0.5,
    }
    if recomputed_gates != result["gates"]["checks"]:
        raise ValueError("Registered gate recompute mismatch")

    return {
        "maximum_absolute_metric_difference": max(differences, default=0.0),
        "metrics_recomputed": len(differences),
        "rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "strict_failures": int((~success).sum()),
        "gates": recomputed_gates,
        "target_not_representable": {
            "count": len(lost),
            "all_review_required": all(row["policy_status"] == "REVIEW_REQUIRED" for row in lost),
            "mean_reliability": float(lost_scores.mean()),
            "representable_mean_reliability": float(representable_scores.mean()),
            "captured_in_top20_risk": int(lost_in_top20),
        },
        "pixel_calibration_verification": (
            "Result and standalone JSON are byte-equivalent data; per-pixel probabilities were not persisted, "
            "so the descriptive calibration aggregates cannot be independently regenerated without forbidden repeat inference."
        ),
    }


def manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x03_upenn_gbm",
        "status": "COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": x01.sha256(path),
            }
            for path in sorted(set(paths))
        ],
    }


def main() -> None:
    evaluation_path = ARTIFACTS / "evaluation_result.json"
    result = x03.read_json(evaluation_path)
    if result["status"] != "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING":
        raise ValueError("X03 result is not awaiting first finalization")
    if result["verification"]["source_commit"] != EXPECTED_LAUNCH_COMMIT:
        raise ValueError("X03 launch commit mismatch")
    if result["verification"]["brisc_files_opened"]["sealed_test"] != 0:
        raise ValueError("BRISC sealed-test access is nonzero")
    if result["verification"]["inference_attempts"] != {
        "attempt_1_target_representability_stop_slices_inferred": 0,
        "attempt_2_loader_defect_stop_slices_inferred": 144,
        "attempt_3_domain_bootstrap_stop_slices_inferred": 22785,
        "attempt_4_completed": 1,
    }:
        raise ValueError("X03 inference-attempt audit trail changed")
    verification = recompute(result)
    if x03.read_json(ARTIFACTS / "pixel_calibration.json") != result["pixel_calibration"]:
        raise ValueError("Pixel-calibration standalone artifact mismatch")
    if x03.read_json(ARTIFACTS / "domain_gap.json") != result["domain_gap"]:
        raise ValueError("Domain-gap standalone artifact mismatch")
    if x03.read_json(ARTIFACTS / "failure_taxonomy_summary.json") != result["failure_taxonomy"]:
        raise ValueError("Failure-taxonomy standalone artifact mismatch")

    result["status"] = "FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL"
    result["decision"] = (
        "Reject H8 transport for X03. Preserve the independent external result as evidence "
        "of severe domain shift; do not tune on X03."
    )
    result["visual_review"] = {
        "status": "PASS_NO_SYSTEMATIC_TRANSFORM_DEFECT",
        "source_alignment_panel": "experiments/x03_upenn_gbm/artifacts/source_alignment_panel.png",
        "prediction_panel": "experiments/x03_upenn_gbm/artifacts/visual_audit_panel.png",
        "representable_prediction_panel": "experiments/x03_upenn_gbm/artifacts/visual_audit_representable_panel.png",
        "finding": (
            "The source panel shows varied axial scans with whole-tumor references aligned to visible lesions. "
            "Best predictions overlap closely; worst predictions are complete misses or false regions without "
            "a common flip, transpose, or fixed offset. The two deterministic prediction panels are byte-identical "
            "because all selected extrema are representable cases, which is an expected selection outcome."
        ),
    }
    result["independent_recompute"] = verification
    x03.write_json(evaluation_path, result)
    x03.write_json(EXPERIMENT / "result.json", result)

    classifier = result["classifier"]
    segmentation = result["segmentation"]
    reliability = result["reliability_transfer"]
    calibration = result["pixel_calibration"]
    failure = result["failure_taxonomy"]["all"]
    gap = result["domain_gap"]
    gate_failures = [name for name, passed in result["gates"]["checks"].items() if not passed]
    report = f"""# X03 UPENN-GBM external evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

This is an independent, retrospective research transport evaluation of the frozen E08 + C03 + R02 stack on the registered UPENN-GBM derivative release. Glioblastoma is a subtype of glioma, so the registered mapping is coarse; these results are not directly comparable to the heterogeneous BRISC glioma category. This is not clinical validation.

## Cohort and execution

- 147 unique subjects, 9,464 tumor-bearing axial slices, and 22,785 total axial slices.
- Eighty-two tumor masks from 54 subjects vanish under the frozen 240×240 → 128×128 nearest-neighbor target transform. They remain in the strict gate with zero scores and are listed by identifier in `target_not_representable.csv`.
- Attempts 1–3 stopped fail-closed for target representability, a native-datatype loader defect, and a bootstrap-membership defect. Amendment 19 authorized attempt 4, which completed from frozen launch commit `{EXPECTED_LAUNCH_COMMIT}`.
- Attempt 4 inferred all 22,785 slices once. The neural models, C03 mapping, R02 model, image adapter, mask transform, and registered thresholds remained frozen.

## Five analytical views

| View | Population | Subject-macro Dice | 95% CI | Role |
|---|---:|---:|---:|---|
| Strict model-space | 9,464 slices / 147 subjects | {segmentation['subject_macro']['dice']['subject_macro_mean']:.4f} | {segmentation['subject_macro']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['subject_macro']['dice']['bootstrap_ci95'][1]:.4f} | Registered gate; 82 erased targets receive 0 |
| Representable model-space | 9,382 slices / 147 subjects | {segmentation['model_space']['dice']['subject_macro_mean']:.4f} | {segmentation['model_space']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['model_space']['dice']['bootstrap_ci95'][1]:.4f} | Secondary sensitivity |
| Native source-coordinate | 9,464 slices / 147 subjects | {segmentation['source_coordinate_all_9464']['dice']['subject_macro_mean']:.4f} | {segmentation['source_coordinate_all_9464']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['source_coordinate_all_9464']['dice']['bootstrap_ci95'][1]:.4f} | Restored hard predictions at original resolution |
| Exploratory whole volume | 22,785 slices / 147 subjects | {segmentation['exploratory_whole_volume']['subject_macro_dice']:.4f} | {segmentation['exploratory_whole_volume']['subject_macro_dice_ci95'][0]:.4f}–{segmentation['exploratory_whole_volume']['subject_macro_dice_ci95'][1]:.4f} | Exploratory; not the gate |
| Confidence transport | Calibration n=9,382; reliability n=9,464 | Reliability AUROC {reliability['failure_detection_auroc']:.4f} | {reliability['bootstrap_ci95']['failure_detection_auroc'][0]:.4f}–{reliability['bootstrap_ci95']['failure_detection_auroc'][1]:.4f} | Pixel calibration + strict reliability outcome |

The 82 erased targets are not the main explanation for the low Dice. Removing them raises subject-macro Dice only from `{segmentation['subject_macro']['dice']['subject_macro_mean']:.4f}` to `{segmentation['model_space']['dice']['subject_macro_mean']:.4f}`, and the native-resolution view remains `{segmentation['source_coordinate_all_9464']['dice']['subject_macro_mean']:.4f}`.

## Classification

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Glioma recall / patient accuracy | {classifier['glioma_recall']:.4f} | ≥0.85 | Fail |
| Top-label ECE | {classifier['top_label_ece']:.4f} | ≤0.05 | Fail |
| NLL | {classifier['nll']:.4f} | Descriptive | — |
| Brier | {classifier['brier']:.4f} | Descriptive | — |

The classifier predicts {classifier['confusion']['glioma']}/147 subjects as glioma and {classifier['confusion']['no_tumor']}/147 as no tumor. OVR AUROC, no-tumor specificity, and binary tumor/no-tumor AUROC are not estimable in this single-category cohort.

## Segmentation and failure modes

- Strict glioma subject-macro Dice: `{segmentation['subject_macro']['dice']['subject_macro_mean']:.4f}` against the registered `0.55` gate — **fail**.
- Failures below Dice 0.50: {segmentation['dice_below_0_50']}/9,464; zero-Dice cases: {segmentation['zero_dice']}; empty predictions: {segmentation['empty_prediction']}.
- Complete misses: {failure['complete_miss']['count']}/{failure['count']} ({100 * failure['complete_miss']['fraction']:.1f}%). Non-empty zero-overlap: {failure['nonempty_zero_overlap']['count']} ({100 * failure['nonempty_zero_overlap']['fraction']:.1f}%).
- Fragmented predictions: {failure['fragmented_prediction']['count']} ({100 * failure['fragmented_prediction']['fraction']:.1f}%); disconnected false positives: {failure['disconnected_false_positive']['count']} ({100 * failure['disconnected_false_positive']['fraction']:.1f}%).
- The deterministic panels show valid reference alignment and clean best-case overlap. Worst cases are model/task failures without a common coordinate transform defect.

## Pixel calibration

Pixel calibration uses the 9,382 representable references because lesion-centred regions are undefined for the 82 empty model-space targets.

| Region | Metric | Raw | C03 |
|---|---|---:|---:|
| Lesion-centred | NLL | {calibration['lesion_centered']['raw']['nll']:.4f} | {calibration['lesion_centered']['c03_operational']['nll']:.4f} |
| Lesion-centred | Brier | {calibration['lesion_centered']['raw']['brier']:.4f} | {calibration['lesion_centered']['c03_operational']['brier']:.4f} |
| Lesion-centred | ECE | {calibration['lesion_centered']['raw']['ece']:.4f} | {calibration['lesion_centered']['c03_operational']['ece']:.4f} |
| Background | ECE | {calibration['background']['raw']['ece']:.4f} | {calibration['background']['c03_operational']['ece']:.4f} |
| Whole image | ECE | {calibration['whole_image']['raw']['ece']:.4f} | {calibration['whole_image']['c03_operational']['ece']:.4f} |

C03 lowers NLL, Brier, ECE, and MCE numerically in the lesion-centred region and also improves background and whole-image summaries. Lesion-centred ECE remains high at `{calibration['lesion_centered']['c03_operational']['ece']:.4f}`, so calibration transport remains inadequate despite the improvement.

## Reliability transfer

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Failure AUROC | {reliability['failure_detection_auroc']:.4f} | ≥0.75 | Pass |
| AUROC lower CI | {reliability['bootstrap_ci95']['failure_detection_auroc'][0]:.4f} | ≥0.70 | Pass |
| ECE | {reliability['ece']:.4f} | ≤0.05 | Fail |
| Brier skill | {reliability['brier_skill']:.4f} | >0 with lower CI >0 | Pass |
| Top-20% failure capture | {reliability['top20_risk']['failures_captured']}/{reliability['top20_risk']['failures_total']} ({100 * reliability['top20_risk']['failures_captured'] / reliability['top20_risk']['failures_total']:.1f}%) | ≥50% | Fail |

The top-20% gate is mathematically unattainable when {reliability['top20_risk']['failures_total']:,}/9,464 cases fail because {reliability['top20_risk']['cases']:,} risk slots cannot contain half of those failures. The gate is retained and fails. At the fixed policy threshold, {reliability['policy']['review_required']:,}/9,464 cases require review, {reliability['policy']['failures_caught']:,} failures are caught, {reliability['policy']['failures_missed']:,} are missed, and total P01 cost is {reliability['policy']['p01_total_cost']:,}.

All 82 erased-target cases are retained in reliability evaluation. Their mean score is `{verification['target_not_representable']['mean_reliability']:.4f}` versus `{verification['target_not_representable']['representable_mean_reliability']:.4f}` for representable cases, and {verification['target_not_representable']['captured_in_top20_risk']}/82 enter the highest-risk 20%. Target representability can affect prediction-derived reliability inputs even though R02 never reads a reference mask.

## Domain gap

- Model-input mean Wasserstein distance: `{gap['continuous_distribution_shift']['model_input_mean']['wasserstein']:.4f}`; fixed-histogram JS divergence: `{gap['continuous_distribution_shift']['model_input_mean']['jensen_shannon_divergence']:.4f}`.
- Frozen-encoder centroid shift: `{gap['encoder_representation_shift']['centroid_euclidean']:.4f}`.
- Reference foreground fraction: UPENN-GBM `{gap['continuous_distribution_shift']['foreground_fraction']['external_mean']:.4f}` vs BRISC validation `{gap['continuous_distribution_shift']['foreground_fraction']['brisc_validation_mean']:.4f}`.
- Fragmented reference fraction: UPENN-GBM `{gap['fragmented_reference_fraction']['upenn_gbm']:.4f}` vs BRISC validation `{gap['fragmented_reference_fraction']['brisc_validation']:.4f}`.
- Reliability-score Wasserstein distance: `{gap['reliability_score_shift']['wasserstein']:.4f}`; policy review rate shifts from `{gap['policy_review_rate']['brisc_operational_reference']:.1%}` on BRISC development validation to `{gap['policy_review_rate']['upenn_gbm']:.1%}` on X03.

The strong intensity and encoder shifts, much greater reference fragmentation, and broad complete-miss pattern support a genuine domain-transport failure rather than an explanation limited to the 82 erased targets. Mask-derived domain features were bootstrapped on exactly 9,382 representable rows; input, encoder, reliability, and policy features used exactly 9,464 primary rows under Amendment 19.

## Registered decision

Failed gates: {', '.join(f'`{name}`' for name in gate_failures)}.

H8 is rejected for X03. No threshold, gate, model, calibration mapping, or reliability model is changed. X03 must not be used for tuning under this protocol.

## Verification and limits

- Independent recomputation checked {verification['metrics_recomputed']} decision-driving values across 9,464 primary rows, 147 subjects, 22,785 volume slices, classifier aggregation, segmentation views, failure taxonomy, reliability bootstrap, policy counts, and all gates. Maximum absolute difference: `{verification['maximum_absolute_metric_difference']}`.
- Pixel-calibration and domain-gap standalone JSON artifacts match the completed result exactly. Per-pixel probability maps were not persisted, so the descriptive pixel-calibration aggregates cannot be independently regenerated without another inference run; no repeat inference was performed.
- BRISC development validation files used for frozen domain reference: 680. BRISC sealed-test files opened: 0.
- Neural retraining: 0. Calibrator refits: 0. Reliability refits: 0. Segmentation-threshold changes: 0. Policy-threshold changes: 0.
- This evaluates one retrospective, single-site, glioma-only cohort. It does not establish prospective generalization, clinical validity, diagnostic safety, or suitability for patient care.
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = x03.read_json(config_path)
    if config["status"] != "AUDIT_COMPLETE":
        raise ValueError("X03 config state changed before finalization")
    config.update(
        {
            "status": "COMPLETE",
            "evaluation_outcome": "FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL",
            "amendment_19": "audit/preregistration_amendment_19.md",
            "aborted_inference_attempts": 3,
            "completed_inference_attempts": 1,
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    x03.write_json(config_path, config)

    decision_path = EXPERIMENT / "decision.md"
    with decision_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "\n## Final external-evaluation decision\n\n"
            "**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`\n\n"
            "The independent UPENN-GBM cohort failed classifier recall and calibration, strict glioma Dice, "
            "case-reliability ECE, and top-20% failure-capture gates. Reliability ranking AUROC "
            "passed, but it does not override the conjunctive failures. The 82 erased targets "
            "remain explicit zero-score cases and do not explain the broad failure. No gate is relaxed and X03 "
            "will not be used for tuning.\n\n"
            "**Attempt 4:** Completed inference on all 22,785 slices and wrote the final registered strict, "
            "representable-only, native-resolution, and exploratory whole-volume views. Independent "
            "recomputation matched all 45 checked values exactly.\n"
        )

    registry_path = ROOT / "experiments/registry.json"
    registry = x03.read_json(registry_path)
    matches = [item for item in registry["experiments"] if item["id"] == "x03_upenn_gbm"]
    if len(matches) != 1 or matches[0]["status"] != "AUDIT_COMPLETE":
        raise ValueError("X03 registry state changed before finalization")
    matches[0]["status"] = "COMPLETE"
    x03.write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Reject X03 transport gates and retain independent external findings

- **Decision made:** Complete X03 as `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`. Reject H8 for this cohort without tuning on X03.
- **Triggering artifact:** `experiments/x03_upenn_gbm/artifacts/evaluation_result.json`, SHA256 `{x01.sha256(evaluation_path)}`; deterministic prediction panel SHA256 `{x01.sha256(ARTIFACTS / 'visual_audit_panel.png')}`.
- **Rationale:** Glioma recall failed at `{classifier['glioma_recall']:.6f}`, and classifier ECE was `{classifier['top_label_ece']:.6f}`. Strict all-9,464 subject-macro Dice was `{segmentation['subject_macro']['dice']['subject_macro_mean']:.6f}` against `0.55`. Reliability AUROC passed at `{reliability['failure_detection_auroc']:.6f}`, while ECE was `{reliability['ece']:.6f}` and top-20% captured `{reliability['top20_risk']['failures_captured']}/{reliability['top20_risk']['failures_total']}` failures. Complete misses accounted for `{failure['complete_miss']['count']}/{failure['count']}` failures.
- **Target representability:** The 82 Amendment-17 cases remain in the strict gate with zero scores. Representable-only Dice `{segmentation['model_space']['dice']['subject_macro_mean']:.6f}` and native source-coordinate Dice `{segmentation['source_coordinate_all_9464']['dice']['subject_macro_mean']:.6f}` show that the 82 cases do not explain the broad failure.
- **Verification:** Independent recomputation covered `{verification['metrics_recomputed']}` decision-driving values with maximum absolute difference `{verification['maximum_absolute_metric_difference']}`. Visual review found no systematic flip, transpose, offset, or source-mask alignment defect. No BRISC sealed-test image was opened. Neural retraining, calibration/reliability refit, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Drop the 82 nonrepresentable targets or report only the 9,382-slice sensitivity; rejected by Amendment 17. Modify native dtype handling instead of honoring the registered adapter contract; rejected by Amendment 18. Reuse primary membership for mask-domain bootstraps; rejected by Amendment 19. Tune on X03 or relax the top-20% gate; rejected because external data are evaluation-only and gates remain fixed.
- **Who approved:** User approved Amendments 17–19 and exactly one fourth attempt after the Amendment-19 correction.
"""
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(review_entry)

    artifact_paths = [
        path
        for path in ARTIFACTS.iterdir()
        if path.is_file() and path.name not in {".gitkeep", "experiment_manifest.json"}
    ]
    tracked_paths = [
        REPORT,
        ROOT / "audit/preregistration_amendment_17.md",
        ROOT / "audit/preregistration_amendment_18.md",
        ROOT / "audit/preregistration_amendment_19.md",
        ROOT / "scripts/evaluate_x03_upenn_gbm.py",
        ROOT / "scripts/finalize_x03_upenn_gbm.py",
        ROOT / "tests/test_evaluate_x03_upenn_gbm.py",
        ROOT / "tests/test_finalize_x03_upenn_gbm.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        EXPERIMENT / "result.json",
        decision_path,
    ]
    x03.write_json(ARTIFACTS / "experiment_manifest.json", manifest(artifact_paths + tracked_paths))


if __name__ == "__main__":
    main()
