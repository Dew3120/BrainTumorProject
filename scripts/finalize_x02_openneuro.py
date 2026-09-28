"""Independently verify and finalize the completed X02 external evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import evaluate_x01_cheng as x01
import evaluate_x02_openneuro as x02
import numpy as np
import stage6_external_common as external

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x02_openneuro_pituitary"
ARTIFACTS = EXPERIMENT / "artifacts"
REPORT = ROOT / "audit/x02_openneuro_evaluation.md"
EXPECTED_LAUNCH_COMMIT = "99f17f7ca02af5adf735bd511cb66f9154043110"


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
    if len(rows) != 5:
        raise ValueError("Expected five target-not-representable rows")
    observed = {
        row["case_id"]: (
            int(row["source_tumor_pixels"]),
            int(row["post_resize_tumor_pixels"]),
        )
        for row in rows
    }
    expected = {case: (pixels, 0) for case, pixels in x02.EXPECTED_NONREPRESENTABLE.items()}
    if observed != expected:
        raise ValueError("Target-not-representable artifact changed")


def recompute(result: dict) -> dict:
    per_case = x02.read_csv(ARTIFACTS / "per_case_predictions.csv")
    target_loss = x02.read_csv(ARTIFACTS / "target_not_representable.csv")
    whole_subject = x02.read_csv(ARTIFACTS / "whole_volume_subject_metrics.csv")
    whole_slice = x02.read_csv(ARTIFACTS / "whole_volume_slice_counts.csv")
    failures = x02.read_csv(ARTIFACTS / "failure_taxonomy.csv")
    verify_target_losses(target_loss)
    if len(per_case) != 425 or len(whole_subject) != 50 or len(whole_slice) != 624:
        raise ValueError("Saved X02 evaluation membership changed")
    if len({row["case_id"] for row in per_case}) != 425:
        raise ValueError("Duplicate per-case prediction row")

    rows = [
        {**row, "patient_id": row["subject_id"], "label": "pituitary"}
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
            [float(row[f"classifier_probability_{name}"]) for name in x02.CLASS_NAMES]
            for row in rows
        ]
    )
    reliability = np.asarray([float(row["r02_reliability"]) for row in rows])
    success = dice >= 0.5
    differences: list[float] = []

    classifier = x02.subject_classifier_evaluation(rows, probabilities)
    for key in ("accuracy", "pituitary_recall", "nll", "brier", "top_label_ece", "top_label_mce"):
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
        segmentation["representable_only_model_space"]["dice"]["subject_macro_mean"],
        differences,
    )
    close(
        "representable subject iou",
        subject_mean(iou[representable], representable_rows),
        segmentation["representable_only_model_space"]["iou"]["subject_macro_mean"],
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
        segmentation["source_coordinate_all_425"]["dice"]["subject_macro_mean"],
        differences,
    )
    close(
        "source subject iou",
        subject_mean(source_iou, rows),
        segmentation["source_coordinate_all_425"]["iou"]["subject_macro_mean"],
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
    taxonomy = x02.summarize_failure_modes(normalized_failures)
    if taxonomy != result["failure_taxonomy"]:
        raise ValueError("Failure-taxonomy independent recompute mismatch")

    lost = [row for row in rows if as_bool(row["target_not_representable_at_128"])]
    lost_scores = np.asarray([float(row["r02_reliability"]) for row in lost])
    representable_scores = reliability[representable]
    lost_in_top20 = sum(int(index) in set(top.tolist()) for index in np.where(~representable)[0])
    recomputed_gates = {
        "classifier_pituitary_recall_at_least_0_85": classifier["pituitary_recall"] >= 0.85,
        "classifier_top_label_ece_at_most_0_05": classifier["top_label_ece"] <= 0.05,
        "pituitary_subject_macro_dice_at_least_0_75": subject_mean(dice, rows) >= 0.75,
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
        "experiment_id": "x02_openneuro_pituitary",
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
    result = x02.read_json(evaluation_path)
    if result["status"] != "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING":
        raise ValueError("X02 result is not awaiting first finalization")
    if result["verification"]["source_commit"] != EXPECTED_LAUNCH_COMMIT:
        raise ValueError("X02 launch commit mismatch")
    if result["verification"]["brisc_files_opened"]["sealed_test"] != 0:
        raise ValueError("BRISC sealed-test access is nonzero")
    if result["verification"]["inference_attempts"] != {
        "aborted_before_results": 1,
        "aborted_slices_inferred": 80,
        "completed": 1,
    }:
        raise ValueError("X02 inference-attempt audit trail changed")
    verification = recompute(result)
    if x02.read_json(ARTIFACTS / "pixel_calibration.json") != result["pixel_calibration"]:
        raise ValueError("Pixel-calibration standalone artifact mismatch")
    if x02.read_json(ARTIFACTS / "domain_gap.json") != result["domain_gap"]:
        raise ValueError("Domain-gap standalone artifact mismatch")
    if x02.read_json(ARTIFACTS / "failure_taxonomy_summary.json") != result["failure_taxonomy"]:
        raise ValueError("Failure-taxonomy standalone artifact mismatch")

    result["status"] = "FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL"
    result["decision"] = (
        "Reject H8 transport for X02. Preserve the independent external result as evidence "
        "of severe domain shift; do not tune on X02."
    )
    result["visual_review"] = {
        "status": "PASS_NO_SYSTEMATIC_TRANSFORM_DEFECT",
        "source_alignment_panel": "experiments/x02_openneuro_pituitary/artifacts/source_alignment_panel.png",
        "prediction_panel": "experiments/x02_openneuro_pituitary/artifacts/visual_audit_panel.png",
        "finding": (
            "The source panel shows label-1 masks on visible sellar lesions. Best predictions overlap the "
            "reference closely. Worst predictions are empty or spatial misses without a common flip, transpose, "
            "or fixed offset. The deterministic six worst rows come from sub-001 because tied zero-Dice rows use "
            "case ID as the tie-breaker; the source-alignment panel supplies cross-patient coverage."
        ),
    }
    result["independent_recompute"] = verification
    x02.write_json(evaluation_path, result)
    x02.write_json(EXPERIMENT / "result.json", result)

    classifier = result["classifier"]
    segmentation = result["segmentation"]
    reliability = result["reliability_transfer"]
    calibration = result["pixel_calibration"]
    failure = result["failure_taxonomy"]["all"]
    gap = result["domain_gap"]
    gate_failures = [name for name, passed in result["gates"]["checks"].items() if not passed]
    report = f"""# X02 OpenNeuro pituitary external evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`

This is an independent, retrospective research transport evaluation of the frozen E08 + C03 + R02 stack on OpenNeuro `ds006248` v1.0.0. It is not clinical validation.

## Cohort and execution

- 50 unique pituitary-adenoma subjects, 425 native source-positive coronal CE-T1 slices, and 624 total coronal slices.
- One first launch stopped fail-closed after 80 inferred slices and saved no result. Amendment 15 was registered before the one authorized restart.
- Five native source-positive edge masks (1.18%) vanish under the frozen 128×128 nearest-neighbor transform. They remain in the strict gate with zero scores and are listed in `target_not_representable.csv`.
- Frozen launch commit: `{EXPECTED_LAUNCH_COMMIT}`. Completed external inference runs: 1; aborted attempts: 1.

## Five analytical views

| View | Population | Subject-macro Dice | 95% CI | Role |
|---|---:|---:|---:|---|
| Strict model-space | 425 slices / 50 subjects | {segmentation['subject_macro']['dice']['subject_macro_mean']:.4f} | {segmentation['subject_macro']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['subject_macro']['dice']['bootstrap_ci95'][1]:.4f} | Registered gate; five erased targets receive 0 |
| Representable model-space | 420 slices / 50 subjects | {segmentation['representable_only_model_space']['dice']['subject_macro_mean']:.4f} | {segmentation['representable_only_model_space']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['representable_only_model_space']['dice']['bootstrap_ci95'][1]:.4f} | Secondary sensitivity |
| Native source-coordinate | 425 slices / 50 subjects | {segmentation['source_coordinate_all_425']['dice']['subject_macro_mean']:.4f} | {segmentation['source_coordinate_all_425']['dice']['bootstrap_ci95'][0]:.4f}–{segmentation['source_coordinate_all_425']['dice']['bootstrap_ci95'][1]:.4f} | Restored hard predictions at original resolution |
| Exploratory whole volume | 624 slices / 50 subjects | {segmentation['exploratory_whole_volume']['subject_macro_dice']:.4f} | {segmentation['exploratory_whole_volume']['subject_macro_dice_ci95'][0]:.4f}–{segmentation['exploratory_whole_volume']['subject_macro_dice_ci95'][1]:.4f} | Exploratory; not the gate |
| Confidence transport | Calibration n=420; reliability n=425 | Reliability AUROC {reliability['failure_detection_auroc']:.4f} | {reliability['bootstrap_ci95']['failure_detection_auroc'][0]:.4f}–{reliability['bootstrap_ci95']['failure_detection_auroc'][1]:.4f} | Pixel calibration + strict reliability outcome |

The five erased cases are not the explanation for the low Dice. Removing them raises subject-macro Dice only from `{segmentation['subject_macro']['dice']['subject_macro_mean']:.4f}` to `{segmentation['representable_only_model_space']['dice']['subject_macro_mean']:.4f}`, and the native-resolution view remains `{segmentation['source_coordinate_all_425']['dice']['subject_macro_mean']:.4f}`.

## Classification

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Pituitary recall / patient accuracy | {classifier['pituitary_recall']:.4f} | ≥0.85 | Pass |
| Top-label ECE | {classifier['top_label_ece']:.4f} | ≤0.05 | Fail |
| NLL | {classifier['nll']:.4f} | Descriptive | — |
| Brier | {classifier['brier']:.4f} | Descriptive | — |

The classifier predicts 47/50 subjects as pituitary and 3/50 as no tumor. OVR AUROC, no-tumor specificity, and binary tumor/no-tumor AUROC are not estimable in this single-category cohort.

## Segmentation and failure modes

- Strict pituitary subject-macro Dice: `{segmentation['subject_macro']['dice']['subject_macro_mean']:.4f}` against the unchanged `0.75` gate — **fail**.
- Failures below Dice 0.50: {segmentation['dice_below_0_50']}/425; zero-Dice cases: {segmentation['zero_dice']}; empty predictions: {segmentation['empty_prediction']}.
- Complete misses: {failure['complete_miss']['count']}/{failure['count']} ({100 * failure['complete_miss']['fraction']:.1f}%). Non-empty zero-overlap: {failure['nonempty_zero_overlap']['count']} ({100 * failure['nonempty_zero_overlap']['fraction']:.1f}%).
- Fragmented predictions: {failure['fragmented_prediction']['count']} ({100 * failure['fragmented_prediction']['fraction']:.1f}%); disconnected false positives: {failure['disconnected_false_positive']['count']} ({100 * failure['disconnected_false_positive']['fraction']:.1f}%).
- The deterministic panels show valid reference alignment and clean best-case overlap. Worst cases are model/task failures without a common coordinate transform defect.

## Pixel calibration

Pixel calibration uses the 420 representable references because lesion-centred regions are undefined for the five empty model-space targets.

| Region | Metric | Raw | C03 |
|---|---|---:|---:|
| Lesion-centred | NLL | {calibration['lesion_centered']['raw']['nll']:.4f} | {calibration['lesion_centered']['c03_operational']['nll']:.4f} |
| Lesion-centred | Brier | {calibration['lesion_centered']['raw']['brier']:.4f} | {calibration['lesion_centered']['c03_operational']['brier']:.4f} |
| Lesion-centred | ECE | {calibration['lesion_centered']['raw']['ece']:.4f} | {calibration['lesion_centered']['c03_operational']['ece']:.4f} |
| Background | ECE | {calibration['background']['raw']['ece']:.4f} | {calibration['background']['c03_operational']['ece']:.4f} |
| Whole image | ECE | {calibration['whole_image']['raw']['ece']:.4f} | {calibration['whole_image']['c03_operational']['ece']:.4f} |

C03 improves lesion NLL, background calibration, and whole-image calibration, but lesion Brier and lesion ECE worsen. Calibration therefore does not transfer uniformly to the lesion region.

## Reliability transfer

| Metric | Result | Gate | Outcome |
|---|---:|---:|---|
| Failure AUROC | {reliability['failure_detection_auroc']:.4f} | ≥0.75 | Pass |
| AUROC lower CI | {reliability['bootstrap_ci95']['failure_detection_auroc'][0]:.4f} | ≥0.70 | Pass |
| ECE | {reliability['ece']:.4f} | ≤0.05 | Fail |
| Brier skill | {reliability['brier_skill']:.4f} | >0 with lower CI >0 | Pass |
| Top-20% failure capture | {reliability['top20_risk']['failures_captured']}/{reliability['top20_risk']['failures_total']} ({100 * reliability['top20_risk']['failures_captured'] / reliability['top20_risk']['failures_total']:.1f}%) | ≥50% | Fail |

The top-20% gate is mathematically unattainable when 357/425 cases fail because 85 risk slots cannot contain half of 357 failures. The gate is retained and fails; it is not relaxed after seeing prevalence. At the fixed policy threshold, 385/425 cases require review, 340 failures are caught, 17 are missed, and total P01 cost is 555.

All five erased-target cases are `REVIEW_REQUIRED`, but only 1/5 enters the highest-risk 20%. Their mean reliability score is `{verification['target_not_representable']['mean_reliability']:.4f}` versus `{verification['target_not_representable']['representable_mean_reliability']:.4f}` for representable cases. This supports the required limitation: target representability can interact with prediction-derived reliability inputs even though R02 never reads a reference mask.

## Domain gap

- Model-input mean Wasserstein distance: `{gap['continuous_distribution_shift']['model_input_mean']['wasserstein']:.4f}`; fixed-histogram JS divergence: `{gap['continuous_distribution_shift']['model_input_mean']['jensen_shannon_divergence']:.4f}`.
- Frozen-encoder centroid shift: `{gap['encoder_representation_shift']['centroid_euclidean']:.4f}`.
- Reference foreground fraction: OpenNeuro `{gap['continuous_distribution_shift']['foreground_fraction']['external_mean']:.4f}` vs BRISC validation `{gap['continuous_distribution_shift']['foreground_fraction']['brisc_validation_mean']:.4f}`.
- Fragmented reference fraction: OpenNeuro `{gap['fragmented_reference_fraction']['openneuro']:.4f}` vs BRISC validation `{gap['fragmented_reference_fraction']['brisc_validation']:.4f}`.
- Reliability-score Wasserstein distance: `{gap['reliability_score_shift']['wasserstein']:.4f}`; policy review rate shifts from `{gap['policy_review_rate']['brisc_operational_reference']:.1%}` on BRISC development validation to `{gap['policy_review_rate']['openneuro']:.1%}` on X02.

The strong intensity and encoder shifts, smaller lesions, more fragmented references, and broad complete-miss pattern support a genuine domain-transport failure rather than a five-slice preprocessing artifact.

## Registered decision

Failed gates: {', '.join(f'`{name}`' for name in gate_failures)}.

H8 is rejected for X02. No threshold, gate, model, calibration mapping, or reliability model is changed. X02 must not be used for tuning under this protocol.

## Verification and limits

- Independent recomputation checked {verification['metrics_recomputed']} decision-driving values across 425 primary rows, 50 subjects, 624 volume slices, classifier aggregation, segmentation views, failure taxonomy, reliability bootstrap, policy counts, and all gates. Maximum absolute difference: `{verification['maximum_absolute_metric_difference']}`.
- Pixel-calibration and domain-gap standalone JSON artifacts match the completed result exactly. Per-pixel probability maps were not persisted, so the descriptive pixel-calibration aggregates cannot be independently regenerated without another inference run; no repeat inference was performed.
- BRISC development validation files used for frozen domain reference: 680. BRISC sealed-test files opened: 0.
- Neural retraining: 0. Calibrator refits: 0. Reliability refits: 0. Segmentation-threshold changes: 0. Policy-threshold changes: 0.
- This evaluates one retrospective, single-site, pituitary-only cohort. It does not establish prospective generalization, clinical validity, diagnostic safety, or suitability for patient care.
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = x02.read_json(config_path)
    if config["status"] != "AUDIT_COMPLETE":
        raise ValueError("X02 config state changed before finalization")
    config.update(
        {
            "status": "COMPLETE",
            "evaluation_outcome": "FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL",
            "amendment_15": "audit/preregistration_amendment_15.md",
            "aborted_inference_attempts": 1,
            "completed_inference_attempts": 1,
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    x02.write_json(config_path, config)

    decision_path = EXPERIMENT / "decision.md"
    with decision_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            "\n## Final external-evaluation decision\n\n"
            "**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`\n\n"
            "The independent OpenNeuro pituitary cohort failed classifier-calibration, strict pituitary Dice, "
            "case-reliability ECE, and top-20% failure-capture gates. Classification recall and reliability "
            "ranking AUROC passed, but they do not override the conjunctive failures. The five erased targets "
            "remain explicit zero-score cases and do not explain the broad failure. No gate is relaxed and X02 "
            "will not be used for tuning.\n"
        )

    registry_path = ROOT / "experiments/registry.json"
    registry = x02.read_json(registry_path)
    matches = [item for item in registry["experiments"] if item["id"] == "x02_openneuro_pituitary"]
    if len(matches) != 1 or matches[0]["status"] != "AUDIT_COMPLETE":
        raise ValueError("X02 registry state changed before finalization")
    matches[0]["status"] = "COMPLETE"
    x02.write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Reject X02 transport gates and retain independent external findings

- **Decision made:** Complete X02 as `FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL`. Reject H8 for this cohort without tuning on X02.
- **Triggering artifact:** `experiments/x02_openneuro_pituitary/artifacts/evaluation_result.json`, SHA256 `{x01.sha256(evaluation_path)}`; deterministic prediction panel SHA256 `{x01.sha256(ARTIFACTS / 'visual_audit_panel.png')}`.
- **Rationale:** Pituitary recall passed at `{classifier['pituitary_recall']:.6f}`, but classifier ECE was `{classifier['top_label_ece']:.6f}`. Strict all-425 subject-macro Dice was `{segmentation['subject_macro']['dice']['subject_macro_mean']:.6f}` against `0.75`. Reliability AUROC passed at `{reliability['failure_detection_auroc']:.6f}`, while ECE was `{reliability['ece']:.6f}` and top-20% captured `{reliability['top20_risk']['failures_captured']}/{reliability['top20_risk']['failures_total']}` failures. Complete misses accounted for `{failure['complete_miss']['count']}/{failure['count']}` failures.
- **Target representability:** The five Amendment-15 cases remain in the strict gate with zero scores. Representable-only Dice `{segmentation['representable_only_model_space']['dice']['subject_macro_mean']:.6f}` and native source-coordinate Dice `{segmentation['source_coordinate_all_425']['dice']['subject_macro_mean']:.6f}` show that these five cases do not explain the broad failure.
- **Verification:** Independent recomputation covered `{verification['metrics_recomputed']}` decision-driving values with maximum absolute difference `{verification['maximum_absolute_metric_difference']}`. Visual review found no systematic flip, transpose, offset, or source-mask alignment defect. No BRISC sealed-test image was opened. Neural retraining, calibration/reliability refit, segmentation-threshold changes, policy-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Drop the five nonrepresentable targets or report only the 420-slice sensitivity; rejected by Amendment 15. Tune preprocessing/calibration/reliability on X02; rejected because external data are evaluation-only. Relax the top-20% gate because failure prevalence makes it infeasible; rejected because gates remain fixed after results.
- **Who approved:** User approved Amendment 15 and one restart with fail-closed handling.
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
        ROOT / "audit/preregistration_amendment_15.md",
        ROOT / "scripts/evaluate_x02_openneuro.py",
        ROOT / "scripts/finalize_x02_openneuro.py",
        ROOT / "tests/test_evaluate_x02_openneuro.py",
        ROOT / "tests/test_finalize_x02_openneuro.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        EXPERIMENT / "result.json",
        decision_path,
    ]
    x02.write_json(ARTIFACTS / "experiment_manifest.json", manifest(artifact_paths + tracked_paths))


if __name__ == "__main__":
    main()
