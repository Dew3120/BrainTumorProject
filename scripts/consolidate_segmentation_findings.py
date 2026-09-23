"""Consolidate the verified segmentation audit without inference or retraining."""

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "20260923-072439-397263-full"
RUN = ROOT / "runs" / "segmentation" / RUN_NAME
FINDINGS_PATH = ROOT / "audit" / "segmentation_training_findings.json"
AGREEMENT_PATH = ROOT / "audit" / "classifier_segmenter_agreement.json"
UNCERTAINTY_PATH = ROOT / "audit" / "segmentation_uncertainty_signals.json"
MANIFEST_PATH = RUN / "segmentation_run_manifest.json"
FAILURE_CSV_PATH = RUN / "segmentation_failure_modes.csv"
RESULT_PATH = RUN / "result.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def verify_manifest(run, manifest):
    if manifest["run"] != RUN_NAME:
        raise ValueError("Run-manifest identity mismatch")
    expected = {}
    for item in manifest["files"]:
        relative = item["path"]
        if relative in expected:
            raise ValueError(f"Duplicate manifest entry: {relative}")
        expected[relative] = item

    actual = {
        path.relative_to(run).as_posix(): path
        for path in run.rglob("*")
        if path.is_file() and path.name != MANIFEST_PATH.name
    }
    if set(actual) != set(expected):
        raise ValueError("Run-folder membership differs from the reproducibility audit manifest")
    for relative, path in actual.items():
        item = expected[relative]
        if path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"Run artifact size mismatch: {relative}")
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Run artifact SHA256 mismatch: {relative}")


def assert_close(actual, expected, name, tolerance=1e-12):
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: {actual} != {expected}")


def summarize_failures(rows):
    if len(rows) != 83:
        raise ValueError(f"Expected 83 sub-0.50 cases, found {len(rows)}")
    if any(row["draft_split"] != "validation" for row in rows):
        raise ValueError("Failure-mode CSV contains a non-validation row")
    if any(float(row["dice"]) >= 0.50 for row in rows):
        raise ValueError("Failure-mode CSV contains Dice >= 0.50")

    category_counts = Counter(row["category"] for row in rows)
    expected_categories = {"glioma": 60, "meningioma": 8, "pituitary": 15}
    if dict(category_counts) != expected_categories:
        raise ValueError(f"Failure category counts changed: {dict(category_counts)}")

    fragmented = sum(row["fragmented_prediction"].lower() == "true" for row in rows)
    disconnected = sum(
        row["disconnected_false_positive"].lower() == "true" for row in rows
    )
    if fragmented != 70 or disconnected != 67:
        raise ValueError(
            "failure-mode analysis topology counts changed: "
            f"fragmented={fragmented}, disconnected={disconnected}"
        )
    return {
        "population": "83 BRISC internal-validation cases with hard-mask Dice below 0.50",
        "case_count": len(rows),
        "category_counts": expected_categories,
        "threshold_unchanged": 0.50,
        "fragmented_prediction": {
            "count": fragmented,
            "fraction": fragmented / len(rows),
            "reported_percent": 84.3,
        },
        "disconnected_false_positive": {
            "count": disconnected,
            "fraction": disconnected / len(rows),
            "reported_percent": 80.7,
        },
        "interpretation": (
            "Fragmented predictions and disconnected false-positive components are the "
            "dominant cross-category topology signals among weak cases."
        ),
    }


def verify_inputs(findings, result, agreement, uncertainty, manifest, failures):
    if findings["accepted_run"] != RUN_NAME:
        raise ValueError("Findings accepted-run mismatch")
    if result["test_evaluated"] or findings["data"]["test_evaluated"]:
        raise ValueError("A source artifact reports sealed-test evaluation")
    if manifest["constraints"] != {
        "retraining_performed": False,
        "threshold_changed": False,
        "test_images_accessed": 0,
        "calibration_work_started": False,
    }:
        raise ValueError("reproducibility audit constraints changed")
    for name, source in {
        "agreement": agreement,
        "uncertainty": uncertainty,
    }.items():
        if source["verification"]["test_images_accessed"] != 0:
            raise ValueError(f"{name} reports sealed-test access")

    metrics = result["best_validation_metrics"]
    original = findings["best_validation"]
    metric_pairs = {
        "image mean Dice": (metrics["val_dice_image_mean"], original["image_mean_dice"]),
        "image mean IoU": (metrics["val_iou_image_mean"], original["image_mean_iou"]),
        "macro Dice": (
            metrics["val_dice_macro_type_mean"],
            original["macro_type_mean_dice"],
        ),
        "glioma Dice": (metrics["val_dice_glioma"], original["dice_by_type"]["glioma"]),
        "meningioma Dice": (
            metrics["val_dice_meningioma"],
            original["dice_by_type"]["meningioma"],
        ),
        "pituitary Dice": (
            metrics["val_dice_pituitary"],
            original["dice_by_type"]["pituitary"],
        ),
    }
    for name, (actual, expected) in metric_pairs.items():
        assert_close(actual, expected, name)

    observations = agreement["observations"]
    if observations["segmenter_failure_count"] != len(failures):
        raise ValueError("failure-mode analysis and classifier/segmenter agreement audit failure counts disagree")
    if observations["segmenter_failures_with_correct_classifier"] != 78:
        raise ValueError("classifier/segmenter agreement audit correct-classifier failure count changed")
    if observations["zero_dice_with_correct_classifier_type"] != 16:
        raise ValueError("classifier/segmenter agreement audit zero-Dice agreement count changed")

    correlations = uncertainty["cross_model_correlations_all_680"]
    if correlations["population"] != 680:
        raise ValueError("uncertainty analysis correlation population changed")
    assert_close(
        correlations["spearman_classifier_confidence_vs_segmenter_dice"],
        0.32524900331879836,
        "classifier-confidence versus Dice correlation",
    )
    assert_close(
        correlations[
            "spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference"
        ],
        0.16603542526179857,
        "classifier-confidence versus inside-reference correlation",
    )


def build_consolidation(result, failures, agreement, uncertainty, manifest):
    metrics = result["best_validation_metrics"]
    observations = agreement["observations"]
    correlations = uncertainty["cross_model_correlations_all_680"]
    group_summaries = uncertainty["group_summaries"]
    positional = manifest["findings_carry_forward"]["zero_dice_position_check"]
    zero_position = positional["zero_dice"]

    return {
        "scope": (
            "Consolidated visual, failure-mode, agreement, uncertainty, and reproducibility evidence for the 680-image BRISC internal "
            "segmentation validation partition; not clinical validation"
        ),
        "overall_metrics": {
            "validation_images": 680,
            "image_mean_dice": metrics["val_dice_image_mean"],
            "image_mean_iou": metrics["val_iou_image_mean"],
            "macro_type_mean_dice": metrics["val_dice_macro_type_mean"],
            "reported_4dp": {
                "image_mean_dice": 0.7694,
                "image_mean_iou": 0.6724,
                "macro_type_mean_dice": 0.7627,
            },
        },
        "per_category_mean_dice": {
            "glioma": {
                "count": 204,
                "value": metrics["val_dice_glioma"],
                "reported_4dp": 0.6090,
            },
            "meningioma": {
                "count": 227,
                "value": metrics["val_dice_meningioma"],
                "reported_4dp": 0.8702,
            },
            "pituitary": {
                "count": 249,
                "value": metrics["val_dice_pituitary"],
                "reported_4dp": 0.8088,
            },
        },
        "failure_modes": summarize_failures(failures),
        "classifier_segmenter_agreement": {
            "segmenter_failure_definition": "hard-mask Dice below 0.50",
            "segmenter_failure_count": observations["segmenter_failure_count"],
            "failures_with_correct_classifier": {
                "count": observations["segmenter_failures_with_correct_classifier"],
                "denominator": observations["segmenter_failure_count"],
                "fraction": observations["segmenter_failures_with_correct_classifier"]
                / observations["segmenter_failure_count"],
            },
            "zero_dice_with_correct_classifier": {
                "count": observations["zero_dice_with_correct_classifier_type"],
                "denominator": observations["zero_dice_count"],
                "fraction": observations["zero_dice_with_correct_classifier_type"]
                / observations["zero_dice_count"],
            },
            "interpretation": (
                "Correct tumor-type classification does not imply reliable spatial "
                "segmentation."
            ),
        },
        "uncertainty_signals": {
            "finding": (
                "Zero-Dice cases are confident spatial errors rather than globally "
                "uncertain cases. Classifier confidence cannot serve as a segmentation-"
                "reliability proxy."
            ),
            "spearman_correlations_all_680": {
                "classifier_confidence_vs_segmenter_dice": correlations[
                    "spearman_classifier_confidence_vs_segmenter_dice"
                ],
                "classifier_confidence_vs_segmenter_inside_reference_probability": (
                    correlations[
                        "spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference"
                    ]
                ),
                "reported_4dp": {
                    "classifier_confidence_vs_segmenter_dice": 0.3252,
                    "classifier_confidence_vs_segmenter_inside_reference_probability": 0.1660,
                },
            },
            "tumor_type_confound": {
                "statement": (
                    "The extreme-group confidence comparison is not tumor-type-controlled."
                ),
                "zero_dice_group": group_summaries["zero_dice"]["category_counts"],
                "highest_dice_20_group": group_summaries["highest_dice_20"][
                    "category_counts"
                ],
                "key_imbalance": (
                    "12/17 zero-Dice cases are glioma; 15/20 highest-Dice cases are "
                    "meningioma."
                ),
            },
        },
        "falsified_hypotheses": [
            {
                "hypothesis": (
                    "Zero-Dice predictions show a consistent skull-base positional bias."
                ),
                "result": "NOT_CONFIRMED_BY_PREDECLARED_SUPPORT_RULE",
                "interpretation": (
                    "The errors are spatially heterogeneous, with only a modest lower-half "
                    "shift; the observed four-case visual cluster did not hold across all 17."
                ),
                "supporting_numbers": {
                    "zero_dice_cases": zero_position["count"],
                    "centroids_below_image_midline": zero_position[
                        "centroid_below_image_midline_count"
                    ],
                    "centroids_below_normalized_y_0_60": zero_position[
                        "centroid_below_normalized_y_0_60_count"
                    ],
                    "median_lower_third_predicted_pixel_fraction": zero_position[
                        "median_fraction_in_lower_third"
                    ],
                    "reported_median_lower_third_fraction": 0.008,
                },
                "predeclared_support_rule": positional["predeclared_support_rule"],
            }
        ],
        "reproducibility_posture": {
            "conclusion": "CONTROLLED_BUT_NOT_FULLY_GUARANTEED",
            "training_time_git_head": {
                "recorded": False,
                "source_equivalent_commit": manifest["git"][
                    "training_source_equivalent_commit"
                ],
                "explicit_limitation": (
                    "The exact training-time Git HEAD was not recorded. Commit 459c675 is "
                    "the source-equivalent commit, not a proven launch-time commit."
                ),
            },
            "controls": manifest["determinism_audit"]["controls"],
            "current_runtime_probe": manifest["determinism_audit"][
                "strict_current_runtime_probe"
            ],
            "specific_unenforced_or_environment_sensitive_operations": manifest[
                "determinism_audit"
            ]["specific_unenforced_or_environment_sensitive_operations"],
            "checkpoint_reload": manifest["checkpoint_reload_verification"],
        },
        "what_segmentation_validation_does_not_prove": [
            {
                "claim": "clinical validity or safety",
                "status": "NOT_ESTABLISHED",
                "reason": "No prospective clinical evaluation or clinician-validated deployment study was performed.",
            },
            {
                "claim": "cross-dataset or real-world generalization",
                "status": "NOT_ESTABLISHED",
                "reason": "All reported segmentation metrics are from BRISC internal validation.",
            },
            {
                "claim": "patient-independent performance",
                "status": "NOT_ESTABLISHED",
                "reason": "BRISC patient independence is unverified.",
            },
            {
                "claim": "resolved glioma segmentation",
                "status": "NOT_ESTABLISHED",
                "reason": (
                    "Glioma mean Dice is quantified at 0.6090, with 60/204 cases below "
                    "Dice 0.50 and 12 zero-Dice cases; the limitation remains unresolved."
                ),
            },
            {
                "claim": "sealed-test performance",
                "status": "NOT_EVALUATED",
                "reason": "The sealed BRISC test partition was not accessed.",
            },
        ],
        "constraints_verified": {
            "retraining_performed": False,
            "threshold_changed": False,
            "test_images_accessed": 0,
            "calibration_work_started": False,
        },
    }


def main():
    findings = load_json(FINDINGS_PATH)
    result = load_json(RESULT_PATH)
    agreement = load_json(AGREEMENT_PATH)
    uncertainty = load_json(UNCERTAINTY_PATH)
    manifest = load_json(MANIFEST_PATH)
    failures = load_csv(FAILURE_CSV_PATH)

    verify_manifest(RUN, manifest)
    verify_inputs(findings, result, agreement, uncertainty, manifest, failures)
    consolidation = build_consolidation(
        result, failures, agreement, uncertainty, manifest
    )

    findings["schema_version"] = 2
    findings["audit_status"] = "SEGMENTATION_FINDINGS_CONSOLIDATED_AWAITING_USER_REVIEW"
    findings["segmentation_validation_consolidation"] = consolidation
    findings["source_artifacts"] = {
        "result_json": {"path": str(RESULT_PATH), "sha256": sha256(RESULT_PATH)},
        "failure_modes_csv": {
            "path": str(FAILURE_CSV_PATH),
            "sha256": sha256(FAILURE_CSV_PATH),
        },
        "classifier_segmenter_agreement_json": {
            "path": str(AGREEMENT_PATH),
            "sha256": sha256(AGREEMENT_PATH),
        },
        "uncertainty_signals_json": {
            "path": str(UNCERTAINTY_PATH),
            "sha256": sha256(UNCERTAINTY_PATH),
        },
        "segmentation_run_manifest_json": {
            "path": str(MANIFEST_PATH),
            "sha256": sha256(MANIFEST_PATH),
        },
    }
    findings["limitations"] = [
        item["reason"] for item in consolidation["what_segmentation_validation_does_not_prove"]
    ]
    findings["calibration_work_started"] = False
    FINDINGS_PATH.write_text(
        json.dumps(findings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    config = manifest["config"]
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_artifact(str(FINDINGS_PATH))

    print(f"Consolidated segmentation findings: {FINDINGS_PATH}")
    print(f"SHA256: {sha256(FINDINGS_PATH)}")
    print("No retraining, threshold change, sealed-test access, or calibration work occurred.")


if __name__ == "__main__":
    main()
