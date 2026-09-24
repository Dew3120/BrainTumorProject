"""Consolidate frozen confidence experiments into a fail-closed completion report."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "audit"
SOURCE_PATHS = {
    "classifier_multiclass": AUDIT / "classifier_calibration.json",
    "classifier_binary": AUDIT / "classifier_binary_calibration.json",
    "segmenter_pixel": AUDIT / "segmenter_pixel_calibration.json",
    "segmenter_case": AUDIT / "segmenter_case_reliability.json",
}
FINDINGS_PATH = AUDIT / "confidence_calibration_findings.json"
REPORT_PATH = AUDIT / "confidence_calibration_completion.md"
CASE_RELIABILITY_SCRIPT = (
    ROOT / "scripts" / "snapshots" / "fit_segmenter_case_reliability_v1.py"
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_close(actual, expected, name, tolerance=1e-12):
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: {actual} != {expected}")


def load_sources():
    return {name: load_json(path) for name, path in SOURCE_PATHS.items()}


def verify_constraints(sources):
    multiclass = sources["classifier_multiclass"]
    binary = sources["classifier_binary"]
    pixel = sources["segmenter_pixel"]
    case = sources["segmenter_case"]

    verifications = {
        "classifier multiclass": multiclass["verification"],
        "classifier binary": binary["verification"],
        "segmenter pixel": pixel["verification"],
        "segmenter case": case["verification"],
    }
    for name, verification in verifications.items():
        if verification["test_images_accessed"] != 0:
            raise ValueError(f"{name} reports sealed-test image access")
        if verification["test_predictions_accessed"] != 0:
            raise ValueError(f"{name} reports sealed-test prediction access")
        if verification["threshold_changed"]:
            raise ValueError(f"{name} reports a threshold change")

    if multiclass["verification"]["model_retrained"]:
        raise ValueError("classifier multiclass reports model retraining")
    if binary["verification"]["model_retrained"]:
        raise ValueError("classifier binary reports model retraining")
    if pixel["verification"]["model_retrained"]:
        raise ValueError("segmenter pixel reports model retraining")
    if case["verification"]["base_models_retrained"]:
        raise ValueError("segmenter case reports base-model retraining")

    for name, source in {
        "classifier multiclass": multiclass,
        "segmenter pixel": pixel,
        "segmenter case": case,
    }.items():
        if source["verification"].get("application_integration_performed", False):
            raise ValueError(f"{name} reports application integration")

    split_hashes = {
        source["verification"]["split_sha256"] for source in sources.values()
    }
    if len(split_hashes) != 1:
        raise ValueError("calibration artifacts use different frozen split hashes")

    reused_fold_hashes = {
        binary["verification"]["reused_fold_assignments_sha256"],
        pixel["verification"]["reused_fold_assignments_sha256"],
        case["verification"]["reused_fold_assignments_sha256"],
    }
    if len(reused_fold_hashes) != 1:
        raise ValueError("calibration artifacts use different fold assignments")


def verify_decisions(sources):
    multiclass = sources["classifier_multiclass"]
    binary = sources["classifier_binary"]
    pixel = sources["segmenter_pixel"]
    case = sources["segmenter_case"]

    if multiclass["gate"]["status"] != "FAIL":
        raise ValueError("multiclass classifier calibration decision changed")
    if binary["gate"]["status"] != "FAIL":
        raise ValueError("binary classifier calibration decision changed")
    if pixel["gate"]["status"] != "FAIL":
        raise ValueError("segmenter pixel calibration decision changed")
    if case["gate"]["status"] != "PASS":
        raise ValueError("segmenter case reliability decision changed")
    if not all(case["gate"]["checks"].values()):
        raise ValueError("segmenter case reliability has a failed gate check")

    if multiclass["decision_rules"]["temperature_accepted_for_application"]:
        raise ValueError("rejected multiclass temperature is marked accepted")
    if binary["decision_rules"]["temperature_accepted_for_application"]:
        raise ValueError("rejected binary temperature is marked accepted")
    if pixel["gate"]["checks"]["lesion_centered_ece_at_most_0.05"]:
        raise ValueError("segmenter foreground ECE gate unexpectedly passes")
    if pixel["gate"]["checks"]["background_nll_improved"]:
        raise ValueError("segmenter background NLL gate unexpectedly passes")
    if not pixel["gate"]["checks"]["oof_hard_masks_bitwise_identical"]:
        raise ValueError("segmenter OOF hard-mask identity check failed")
    if not pixel["gate"]["checks"]["final_temperature_hard_masks_bitwise_identical"]:
        raise ValueError("segmenter final-temperature hard-mask identity check failed")


def verify_case_reliability(case):
    target = case["target"]
    metrics = case["metrics"]
    categories = metrics["per_category"]
    coefficients = case["coefficients"]["features"]

    if target["success_cases"] + target["failure_cases"] != case["validation_cases"]:
        raise ValueError(
            "case reliability target counts do not sum to validation count"
        )
    if target["failure_cases"] != 83 or target["zero_dice_cases"] != 17:
        raise ValueError("case reliability failure population changed")
    if sum(item["count"] for item in categories.values()) != 680:
        raise ValueError("case reliability category counts do not sum to 680")
    if sum(item["failures"] for item in categories.values()) != 83:
        raise ValueError("case reliability category failures do not sum to 83")
    if sum(item["zero_dice"] for item in categories.values()) != 17:
        raise ValueError("case reliability category zero-Dice cases do not sum to 17")

    top20 = metrics["top20_risk"]
    if top20["case_count"] != 136 or top20["zero_dice_captured"] != 14:
        raise ValueError("case reliability catastrophic-failure capture changed")

    boundary = coefficients["boundary_entropy_mean"]
    nonzero = {
        name: abs(item["mean"])
        for name, item in coefficients.items()
        if item["mean"] != 0.0
    }
    if max(nonzero, key=nonzero.get) != "boundary_entropy_mean":
        raise ValueError("boundary entropy is no longer the strongest mean coefficient")
    if not boundary["all_folds_same_sign"] or boundary["negative_folds"] != 5:
        raise ValueError("boundary entropy sign agreement changed")

    high_confidence = coefficients["high_confidence_foreground_fraction"]
    if (
        high_confidence["mean"] >= 0.0
        or not high_confidence["all_folds_same_sign"]
        or high_confidence["negative_folds"] != 5
    ):
        raise ValueError("high-confidence foreground association changed")

    glioma = categories["glioma"]
    comparison = [categories["meningioma"], categories["pituitary"]]
    if not all(
        glioma["failure_detection_auroc"] < item["failure_detection_auroc"]
        for item in comparison
    ):
        raise ValueError("glioma is no longer the weakest subgroup by AUROC")
    if not all(glioma["ece"] > item["ece"] for item in comparison):
        raise ValueError("glioma is no longer the weakest subgroup by ECE")
    if sha256(CASE_RELIABILITY_SCRIPT) != case["verification"]["script_sha256"]:
        raise ValueError("case reliability source script hash changed")


def verify_inputs(sources):
    verify_constraints(sources)
    verify_decisions(sources)
    verify_case_reliability(sources["segmenter_case"])

    classifier = sources["classifier_multiclass"]
    binary = sources["classifier_binary"]
    pixel = sources["segmenter_pixel"]
    case = sources["segmenter_case"]

    assert_close(
        classifier["metrics"]["raw"]["top_label_ece"],
        0.0083915474361794,
        "raw multiclass ECE",
    )
    assert_close(
        binary["metrics"]["raw"]["ece"],
        0.0020682140151788257,
        "raw binary ECE",
    )
    assert_close(
        pixel["metrics"]["aggregate"]["lesion_centered"]["raw"]["pixel_ece"],
        0.07551320872995725,
        "raw segmenter foreground ECE",
    )
    assert_close(
        case["metrics"]["failure_detection_auroc"],
        0.9409900910173357,
        "case reliability AUROC",
    )


def metric_view(region):
    return {
        "nll": region["pixel_nll"],
        "brier": region["pixel_brier"],
        "ece": region["pixel_ece"],
        "mce": region["pixel_mce"],
    }


def build_findings(sources):
    multiclass = sources["classifier_multiclass"]
    binary = sources["classifier_binary"]
    pixel = sources["segmenter_pixel"]
    case = sources["segmenter_case"]

    pixel_foreground = pixel["metrics"]["aggregate"]["lesion_centered"]
    pixel_background = pixel["metrics"]["aggregate"]["background"]
    case_metrics = case["metrics"]
    categories = case_metrics["per_category"]
    boundary = case["coefficients"]["features"]["boundary_entropy_mean"]
    high_confidence = case["coefficients"]["features"][
        "high_confidence_foreground_fraction"
    ]
    source_hashes = {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in SOURCE_PATHS.items()
    }
    source_hashes["segmenter_case_script"] = {
        "path": str(CASE_RELIABILITY_SCRIPT),
        "sha256": sha256(CASE_RELIABILITY_SCRIPT),
    }

    return {
        "schema_version": 2,
        "status": "CONFIDENCE_CALIBRATION_COMPLETE",
        "scope": (
            "Frozen-model confidence evaluation on BRISC internal validation; "
            "not clinical validation"
        ),
        "component_decisions": {
            "classifier": {
                "status": "CLOSED_RAW_PROBABILITIES_ACCEPTED",
                "four_class_temperature_gate": multiclass["gate"]["status"],
                "binary_temperature_gate": binary["gate"]["status"],
                "raw_multiclass_ece": multiclass["metrics"]["raw"]["top_label_ece"],
                "raw_binary_ece": binary["metrics"]["raw"]["ece"],
                "finding": (
                    "The raw four-class and binary probabilities were already well "
                    "calibrated internally. Neither temperature experiment produced a "
                    "statistically stable accepted improvement."
                ),
                "accepted_probability_source": "raw classifier probabilities",
                "further_classifier_calibration_allowed": False,
            },
            "segmenter_pixel": {
                "status": "CLOSED_RAW_PROBABILITIES_ACCEPTED",
                "temperature_gate": pixel["gate"]["status"],
                "raw_foreground": metric_view(pixel_foreground["raw"]),
                "oof_calibrated_foreground": metric_view(
                    pixel_foreground["oof_calibrated"]
                ),
                "raw_background": metric_view(pixel_background["raw"]),
                "oof_calibrated_background": metric_view(
                    pixel_background["oof_calibrated"]
                ),
                "foreground_brier_change_ci95": [
                    pixel["metrics"]["brier_paired_similarity_group_bootstrap"][
                        "lesion_centered"
                    ]["ci95_lower"],
                    pixel["metrics"]["brier_paired_similarity_group_bootstrap"][
                        "lesion_centered"
                    ]["ci95_upper"],
                ],
                "gate_failures": {
                    "foreground_ece_excess_over_0_05": (
                        pixel_foreground["oof_calibrated"]["pixel_ece"] - 0.05
                    ),
                    "background_nll_increase": (
                        pixel_background["oof_calibrated"]["pixel_nll"]
                        - pixel_background["raw"]["pixel_nll"]
                    ),
                },
                "finding": (
                    "Temperature scaling substantially improved foreground reliability "
                    "but missed the fixed foreground-ECE gate and slightly worsened "
                    "background NLL. The gate was not relaxed."
                ),
                "accepted_probability_source": "raw segmenter probabilities",
                "further_pixel_calibration_variants_allowed": False,
            },
            "segmenter_case_reliability": {
                "status": "ACCEPTED_FOR_APPLICATION_DESIGN_HANDOFF",
                "gate": case["gate"]["status"],
                "target": case["target"]["model_output"],
                "feature_provenance": {
                    "source": case["model"]["feature_source"],
                    "features": case["model"]["features"],
                    "reference_mask_use": case["target"]["reference_mask_use"],
                    "reference_derived_features": False,
                },
                "reported_scores": case["cross_fitting"]["reported_scores"],
                "failure_detection_auroc": case_metrics["failure_detection_auroc"],
                "auroc_ci95": [
                    case_metrics["bootstrap"]["auroc"]["ci95_lower"],
                    case_metrics["bootstrap"]["auroc"]["ci95_upper"],
                ],
                "zero_dice_capture_top20_risk": {
                    "captured": case_metrics["top20_risk"]["zero_dice_captured"],
                    "total": case_metrics["top20_risk"]["zero_dice_total"],
                    "top_risk_cases": case_metrics["top20_risk"]["case_count"],
                },
                "calibration": {
                    "ece": case_metrics["reliability"]["ece"],
                    "mce": case_metrics["reliability"]["mce"],
                    "brier": case_metrics["reliability"]["brier"],
                    "prevalence_baseline_brier": case_metrics["reliability"][
                        "prevalence_baseline_brier"
                    ],
                    "brier_skill": case_metrics["reliability"]["brier_skill"],
                    "brier_skill_ci95": [
                        case_metrics["bootstrap"]["brier_skill"]["ci95_lower"],
                        case_metrics["bootstrap"]["brier_skill"]["ci95_upper"],
                    ],
                    "brier_skill_definition": (
                        "1 - model Brier / prevalence-baseline Brier"
                    ),
                    "baseline_definition": (
                        "Each held-out fold receives a constant reliability equal to "
                        "one minus the failure prevalence in that fold's training data."
                    ),
                    "bootstrap": {
                        "repetitions": 2000,
                        "seed": 42,
                        "unit": (
                            "similarity groups resampled with replacement, retaining all "
                            "member cases"
                        ),
                    },
                },
                "per_category": categories,
                "interpretable_signals": {
                    "strongest_mean_standardized_coefficient": {
                        "feature": "boundary_entropy_mean",
                        "mean": boundary["mean"],
                        "sample_sd": boundary["sample_sd"],
                        "negative_folds": boundary["negative_folds"],
                        "interpretation": (
                            "Higher boundary entropy is the strongest observed predictor "
                            "of segmentation failure."
                        ),
                    },
                    "confident_wrong_signal": {
                        "feature": "high_confidence_foreground_fraction",
                        "mean": high_confidence["mean"],
                        "sample_sd": high_confidence["sample_sd"],
                        "negative_folds": high_confidence["negative_folds"],
                        "interpretation": (
                            "A larger high-confidence foreground fraction can accompany "
                            "lower reliability, operationalizing the confident-spatial-"
                            "error finding."
                        ),
                    },
                    "causal_interpretation_allowed": False,
                    "limitation": (
                        "The standardized coefficients are descriptive associations. "
                        "Features are correlated and the sample contains 680 cases."
                    ),
                },
                "final_all_validation_model_role": (
                    "Proposed operational model for later integration only; it was not "
                    "used for reported metrics."
                ),
                "application_integration_performed": False,
            },
        },
        "cross_cutting_finding": {
            "title": "Glioma remains the weakest subgroup",
            "statement": (
                "The aggregate case-reliability pass is not uniform across tumor types. "
                "Glioma drives most segmentation failures and has the weakest reliability "
                "discrimination and calibration."
            ),
            "aggregate": {
                "failure_detection_auroc": case_metrics["failure_detection_auroc"],
                "reliability_ece": case_metrics["reliability"]["ece"],
                "failures": case["target"]["failure_cases"],
            },
            "glioma": {
                "failure_detection_auroc": categories["glioma"][
                    "failure_detection_auroc"
                ],
                "reliability_ece": categories["glioma"]["ece"],
                "brier_skill": categories["glioma"]["brier_skill"],
                "failures": categories["glioma"]["failures"],
                "validation_cases": categories["glioma"]["count"],
                "zero_dice_cases": categories["glioma"]["zero_dice"],
            },
            "qualification": (
                "This does not invalidate the aggregate gate pass. It prevents claiming "
                "uniform reliability and requires explicit glioma caution in later use."
            ),
            "subgroup_metrics_role": (
                "Descriptive only; no per-category confidence intervals or predeclared "
                "subgroup acceptance gates were used."
            ),
            "resolved": False,
        },
        "what_this_work_establishes": [
            (
                "The frozen classifier's raw probabilities are acceptably calibrated on "
                "the BRISC internal-validation partition under the declared gates."
            ),
            (
                "Pixel temperature scaling improved several segmenter calibration metrics "
                "but did not satisfy the full fixed gate, so it was rejected."
            ),
            (
                "Prediction-derived case features can rank Dice-below-0.50 failures in "
                "similarity-aware out-of-fold evaluation on these 680 validation cases."
            ),
        ],
        "what_this_work_does_not_establish": [
            "Clinical validity, safety, diagnosis, or autonomous use.",
            "Patient-independent, cross-dataset, scanner-independent, or prospective performance.",
            "Uniform reliability across tumor types; glioma remains materially weaker.",
            (
                "Guaranteed catastrophic-failure capture; 3/17 zero-Dice and 15/83 "
                "Dice-below-0.50 cases remain outside the top-risk quintile."
            ),
            "Acceptance of calibrated segmenter pixel probabilities; raw pixels remain in force.",
            (
                "Prospective performance of the final all-validation case model; reported "
                "metrics are out-of-fold."
            ),
            "Causal meaning for any reliability-model coefficient.",
        ],
        "application_design_handoff": {
            "eligible_input": "case-level reliability score P(Dice >= 0.50)",
            "allowed_next_step": (
                "Design and validate the review policy that maps the score to display, "
                "warning, or required-review states."
            ),
            "required_qualification": (
                "The interface must preserve the glioma subgroup warning and must not "
                "present the score as clinical certainty."
            ),
            "started": False,
        },
        "constraints_verified": {
            "base_models_retrained": False,
            "test_images_accessed": 0,
            "test_predictions_accessed": 0,
            "threshold": 0.50,
            "threshold_changed": False,
            "application_integration_performed": False,
            "gate_relaxed": False,
        },
        "source_artifacts": source_hashes,
    }


def build_report(findings):
    decisions = findings["component_decisions"]
    classifier = decisions["classifier"]
    pixel = decisions["segmenter_pixel"]
    case = decisions["segmenter_case_reliability"]
    glioma = findings["cross_cutting_finding"]
    aggregate = glioma["aggregate"]
    subgroup = glioma["glioma"]
    boundary = case["interpretable_signals"]["strongest_mean_standardized_coefficient"]
    high_confidence = case["interpretable_signals"]["confident_wrong_signal"]
    foreground_raw = pixel["raw_foreground"]
    foreground_oof = pixel["oof_calibrated_foreground"]
    gate_failures = pixel["gate_failures"]
    calibration = case["calibration"]

    established = "\n".join(
        f"- {item}" for item in findings["what_this_work_establishes"]
    )
    not_established = "\n".join(
        f"- {item}" for item in findings["what_this_work_does_not_establish"]
    )

    return f"""# Confidence calibration completion report

All three frozen-model confidence studies are closed. The classifier keeps its raw probabilities, segmenter pixel temperature scaling is rejected under the unchanged gate, and the case-level segmenter reliability model passes for later application design. Reported calibrated and reliability metrics are out-of-fold; no test data was accessed.

| Component | Result | Decisive evidence | Accepted output |
|---|---|---|---|
| Classifier confidence | Closed | Raw four-class ECE `{classifier["raw_multiclass_ece"]:.4f}` and binary ECE `{classifier["raw_binary_ece"]:.4f}`; both temperature gates failed to show an accepted stable improvement | Raw classifier probabilities |
| Segmenter pixel confidence | Closed | Foreground ECE improved `{foreground_raw["ece"]:.4f}` → `{foreground_oof["ece"]:.4f}`, but remained `{gate_failures["foreground_ece_excess_over_0_05"]:.4f}` above the `0.05` gate; background NLL worsened by `{gate_failures["background_nll_increase"]:.4f}` | Raw segmenter probabilities |
| Segmenter case reliability | **Pass** | AUROC `{case["failure_detection_auroc"]:.4f}` (95% CI `{case["auroc_ci95"][0]:.4f}`–`{case["auroc_ci95"][1]:.4f}`); top-risk 20% captured `{case["zero_dice_capture_top20_risk"]["captured"]}/{case["zero_dice_capture_top20_risk"]["total"]}` zero-Dice cases; ECE `{calibration["ece"]:.4f}`; Brier skill `{calibration["brier_skill"]:.4f}` | Reliability score eligible for later review-policy design |

## Findings that matter

Boundary entropy is the strongest observed failure signal: mean standardized coefficient `{boundary["mean"]:.4f}` with a negative sign in `{boundary["negative_folds"]}/5` folds. High-confidence foreground fraction is also negatively associated with reliability (mean `{high_confidence["mean"]:.4f}`, negative in `{high_confidence["negative_folds"]}/5` folds). This supports the earlier finding that spatial failures can be confident. These coefficients are descriptive associations, not causal effects; features are correlated and the study contains 680 cases.

**Glioma remains the weakest subgroup.** The aggregate AUROC of `{aggregate["failure_detection_auroc"]:.4f}` is not uniform. Glioma AUROC is `{subgroup["failure_detection_auroc"]:.4f}`, reliability ECE is `{subgroup["reliability_ece"]:.4f}` versus `{aggregate["reliability_ece"]:.4f}` overall, and glioma contributes `{subgroup["failures"]}/{aggregate["failures"]}` Dice-below-0.50 failures plus `{subgroup["zero_dice_cases"]}` zero-Dice cases. The aggregate pass remains valid under its declared gate, but it does not establish subgroup equivalence. Per-category metrics are descriptive and have no confidence intervals. Later use must keep this qualification visible.

## What this work establishes

{established}

## What this work does not establish

{not_established}

## Handoff

The only new confidence output eligible for the next application stage is the case-level score `P(Dice ≥ 0.50)`. Its features come only from the raw prediction map and unchanged `0.50` hard mask; reference masks created and evaluated the development target but supplied no features. The final model fitted on all validation cases is a proposed operational model only and did not produce the reported metrics. The next stage may design and validate how this score maps to display, warning, or required-review states. It must retain the glioma warning and must not describe the score as clinical certainty.

No base model was retrained, no test image or prediction was accessed, the segmentation threshold remained `0.50`, no gate was relaxed, and no application integration was performed.
"""


def main():
    sources = load_sources()
    verify_inputs(sources)
    findings = build_findings(sources)
    report = build_report(findings)
    FINDINGS_PATH.write_text(
        json.dumps(findings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {FINDINGS_PATH}")
    print(f"Wrote {REPORT_PATH}")
    print("Confidence work is complete; application integration has not started.")


if __name__ == "__main__":
    main()
