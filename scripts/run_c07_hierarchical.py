"""Run and close C07 nested hierarchical pixel calibration."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from run_c05_reliability_weighted import load_reliability_scores
from run_c06_per_category import TUMOR_ROUTES
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c07_hierarchical"
ARTIFACTS = EXPERIMENT / "artifacts"


def load_predicted_routes(config, rows):
    path = ROOT / config["classifier_predictions_path"]
    if common.sha256(path) != config["classifier_predictions_sha256"]:
        raise ValueError("Classifier prediction hash mismatch")
    source = {row["path"]: row for row in common.read_csv(path)}
    routes = []
    for row in rows:
        candidate = source.get(row["path"])
        if candidate is None or candidate["label"] != row["label"]:
            raise ValueError(f"Classifier routing mismatch: {row['path']}")
        routes.append(candidate["predicted"])
    routes = np.asarray(routes)
    if set(routes) != {*TUMOR_ROUTES, "no_tumor"}:
        raise ValueError("Unexpected classifier route set")
    if int(np.sum(routes == "no_tumor")) != 1:
        raise ValueError("Registered single no_tumor fallback count changed")
    return routes


def fit_scalar(logits, targets, regions, indices, region_name, case_weights):
    """Fit bounded inverse temperature by convex Newton updates."""
    x_parts = []
    y_parts = []
    weight_parts = []
    selected_case_weights = np.asarray(case_weights[indices], dtype=np.float64)
    selected_case_weights /= selected_case_weights.sum()
    for case_weight, index in zip(selected_case_weights, indices, strict=True):
        mask = common.region_mask(regions, index, region_name)
        count = int(mask.sum())
        x_parts.append(np.asarray(logits[index][mask], dtype=np.float64))
        y_parts.append(np.asarray(targets[index][mask], dtype=np.float64))
        weight_parts.append(np.full(count, case_weight / count, dtype=np.float64))
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    weight = np.concatenate(weight_parts)
    inverse_temperature = 1.0
    lower, upper = 0.05, 20.0
    for _ in range(50):
        transformed = inverse_temperature * x
        probability = expit(transformed)
        gradient = float(np.sum(weight * (probability - y) * x))
        hessian = float(np.sum(weight * probability * (1 - probability) * x * x))
        if hessian <= 0 or not np.isfinite(hessian):
            raise RuntimeError("Invalid scalar-temperature Hessian")
        updated = float(np.clip(inverse_temperature - gradient / hessian, lower, upper))
        if abs(updated - inverse_temperature) <= 1e-10:
            inverse_temperature = updated
            break
        inverse_temperature = updated
    transformed = inverse_temperature * x
    objective = float(
        np.sum(weight * (np.logaddexp(0.0, transformed) - y * transformed))
    )
    return {
        "temperature": 1.0 / inverse_temperature,
        "inverse_temperature": inverse_temperature,
        "fit_images": len(indices),
        "fit_pixels": len(x),
        "fit_objective": objective,
    }


def fit_candidate(
    candidate, specification, logits, targets, regions, routes, reliability, indices
):
    region_names = (
        ("lesion_centered", "background")
        if specification["region_split"]
        else ("whole_image",)
    )
    case_weights = reliability if specification["reliability_weighted"] else np.ones(len(logits))
    model = {"candidate": candidate, "specification": specification, "cells": {}}
    route_names = TUMOR_ROUTES if specification["category_routed"] else ()
    for region_name in region_names:
        model["cells"][f"all::{region_name}"] = fit_scalar(
            logits, targets, regions, indices, region_name, case_weights
        )
        for route in route_names:
            route_indices = indices[routes[indices] == route]
            if len(route_indices) < 2:
                raise ValueError(f"Insufficient cases for {candidate} {route}")
            model["cells"][f"{route}::{region_name}"] = fit_scalar(
                logits, targets, regions, route_indices, region_name, case_weights
            )
    return model


def apply_candidate(model, logits, regions, routes, indices):
    specification = model["specification"]
    output = np.empty_like(logits[indices])
    for output_index, image_index in enumerate(indices):
        route = (
            routes[image_index]
            if specification["category_routed"]
            and routes[image_index] in TUMOR_ROUTES
            else "all"
        )
        if specification["region_split"]:
            lesion = regions[image_index]
            for region_name, mask in (
                ("lesion_centered", lesion),
                ("background", ~lesion),
            ):
                parameter = model["cells"].get(
                    f"{route}::{region_name}", model["cells"][f"all::{region_name}"]
                )
                output[output_index][mask] = (
                    logits[image_index][mask] / parameter["temperature"]
                )
        else:
            parameter = model["cells"].get(
                f"{route}::whole_image", model["cells"]["all::whole_image"]
            )
            output[output_index] = logits[image_index] / parameter["temperature"]
    return output, expit(output)


def inner_candidate_metrics(
    candidate,
    specification,
    outer_training_folds,
    assignment,
    logits,
    targets,
    regions,
    routes,
    reliability,
):
    population = np.flatnonzero(np.isin(assignment, outer_training_folds))
    oof_logits = np.empty_like(logits)
    oof_probability = np.empty_like(logits)
    fold_models = []
    for inner_fold in outer_training_folds:
        evaluation = np.flatnonzero(assignment == inner_fold)
        fitting = population[assignment[population] != inner_fold]
        model = fit_candidate(
            candidate,
            specification,
            logits,
            targets,
            regions,
            routes,
            reliability,
            fitting,
        )
        transformed, probability = apply_candidate(
            model, logits, regions, routes, evaluation
        )
        oof_logits[evaluation] = transformed
        oof_probability[evaluation] = probability
        fold_models.append(
            {
                "inner_evaluation_fold": int(inner_fold) + 1,
                "temperatures": {
                    key: value["temperature"] for key, value in model["cells"].items()
                },
            }
        )
    metrics, _, _ = common.region_metrics(
        oof_probability,
        oof_logits,
        targets,
        regions,
        population,
        "lesion_centered",
    )
    return metrics, fold_models


def select_candidate(
    outer_training_folds,
    candidates,
    assignment,
    logits,
    targets,
    regions,
    routes,
    reliability,
):
    comparisons = []
    details = {}
    for candidate in sorted(candidates):
        metrics, fold_models = inner_candidate_metrics(
            candidate,
            candidates[candidate],
            outer_training_folds,
            assignment,
            logits,
            targets,
            regions,
            routes,
            reliability,
        )
        comparisons.append(
            {
                "candidate": candidate,
                "pixel_ece": metrics["pixel_ece"],
                "pixel_nll": metrics["pixel_nll"],
                "pixel_brier": metrics["pixel_brier"],
            }
        )
        details[candidate] = fold_models
    comparisons.sort(
        key=lambda item: (
            item["pixel_ece"],
            item["pixel_nll"],
            item["pixel_brier"],
            item["candidate"],
        )
    )
    return comparisons[0]["candidate"], comparisons, details


def compact(metrics):
    return {
        key: metrics[key]
        for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
    }


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C07 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    routes = load_predicted_routes(config, rows)
    reliability_config = {
        "reliability_score_path": config["reliability_score_path"],
        "reliability_score_sha256": config["reliability_score_sha256"],
        "reliability_score_column": "oof_reliability_score",
    }
    reliability = load_reliability_scores(reliability_config, rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    regions = common.lesion_regions(targets)
    candidates = config["candidates"]
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    outer_selections = []
    selection_rows = []
    for outer_fold in range(5):
        training_folds = [fold for fold in range(5) if fold != outer_fold]
        selected, comparisons, inner_details = select_candidate(
            training_folds,
            candidates,
            assignment,
            raw_logits,
            targets,
            regions,
            routes,
            reliability,
        )
        fit_indices = np.flatnonzero(assignment != outer_fold)
        evaluation_indices = np.flatnonzero(assignment == outer_fold)
        model = fit_candidate(
            selected,
            candidates[selected],
            raw_logits,
            targets,
            regions,
            routes,
            reliability,
            fit_indices,
        )
        transformed, probability = apply_candidate(
            model, raw_logits, regions, routes, evaluation_indices
        )
        oof_logits[evaluation_indices] = transformed
        oof_probability[evaluation_indices] = probability
        outer_selections.append(
            {
                "outer_fold": outer_fold + 1,
                "selected_candidate": selected,
                "selected_is_oracle": candidates[selected]["region_split"],
                "fit_temperatures": {
                    key: value["temperature"] for key, value in model["cells"].items()
                },
                "inner_comparisons": comparisons,
                "inner_fold_models": inner_details,
            }
        )
        for item in comparisons:
            selection_rows.append(
                {
                    "outer_fold": outer_fold + 1,
                    "candidate": item["candidate"],
                    "selected": item["candidate"] == selected,
                    "inner_lesion_centered_ece": item["pixel_ece"],
                    "inner_lesion_centered_nll": item["pixel_nll"],
                    "inner_lesion_centered_brier": item["pixel_brier"],
                }
            )
    all_folds = list(range(5))
    final_candidate, final_comparisons, final_inner_details = select_candidate(
        all_folds,
        candidates,
        assignment,
        raw_logits,
        targets,
        regions,
        routes,
        reliability,
    )
    all_indices = np.arange(len(rows))
    final_model = fit_candidate(
        final_candidate,
        candidates[final_candidate],
        raw_logits,
        targets,
        regions,
        routes,
        reliability,
        all_indices,
    )
    groups = [row["similarity_group"] for row in rows]
    aggregate = {}
    bootstrap = {}
    for region in common.REGIONS:
        raw, raw_nll, raw_brier = common.region_metrics(
            raw_probability, raw_logits, targets, regions, all_indices, region
        )
        calibrated, calibrated_nll, calibrated_brier = common.region_metrics(
            oof_probability, oof_logits, targets, regions, all_indices, region
        )
        aggregate[region] = {"raw": raw, "oof_calibrated": calibrated}
        bootstrap[region] = {
            "nll_change": common.paired_group_bootstrap(
                raw_nll, calibrated_nll, groups
            ),
            "brier_change": common.paired_group_bootstrap(
                raw_brier, calibrated_brier, groups
            ),
        }
    by_predicted_route = {}
    for route in TUMOR_ROUTES:
        indices = np.flatnonzero(routes == route)
        by_predicted_route[route] = {}
        for region in common.REGIONS:
            raw, _, _ = common.region_metrics(
                raw_probability, raw_logits, targets, regions, indices, region
            )
            calibrated, _, _ = common.region_metrics(
                oof_probability, oof_logits, targets, regions, indices, region
            )
            by_predicted_route[route][region] = {
                "raw": compact(raw),
                "oof_calibrated": compact(calibrated),
            }
    label_names = np.asarray([row["label"] for row in rows])
    by_true_label = {}
    for label in common.LABELS:
        indices = np.flatnonzero(label_names == label)
        by_true_label[label] = {}
        for region in common.REGIONS:
            raw, _, _ = common.region_metrics(
                raw_probability, raw_logits, targets, regions, indices, region
            )
            calibrated, _, _ = common.region_metrics(
                oof_probability, oof_logits, targets, regions, indices, region
            )
            by_true_label[label][region] = {
                "raw": compact(raw),
                "oof_calibrated": compact(calibrated),
            }
    checks = {
        "aggregate_lesion_centered_ece_at_most_0_03": aggregate[
            "lesion_centered"
        ]["oof_calibrated"]["pixel_ece"]
        <= 0.03,
        **{
            f"{route}_lesion_centered_ece_at_most_0_05": by_predicted_route[
                route
            ]["lesion_centered"]["oof_calibrated"]["pixel_ece"]
            <= 0.05
            for route in TUMOR_ROUTES
        },
    }
    gate_passed = all(checks.values())
    all_selected = [item["selected_candidate"] for item in outer_selections] + [
        final_candidate
    ]
    operational = all(not candidates[item]["region_split"] for item in all_selected)
    raw_mask = raw_probability >= 0.5
    calibrated_mask = oof_probability >= 0.5
    changed = raw_mask != calibrated_mask
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C07 E08 lesion-centered nested hierarchical calibration",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C07 E08 whole-image nested hierarchical calibration",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "inner_selection_summary.csv", selection_rows)
    common.write_json(
        ARTIFACTS / "nested_models.json",
        {
            "outer_selections": outer_selections,
            "final_all_validation_selection": {
                "selected_candidate": final_candidate,
                "comparisons": final_comparisons,
                "inner_fold_models": final_inner_details,
                "fit_temperatures": {
                    key: value["temperature"]
                    for key, value in final_model["cells"].items()
                },
            },
        },
    )
    output = {
        "schema_version": 1,
        "experiment_id": "c07_hierarchical",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC nested calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": len(rows),
        "outer_selections": [
            {
                "outer_fold": item["outer_fold"],
                "selected_candidate": item["selected_candidate"],
                "selected_is_oracle": item["selected_is_oracle"],
            }
            for item in outer_selections
        ],
        "final_all_validation": {
            "status": "proposed only; not used for reported metrics",
            "selected_candidate": final_candidate,
            "selected_is_oracle": candidates[final_candidate]["region_split"],
            "artifact": "artifacts/nested_models.json",
        },
        "metrics": {
            "aggregate": aggregate,
            "by_predicted_route": by_predicted_route,
            "by_true_label_descriptive": by_true_label,
            "paired_group_bootstrap": bootstrap,
        },
        "hard_mask_changed_pixels": int(np.count_nonzero(changed)),
        "hard_mask_changed_images": int(np.any(changed, axis=(1, 2)).sum()),
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Aggregate lesion-centered ECE <=0.03 and each predicted tumor route lesion-centered ECE <=0.05.",
        },
        "operational_eligibility": (
            "ELIGIBLE" if operational and gate_passed else "INELIGIBLE"
        ),
        "operational_eligibility_reason": (
            "Every selected configuration omits reference-region routing."
            if operational
            else "At least one selected outer or final configuration requires reference-region routing."
        ),
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "classifier_predictions_sha256": common.sha256(
                ROOT / config["classifier_predictions_path"]
            ),
            "reliability_score_sha256": common.sha256(
                ROOT / config["reliability_score_path"]
            ),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "reported_outputs_are_nested_outer_fold_predictions": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Candidates containing R require reference-mask routing and are oracle diagnostics.",
            "Reliability weighting transfers frozen v1 OOF scores to E08.",
            "All metrics use the internal validation partition used during model development.",
            "Patient independence and external generalization are unverified.",
        ],
    }
    common.write_json(EXPERIMENT / "result.json", output)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(
        item for item in registry["experiments"] if item["id"] == "c07_hierarchical"
    )["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    if gate_passed and operational:
        decision = "ACCEPTED_OPERATIONAL"
    elif gate_passed:
        decision = "ACCEPTED_LOCALLY_ORACLE_ONLY"
    else:
        decision = "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Nested hierarchical pixel calibration decision

**Decision:** {decision}

The predeclared four-check conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Candidate selection occurred only inside each outer training fold; reported metrics use the five untouched outer folds.

Operational eligibility is `{output['operational_eligibility']}`: {output['operational_eligibility_reason']} The final all-validation fit is a proposal only and did not produce reported metrics.

The fixed threshold remained `0.50`. No model was retrained and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    common.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": "PASS",
            "experiment_id": "c07_hierarchical",
            "registered_decision": output["status"],
            "operational_eligibility": output["operational_eligibility"],
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": common.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in included
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "outer_selections": output["outer_selections"],
                "final_candidate": final_candidate,
                "operational_eligibility": output["operational_eligibility"],
                "aggregate_lesion_centered": compact(
                    aggregate["lesion_centered"]["oof_calibrated"]
                ),
                "route_lesion_centered_ece": {
                    route: by_predicted_route[route]["lesion_centered"][
                        "oof_calibrated"
                    ]["pixel_ece"]
                    for route in TUMOR_ROUTES
                },
                "hard_mask_changed_pixels": output["hard_mask_changed_pixels"],
                "hard_mask_changed_images": output["hard_mask_changed_images"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
