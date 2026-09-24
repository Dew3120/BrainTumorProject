"""Run and close C06 classifier-routed per-category calibration."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from run_c05_reliability_weighted import fit_weighted_temperature
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c06_per_category"
ARTIFACTS = EXPERIMENT / "artifacts"
TUMOR_ROUTES = ("glioma", "meningioma", "pituitary")


def load_predicted_routes(config, rows):
    path = ROOT / config["classifier_predictions_path"]
    if common.sha256(path) != config["classifier_predictions_sha256"]:
        raise ValueError("Classifier prediction hash mismatch")
    source = {row["path"]: row for row in common.read_csv(path)}
    routes = []
    for row in rows:
        candidate = source.get(row["path"])
        if candidate is None:
            raise ValueError(f"Missing classifier prediction: {row['path']}")
        if candidate["label"] != row["label"]:
            raise ValueError(f"Classifier metadata mismatch: {row['path']}")
        routes.append(candidate["predicted"])
    routes = np.asarray(routes)
    expected = config["predicted_route_counts"]
    observed = {name: int(np.sum(routes == name)) for name in expected}
    if observed != expected:
        raise ValueError(f"Predicted route counts changed: {observed}")
    return routes


def fit_temperature(logits, targets, indices):
    reliability = np.ones(len(logits), dtype=np.float64)
    return fit_weighted_temperature(logits, targets, indices, reliability)


def compact(metrics):
    return {
        key: metrics[key]
        for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
    }


def one_case_metrics(probability, logit, target):
    target_float = target.astype(np.float64)
    return {
        "image_count": 1,
        "pixel_count": int(target.size),
        "pixel_nll": float(
            np.mean(np.logaddexp(0.0, logit) - target_float * logit)
        ),
        "pixel_brier": float(np.mean(np.square(probability - target_float))),
        "pixel_ece": None,
        "pixel_mce": None,
        "note": "ECE unavailable for the preregistered one-case no_tumor fallback route",
    }


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C06 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    routes = load_predicted_routes(config, rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    for fold in range(5):
        fit_mask = assignment != fold
        evaluation_indices = np.flatnonzero(assignment == fold)
        global_parameters = fit_temperature(
            raw_logits, targets, np.flatnonzero(fit_mask)
        )
        parameters = {}
        for route in TUMOR_ROUTES:
            fit_indices = np.flatnonzero(fit_mask & (routes == route))
            if len(fit_indices) < 2:
                raise ValueError(f"Insufficient fitting cases for {route} in fold {fold + 1}")
            parameters[route] = fit_temperature(raw_logits, targets, fit_indices)
        for index in evaluation_indices:
            selected = parameters.get(routes[index], global_parameters)
            transformed = raw_logits[index] / selected["temperature"]
            oof_logits[index] = transformed
            oof_probability[index] = expit(transformed)
        fold_parameters.append(
            {
                "fold": fold + 1,
                "fit_images": int(fit_mask.sum()),
                "evaluation_images": len(evaluation_indices),
                "global_fallback_temperature": global_parameters["temperature"],
                **{
                    f"{route}_temperature": parameters[route]["temperature"]
                    for route in TUMOR_ROUTES
                },
                **{
                    f"{route}_fit_images": parameters[route]["fit_images"]
                    for route in TUMOR_ROUTES
                },
            }
        )
    all_indices = np.arange(len(rows))
    final_parameters = {
        "global_fallback": fit_temperature(raw_logits, targets, all_indices),
        **{
            route: fit_temperature(
                raw_logits, targets, np.flatnonzero(routes == route)
            )
            for route in TUMOR_ROUTES
        },
    }
    regions = common.lesion_regions(targets)
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
    no_tumor_index = int(np.flatnonzero(routes == "no_tumor")[0])
    no_tumor_fallback = {
        "path": rows[no_tumor_index]["path"],
        "true_label": rows[no_tumor_index]["label"],
        "raw_whole_image": one_case_metrics(
            raw_probability[no_tumor_index],
            raw_logits[no_tumor_index],
            targets[no_tumor_index],
        ),
        "oof_calibrated_whole_image": one_case_metrics(
            oof_probability[no_tumor_index],
            oof_logits[no_tumor_index],
            targets[no_tumor_index],
        ),
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
        f"{route}_{region}_ece_at_most_0_05": by_predicted_route[route][region][
            "oof_calibrated"
        ]["pixel_ece"]
        <= 0.05
        for route in TUMOR_ROUTES
        for region in common.REGIONS
    }
    gate_passed = all(checks.values())
    raw_mask = raw_probability >= 0.5
    calibrated_mask = oof_probability >= 0.5
    changed = raw_mask != calibrated_mask
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C06 E08 lesion-centered classifier-routed calibration",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C06 E08 whole-image classifier-routed calibration",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "fold_parameters.csv", fold_parameters)
    output = {
        "schema_version": 1,
        "experiment_id": "c06_per_category",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": len(rows),
        "routing": {
            "input": "frozen classifier predicted class",
            "true_label_used": False,
            "route_counts": config["predicted_route_counts"],
        },
        "fold_parameters": fold_parameters,
        "final_all_validation": {
            "status": "proposed operational only; not used for reported metrics",
            "parameters": final_parameters,
        },
        "metrics": {
            "aggregate": aggregate,
            "by_predicted_route": by_predicted_route,
            "no_tumor_fallback": no_tumor_fallback,
            "by_true_label_descriptive": by_true_label,
            "paired_group_bootstrap": bootstrap,
        },
        "hard_mask_changed_pixels": int(np.count_nonzero(changed)),
        "hard_mask_changed_images": int(np.any(changed, axis=(1, 2)).sum()),
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "ECE <=0.05 in all three regions within each classifier-predicted tumor route; nine conjunctive checks.",
        },
        "operational_eligibility": "ELIGIBLE_IF_GATE_PASSES",
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "classifier_predictions_sha256": common.sha256(
                ROOT / config["classifier_predictions_path"]
            ),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "calibration_outputs_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "The no-tumor route has one segmentation-validation case and uses a global fallback.",
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
        item for item in registry["experiments"] if item["id"] == "c06_per_category"
    )["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Classifier-routed per-category pixel calibration decision

**Decision:** {decision}

The predeclared nine-check conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported metrics use only out-of-fold probabilities. Routing used the frozen classifier prediction, never true tumor type.

The single no-tumor prediction used the preregistered global fallback and did not control the gate. The fixed threshold remained `0.50`.

No model was retrained and no sealed-test image was accessed.
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
            "experiment_id": "c06_per_category",
            "registered_decision": output["status"],
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
                "fold_parameters": fold_parameters,
                "aggregate": {
                    name: {
                        "raw": compact(aggregate[name]["raw"]),
                        "oof_calibrated": compact(
                            aggregate[name]["oof_calibrated"]
                        ),
                    }
                    for name in common.REGIONS
                },
                "route_ece": {
                    route: {
                        region: by_predicted_route[route][region][
                            "oof_calibrated"
                        ]["pixel_ece"]
                        for region in common.REGIONS
                    }
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
