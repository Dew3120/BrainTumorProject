"""Run and close C05 reliability-weighted temperature calibration."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c05_reliability_weighted"
ARTIFACTS = EXPERIMENT / "artifacts"


def load_reliability_scores(config, rows):
    path = ROOT / config["reliability_score_path"]
    if common.sha256(path) != config["reliability_score_sha256"]:
        raise ValueError("Frozen reliability score hash mismatch")
    source = {row["path"]: row for row in common.read_csv(path)}
    scores = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        candidate = source.get(row["path"])
        if candidate is None:
            raise ValueError(f"Missing reliability score: {row['path']}")
        if (
            candidate["similarity_group"] != row["similarity_group"]
            or candidate["label"] != row["label"]
        ):
            raise ValueError(f"Reliability metadata mismatch: {row['path']}")
        scores[index] = float(candidate[config["reliability_score_column"]])
    if not np.isfinite(scores).all() or np.any((scores <= 0) | (scores > 1)):
        raise ValueError("Reliability scores must be finite in (0, 1]")
    return scores


def fit_weighted_temperature(logits, targets, indices, reliability):
    """Fit a scalar temperature with case totals proportional to reliability."""
    x = np.asarray(logits[indices], dtype=np.float64)
    y = np.asarray(targets[indices], dtype=np.float64)
    case_weight = np.asarray(reliability[indices], dtype=np.float64)
    case_weight /= case_weight.sum()

    def objective(value):
        transformed = x / np.exp(value[0])
        per_case_loss = np.mean(
            np.logaddexp(0.0, transformed) - y * transformed, axis=(1, 2)
        )
        per_case_gradient = np.mean(
            -(expit(transformed) - y) * transformed, axis=(1, 2)
        )
        return float(case_weight @ per_case_loss), np.array(
            [case_weight @ per_case_gradient]
        )

    result = minimize(
        objective,
        np.zeros(1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(np.log(0.05), np.log(20.0))],
        options={"maxiter": 200, "ftol": 1e-13, "gtol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Weighted temperature fitting failed: {result.message}")
    return {
        "temperature": float(np.exp(result.x[0])),
        "fit_images": len(indices),
        "minimum_reliability_weight": float(reliability[indices].min()),
        "maximum_reliability_weight": float(reliability[indices].max()),
        "mean_reliability_weight": float(reliability[indices].mean()),
        "fit_objective": float(result.fun),
    }


def compact_metrics(metrics):
    return {
        key: metrics[key]
        for key in ("pixel_nll", "pixel_brier", "pixel_ece", "pixel_mce")
    }


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C05 is not in its registered pre-run state")
    cache_run, rows, raw_probability, targets, _labels = common.load_cache(
        config["calibration_cache_manifest_sha256"]
    )
    assignment = common.fold_assignment(rows)
    reliability = load_reliability_scores(config, rows)
    clipped = np.clip(raw_probability, 1e-7, 1 - 1e-7)
    raw_logits = np.log(clipped / (1 - clipped))
    oof_logits = np.empty_like(raw_logits)
    oof_probability = np.empty_like(raw_probability)
    fold_parameters = []
    for fold in range(5):
        fit_indices = np.flatnonzero(assignment != fold)
        evaluation_indices = np.flatnonzero(assignment == fold)
        parameters = fit_weighted_temperature(
            raw_logits, targets, fit_indices, reliability
        )
        transformed = raw_logits[evaluation_indices] / parameters["temperature"]
        oof_logits[evaluation_indices] = transformed
        oof_probability[evaluation_indices] = expit(transformed)
        fold_parameters.append(
            {
                "fold": fold + 1,
                "evaluation_images": len(evaluation_indices),
                **parameters,
            }
        )
    all_indices = np.arange(len(rows))
    final_parameters = fit_weighted_temperature(
        raw_logits, targets, all_indices, reliability
    )
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
    label_names = np.asarray([row["label"] for row in rows])
    per_category = {}
    for label in common.LABELS:
        indices = np.flatnonzero(label_names == label)
        per_category[label] = {}
        for region in common.REGIONS:
            raw, _, _ = common.region_metrics(
                raw_probability, raw_logits, targets, regions, indices, region
            )
            calibrated, _, _ = common.region_metrics(
                oof_probability, oof_logits, targets, regions, indices, region
            )
            per_category[label][region] = {
                "raw": compact_metrics(raw),
                "oof_calibrated": compact_metrics(calibrated),
            }
    low_order = sorted(
        range(len(rows)),
        key=lambda index: (
            reliability[index],
            common.normalized_path(rows[index]["path"]),
        ),
    )
    low_indices = np.asarray(low_order[:136], dtype=np.int64)
    low_quintile = {}
    for region in common.REGIONS:
        raw, _, _ = common.region_metrics(
            raw_probability, raw_logits, targets, regions, low_indices, region
        )
        calibrated, _, _ = common.region_metrics(
            oof_probability, oof_logits, targets, regions, low_indices, region
        )
        low_quintile[region] = {
            "raw": compact_metrics(raw),
            "oof_calibrated": compact_metrics(calibrated),
        }
    low_rows = [
        {
            "rank": rank,
            "path": rows[index]["path"],
            "label": rows[index]["label"],
            "similarity_group": rows[index]["similarity_group"],
            "fold": int(assignment[index]) + 1,
            "frozen_v1_oof_reliability_score": reliability[index],
        }
        for rank, index in enumerate(low_indices, start=1)
    ]
    common.write_csv(ARTIFACTS / "low_reliability_quintile.csv", low_rows)
    raw_mask = raw_probability >= 0.5
    calibrated_mask = oof_probability >= 0.5
    changed = raw_mask != calibrated_mask
    low_raw = low_quintile["whole_image"]["raw"]
    low_calibrated = low_quintile["whole_image"]["oof_calibrated"]
    checks = {
        "whole_image_ece_at_most_0_04": aggregate["whole_image"][
            "oof_calibrated"
        ]["pixel_ece"]
        <= 0.04,
        "low_quintile_whole_image_ece_not_worse": low_calibrated["pixel_ece"]
        <= low_raw["pixel_ece"] + 1e-12,
        "low_quintile_whole_image_nll_not_worse": low_calibrated["pixel_nll"]
        <= low_raw["pixel_nll"] + 1e-12,
    }
    gate_passed = all(checks.values())
    common.render_binary_diagram(
        ARTIFACTS / "lesion_centered_reliability.png",
        "C05 E08 lesion-centered reliability-weighted calibration",
        aggregate["lesion_centered"]["raw"],
        aggregate["lesion_centered"]["oof_calibrated"],
    )
    common.render_binary_diagram(
        ARTIFACTS / "whole_image_reliability.png",
        "C05 E08 whole-image reliability-weighted calibration",
        aggregate["whole_image"]["raw"],
        aggregate["whole_image"]["oof_calibrated"],
    )
    common.write_csv(ARTIFACTS / "fold_parameters.csv", fold_parameters)
    output = {
        "schema_version": 1,
        "experiment_id": "c05_reliability_weighted",
        "status": "PASS" if gate_passed else "FAIL",
        "scope": "Internal BRISC calibration comparison; not clinical validation",
        "source_model": "e08_ensemble",
        "validation_cases": len(rows),
        "weight_source": {
            "path": config["reliability_score_path"],
            "sha256": config["reliability_score_sha256"],
            "role": "development fitting weight only; no inference-time routing",
            "reference_derived_features": False,
        },
        "fold_parameters": fold_parameters,
        "final_all_validation": {
            "status": "proposed operational only; not used for reported metrics",
            **final_parameters,
        },
        "metrics": {
            "aggregate": aggregate,
            "per_category": per_category,
            "low_reliability_quintile": low_quintile,
            "paired_group_bootstrap": bootstrap,
        },
        "hard_mask_changed_pixels": int(np.count_nonzero(changed)),
        "hard_mask_changed_images": int(np.any(changed, axis=(1, 2)).sum()),
        "gate": {
            "status": "PASS" if gate_passed else "FAIL",
            "checks": checks,
            "rule": "Whole-image ECE <=0.04 and low-reliability-quintile whole-image ECE and NLL do not worsen.",
        },
        "operational_eligibility": "ELIGIBLE_IF_GATE_PASSES",
        "verification": {
            "cache_run": str(cache_run),
            "cache_manifest_sha256": common.sha256(cache_run / "manifest.json"),
            "fold_assignment_sha256": common.sha256(common.FOLDS_PATH),
            "calibration_outputs_are_out_of_fold": True,
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Weights transfer frozen v1 OOF reliability scores to E08 rather than using an E08 reliability model, which is not available before Stage 4.",
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
        item
        for item in registry["experiments"]
        if item["id"] == "c05_reliability_weighted"
    )["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    decision = "ACCEPTED" if gate_passed else "REJECTED"
    (EXPERIMENT / "decision.md").write_text(
        f"""# Reliability-weighted pixel calibration decision

**Decision:** {decision}

The predeclared three-check conjunctive gate was {'satisfied' if gate_passed else 'not satisfied'} and was not relaxed. Reported metrics use only out-of-fold probabilities. The final all-validation temperature is an operational proposal only.

The fitting weights came from frozen v1 out-of-fold prediction-only reliability scores. Dice and reference-mask geometry did not enter the weights. The fixed threshold remained `0.50`.

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
    manifest = {
        "status": "PASS",
        "experiment_id": "c05_reliability_weighted",
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
    }
    common.write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": output["status"],
                "fold_parameters": fold_parameters,
                "aggregate": {
                    name: {
                        "raw": compact_metrics(aggregate[name]["raw"]),
                        "oof_calibrated": compact_metrics(
                            aggregate[name]["oof_calibrated"]
                        ),
                    }
                    for name in common.REGIONS
                },
                "low_reliability_quintile_whole_image": low_quintile[
                    "whole_image"
                ],
                "hard_mask_changed_pixels": output["hard_mask_changed_pixels"],
                "hard_mask_changed_images": output["hard_mask_changed_images"],
                "checks": checks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
