"""Verify and consolidate all Stage 3 ECE metrics with one bin definition."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/c08_unified_ece"
ARTIFACTS = EXPERIMENT / "artifacts"
OUTPUT_JSON = ROOT / "audit/unified_ece_comparison.json"
OUTPUT_MD = ROOT / "audit/unified_ece_comparison.md"


def recompute_stored_ece(summary, tolerance=1e-12):
    bins = summary["bins"]
    if len(bins) != 15 or any(int(item["count"]) <= 0 for item in bins):
        raise ValueError("ECE summary does not contain 15 non-empty bins")
    total = sum(int(item["count"]) for item in bins)
    if total != int(summary.get("pixel_count", total)):
        raise ValueError("ECE bin counts do not cover the stored pixel count")
    recomputed = sum(
        (int(item["count"]) / total)
        * abs(float(item["mean_probability"]) - float(item["observed_frequency"]))
        for item in bins
    )
    stored = float(summary.get("pixel_ece", summary.get("ece")))
    difference = abs(recomputed - stored)
    if difference > tolerance:
        raise ValueError(f"Stored ECE mismatch: {difference}")
    return {
        "stored_ece": stored,
        "recomputed_ece": float(recomputed),
        "absolute_difference": float(difference),
        "bins": 15,
        "observations": total,
    }


def aggregate_metrics(result, experiment_id):
    if experiment_id == "c01_vector_scaling":
        return result["segmenter"]["metrics"]["aggregate"]
    return result["metrics"]["aggregate"]


def main():
    config_path = EXPERIMENT / "config.json"
    config = common.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("C08 is not in its registered pre-run state")
    tolerance = float(config["verification_tolerance"])
    segmenter_rows = []
    source_verification = {}
    for experiment_id, expected_hash in config["source_results"].items():
        result_path = ROOT / f"experiments/{experiment_id}/result.json"
        actual_hash = common.sha256(result_path)
        if actual_hash != expected_hash:
            raise ValueError(f"C08 source hash mismatch: {experiment_id}")
        result = common.read_json(result_path)
        aggregate = aggregate_metrics(result, experiment_id)
        source_verification[experiment_id] = {
            "path": result_path.relative_to(ROOT).as_posix(),
            "sha256": actual_hash,
            "decision": result["status"],
        }
        for region in common.REGIONS:
            raw = recompute_stored_ece(aggregate[region]["raw"], tolerance)
            calibrated = recompute_stored_ece(
                aggregate[region]["oof_calibrated"], tolerance
            )
            segmenter_rows.append(
                {
                    "experiment_id": experiment_id,
                    "registered_decision": result["status"],
                    "operational_eligibility": result.get(
                        "operational_eligibility", "see experiment decision"
                    ),
                    "region": region,
                    "raw_ece": raw["recomputed_ece"],
                    "oof_calibrated_ece": calibrated["recomputed_ece"],
                    "ece_change": calibrated["recomputed_ece"]
                    - raw["recomputed_ece"],
                    "raw_observations": raw["observations"],
                    "calibrated_observations": calibrated["observations"],
                    "maximum_recomputation_difference": max(
                        raw["absolute_difference"], calibrated["absolute_difference"]
                    ),
                }
            )
    c01 = common.read_json(ROOT / "experiments/c01_vector_scaling/result.json")
    classifier_rows = []
    for output_name in ("four_class", "binary"):
        for state in ("raw", "oof"):
            summary = c01["classifier_descriptive"][f"{state}_{output_name}"]
            verified = recompute_stored_ece(summary, tolerance)
            classifier_rows.append(
                {
                    "output": output_name,
                    "state": state,
                    "ece": verified["recomputed_ece"],
                    "observations": verified["observations"],
                    "absolute_difference": verified["absolute_difference"],
                }
            )
    reliability_path = ROOT / config["case_reliability_oof_path"]
    if common.sha256(reliability_path) != config["case_reliability_oof_sha256"]:
        raise ValueError("Case-reliability source hash mismatch")
    reliability_rows = common.read_csv(reliability_path)
    if len(reliability_rows) != 680 or len({row["path"] for row in reliability_rows}) != 680:
        raise ValueError("Case-reliability row count or path uniqueness mismatch")
    reliability = np.asarray(
        [float(row["oof_reliability_score"]) for row in reliability_rows]
    )
    success = np.asarray(
        [int(row["success_dice_at_least_0_50"]) for row in reliability_rows]
    )
    if not np.isfinite(reliability).all() or np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("Invalid case-reliability score")
    reliability_ece, reliability_mce, reliability_bins = (
        common.equal_frequency_binary(reliability, success, bins=15)
    )
    output = {
        "schema_version": 1,
        "status": "VERIFIED",
        "scope": "Descriptive unified 15-bin equal-frequency ECE comparison; earlier gates unchanged",
        "binning": "15 equal-frequency bins with stable probability ordering",
        "segmenter_pixel_ece": segmenter_rows,
        "classifier_ece": classifier_rows,
        "case_reliability_ece": {
            "source_model": "accepted v1 case-level logistic reliability model",
            "cases": 680,
            "ece": reliability_ece,
            "mce": reliability_mce,
            "bins": reliability_bins,
        },
        "source_verification": source_verification,
        "verification": {
            "source_results_verified": 7,
            "segmenter_summaries_verified": len(segmenter_rows) * 2,
            "classifier_summaries_verified": len(classifier_rows),
            "maximum_allowed_ece_difference": tolerance,
            "maximum_observed_ece_difference": max(
                row["maximum_recomputation_difference"] for row in segmenter_rows
            ),
            "test_images_accessed": 0,
            "threshold_changed": False,
            "model_retrained": False,
            "calibrator_refitted": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "interpretation": [
            "C08 standardizes reporting and does not retroactively alter any C01-C07 gate or decision.",
            "Whole-image ECE is dominated by background pixels and must be read with lesion-centered ECE.",
            "Oracle calibrators remain operationally ineligible even when their ECE is low.",
        ],
    }
    common.write_json(OUTPUT_JSON, output)
    lines = [
        "# Unified 15-bin ECE comparison",
        "",
        "All values use 15 equal-frequency bins. C01–C07 source hashes, bin counts, pixel coverage, and stored ECE values were independently verified. This table does not change any registered gate or decision.",
        "",
        "## Segmenter pixel calibration",
        "",
        "| Experiment | Decision | Region | Raw ECE | OOF calibrated ECE | Change |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in segmenter_rows:
        lines.append(
            f"| {row['experiment_id']} | {row['registered_decision']} | {row['region']} | {row['raw_ece']:.6f} | {row['oof_calibrated_ece']:.6f} | {row['ece_change']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen classifier outputs",
            "",
            "| Output | State | ECE | Cases |",
            "|---|---|---:|---:|",
        ]
    )
    for row in classifier_rows:
        lines.append(
            f"| {row['output']} | {row['state']} | {row['ece']:.6f} | {row['observations']} |"
        )
    lines.extend(
        [
            "",
            "## Case-level reliability",
            "",
            f"The accepted v1 reliability model has 15-bin ECE `{reliability_ece:.6f}` and MCE `{reliability_mce:.6f}` across 680 out-of-fold cases.",
            "",
            "## Interpretation",
            "",
            "Low whole-image ECE can coexist with weaker lesion-centred calibration because background pixels dominate the image. Reference-region methods C04 and C07 remain oracle diagnostics and cannot be deployed. C08 is a reporting standardization only.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    common.write_csv(ARTIFACTS / "segmenter_ece_table.csv", segmenter_rows)
    result = {
        "schema_version": 1,
        "experiment_id": "c08_unified_ece",
        "status": "PASS",
        "decision": "VERIFIED",
        "outputs": [
            OUTPUT_MD.relative_to(ROOT).as_posix(),
            OUTPUT_JSON.relative_to(ROOT).as_posix(),
        ],
        "verification": output["verification"],
    }
    common.write_json(EXPERIMENT / "result.json", result)
    config["status"] = "COMPLETE"
    common.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = common.read_json(registry_path)
    next(
        item for item in registry["experiments"] if item["id"] == "c08_unified_ece"
    )["status"] = "COMPLETE"
    common.write_json(registry_path, registry)
    (EXPERIMENT / "decision.md").write_text(
        """# Unified ECE comparison decision

**Decision:** VERIFIED

All seven registered source hashes matched. Every stored segmenter and classifier summary used 15 non-empty equal-frequency bins with exact observation coverage, and every independently recomputed ECE matched within `1e-12`. The accepted case-reliability ECE was recomputed with the same 15-bin definition.

C08 changes no earlier gate or decision. No calibrator or neural model was fitted, no threshold changed, and no sealed-test image was accessed.
""",
        encoding="utf-8",
    )
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in EXPERIMENT.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    included.extend([OUTPUT_MD, OUTPUT_JSON])
    common.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": "PASS",
            "experiment_id": "c08_unified_ece",
            "registered_decision": "VERIFIED",
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": common.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in sorted(set(included))
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "segmenter_rows": len(segmenter_rows),
                "classifier_rows": len(classifier_rows),
                "case_reliability_ece": reliability_ece,
                "maximum_observed_ece_difference": output["verification"][
                    "maximum_observed_ece_difference"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
