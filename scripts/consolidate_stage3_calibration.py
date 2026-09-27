"""Consolidate the completed C01-C09 calibration program."""

import hashlib
import json
from pathlib import Path

import advanced_calibration_common as common

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "audit/stage3_calibration_report.json"
OUTPUT_MD = ROOT / "audit/stage3_calibration_report.md"
EXPERIMENTS = (
    "c01_vector_scaling",
    "c02_dirichlet",
    "c03_isotonic",
    "c04_split_temperature",
    "c05_reliability_weighted",
    "c06_per_category",
    "c07_hierarchical",
    "c08_unified_ece",
    "c09_category_diagrams",
)


def aggregate(result, experiment_id):
    if experiment_id == "c01_vector_scaling":
        return result["segmenter"]["metrics"]["aggregate"]
    return result.get("metrics", {}).get("aggregate")


def main():
    registry = common.read_json(ROOT / "experiments/registry.json")
    registered = {item["id"]: item for item in registry["experiments"]}
    results = {}
    for experiment_id in EXPERIMENTS:
        if registered.get(experiment_id, {}).get("status") != "COMPLETE":
            raise ValueError(f"Stage 3 registry item is incomplete: {experiment_id}")
        result_path = ROOT / f"experiments/{experiment_id}/result.json"
        results[experiment_id] = common.read_json(result_path)
        if results[experiment_id]["status"] not in {"PASS", "FAIL"}:
            raise ValueError(f"Stage 3 result is not closed: {experiment_id}")
    decisions = {
        "c01_vector_scaling": {
            "outcome": "PASS_OPERATIONAL",
            "summary": "Segmenter gate passed; classifier vector scaling rejected descriptively; raw classifier probabilities retained.",
        },
        "c02_dirichlet": {
            "outcome": "FAIL",
            "summary": "Probability metrics passed, but mandatory hard-mask identity failed on 43,228 pixels across 680 images.",
        },
        "c03_isotonic": {
            "outcome": "PASS_OPERATIONAL_SELECTED",
            "summary": "All six ECE/Brier checks passed; selected by the fixed operational ranking.",
        },
        "c04_split_temperature": {
            "outcome": "PASS_ORACLE_ONLY",
            "summary": "Local gate passed with zero mask changes, but reference-region routing is unavailable at inference.",
        },
        "c05_reliability_weighted": {
            "outcome": "FAIL",
            "summary": "Low-reliability-quintile ECE worsened despite NLL improvement.",
        },
        "c06_per_category": {
            "outcome": "FAIL",
            "summary": "Predicted-glioma lesion-centred ECE was 0.055963, failing one of nine checks.",
        },
        "c07_hierarchical": {
            "outcome": "PASS_ORACLE_ONLY",
            "summary": "Nested gate passed, but every outer fold selected R or RC reference-region routing.",
        },
        "c08_unified_ece": {
            "outcome": "VERIFIED",
            "summary": "C01-C07 ECE values independently verified with 15 equal-frequency bins.",
        },
        "c09_category_diagrams": {
            "outcome": "VERIFIED",
            "summary": "Six OOF category diagrams produced; glioma calibration weakness remains explicit.",
        },
    }
    operational_passes = []
    for experiment_id in ("c01_vector_scaling", "c03_isotonic"):
        metrics = aggregate(results[experiment_id], experiment_id)["whole_image"][
            "oof_calibrated"
        ]
        operational_passes.append(
            {
                "experiment_id": experiment_id,
                "whole_image_ece": metrics["pixel_ece"],
                "whole_image_nll": metrics["pixel_nll"],
                "whole_image_brier": metrics["pixel_brier"],
            }
        )
    operational_passes.sort(
        key=lambda item: (
            item["whole_image_ece"],
            item["whole_image_nll"],
            item["whole_image_brier"],
            item["experiment_id"],
        )
    )
    selected = operational_passes[0]
    if selected["experiment_id"] != "c03_isotonic":
        raise ValueError("Fixed operational calibration ranking no longer selects C03")
    c03 = results["c03_isotonic"]
    c09 = common.read_json(ROOT / "audit/category_reliability_diagrams.json")
    experiment_records = []
    for experiment_id in EXPERIMENTS:
        result_path = ROOT / f"experiments/{experiment_id}/result.json"
        decision_path = ROOT / f"experiments/{experiment_id}/decision.md"
        experiment_records.append(
            {
                "id": experiment_id,
                "status": results[experiment_id]["status"],
                **decisions[experiment_id],
                "result_path": result_path.relative_to(ROOT).as_posix(),
                "result_sha256": common.sha256(result_path),
                "decision_path": decision_path.relative_to(ROOT).as_posix(),
                "decision_sha256": common.sha256(decision_path),
            }
        )
    output = {
        "schema_version": 1,
        "status": "STAGE_3_COMPLETE",
        "scope": "Frozen E08 internal-validation calibration program; not clinical validation",
        "experiments": experiment_records,
        "selected_operational_calibrator": {
            **selected,
            "method": "probability-only monotone isotonic mapping",
            "reported_metrics": "five-fold similarity-aware out-of-fold",
            "final_fit_role": "all-validation mapping is a proposed operational transform only and did not produce reported metrics",
            "mapping_path": "experiments/c03_isotonic/artifacts/isotonic_mappings.json",
            "mapping_sha256": common.sha256(
                ROOT / "experiments/c03_isotonic/artifacts/isotonic_mappings.json"
            ),
            "experiment_manifest_sha256": common.sha256(
                ROOT / "experiments/c03_isotonic/artifacts/experiment_manifest.json"
            ),
            "hard_mask_changed_pixels_oof": c03["hard_mask_changed_pixels"],
            "hard_mask_changed_images_oof": c03["hard_mask_changed_images"],
            "threshold": 0.5,
        },
        "operational_ranking": operational_passes,
        "classifier_policy": {
            "accepted_probabilities": "raw",
            "reason": "C01 descriptive classifier scaling did not improve the required calibration evidence; earlier multiclass and binary temperature studies were also rejected.",
            "raw_four_class_ece_15_bin": results["c01_vector_scaling"][
                "classifier_descriptive"
            ]["raw_four_class"]["ece"],
            "raw_binary_ece_15_bin": results["c01_vector_scaling"][
                "classifier_descriptive"
            ]["raw_binary"]["ece"],
        },
        "category_findings": {
            label: {
                "cases": item["cases"],
                "raw_lesion_centered_pixel_ece": item["pixel_lesion_centered"][
                    "raw_ece"
                ],
                "c03_lesion_centered_pixel_ece": item[
                    "pixel_lesion_centered"
                ]["c03_oof_ece"],
                "case_reliability_ece_15_bin": item["case_reliability"]["ece"],
            }
            for label, item in c09["categories"].items()
        },
        "h7_status": {
            "calibration_condition": "PASS",
            "reason": "C03 is operational, passed its unchanged gate, and was selected by the fixed ranking.",
            "reliability_condition": "PENDING_STAGE_4",
            "e08_final_model_nomination": "PENDING; E08 is not final and the sealed test remains closed",
        },
        "constraints": {
            "sealed_test_accesses": 0,
            "neural_retraining_events_during_stage_3": 0,
            "threshold": 0.5,
            "threshold_changes": 0,
            "post_result_gate_relaxations": 0,
            "calibration_experiments_skipped_merged_or_substituted": 0,
            "verification_mismatches": 0,
            "implementation_note": "C07 first invocation stopped before fitting on a missing config-field reference; the loader was corrected and the unchanged registered experiment reran successfully.",
        },
        "what_stage_3_establishes": [
            "C03 is the selected operational E08 probability calibrator under the registered ranking.",
            "OOF aggregate ECE/Brier evidence supports C03's local registered pass.",
            "Calibration performance is heterogeneous: C03 worsens glioma lesion-centred ECE while improving meningioma and pituitary.",
            "Reference-region methods can calibrate better locally but are unavailable during inference.",
        ],
        "what_stage_3_does_not_establish": [
            "Clinical validity or diagnostic safety.",
            "Cross-dataset or patient-independent generalization.",
            "That calibrated confidence fixes spatial segmentation errors.",
            "That E08 replaces v1; H7 reliability evidence is still pending and the sealed test is unopened.",
            "That the proposed all-validation C03 mapping has independent prospective performance.",
        ],
        "verification": {
            "registry_items_complete": 9,
            "experiment_results_closed": 9,
            "review_log_entries_before_final_close": sum(
                1
                for line in (ROOT / "audit/review_log.md")
                .read_text(encoding="utf-8-sig")
                .splitlines()
                if line.startswith("## ")
            ),
            "c08_report_sha256": common.sha256(
                ROOT / "audit/unified_ece_comparison.json"
            ),
            "c09_index_sha256": common.sha256(
                ROOT / "audit/category_reliability_diagrams.json"
            ),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    common.write_json(OUTPUT_JSON, output)
    lines = [
        "# Stage 3 calibration completion report",
        "",
        "**Status: COMPLETE. Stage 4 has not started. The sealed test remains unopened.**",
        "",
        "## Decision",
        "",
        "C03 isotonic calibration is the selected operational E08 calibrator. It passed its unchanged six-check gate and ranked ahead of C01 by whole-image OOF ECE (`0.000076` versus `0.000331`). C04 and C07 passed locally but are excluded because they require reference-mask routing at inference.",
        "",
        "The selected C03 mapping changes 22,891 OOF hard-mask pixels across 679 of 680 images at the unchanged `0.50` threshold. Amendment 02 therefore requires Stage 4 to use the locked calibrated probabilities and resulting masks. The final all-validation mapping is an operational proposal; it was not used for reported metrics.",
        "",
        "## Experiment outcomes",
        "",
        "| Experiment | Outcome | Finding |",
        "|---|---|---|",
    ]
    for record in experiment_records:
        lines.append(
            f"| {record['id']} | {record['outcome']} | {record['summary']} |"
        )
    lines.extend(
        [
            "",
            "## Category limitation",
            "",
            "| Category | Cases | Raw lesion ECE | C03 lesion ECE | Case-reliability ECE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, item in output["category_findings"].items():
        lines.append(
            f"| {label} | {item['cases']} | {item['raw_lesion_centered_pixel_ece']:.6f} | {item['c03_lesion_centered_pixel_ece']:.6f} | {item['case_reliability_ece_15_bin']:.6f} |"
        )
    lines.extend(
        [
            "",
            "C03 improves meningioma and pituitary lesion-centred calibration but worsens glioma (`0.082850` to `0.085523`). The aggregate pass therefore does not mean calibration is uniformly improved across tumor types.",
            "",
            "## H7 and governance",
            "",
            "H7's calibration condition is satisfied. Its reliability condition remains pending Stage 4, so E08 is not the final nominated model. All C01–C09 experiments are separately registered, closed, committed, and represented in the review log. No neural retraining occurred, the threshold stayed `0.50`, no gate was relaxed, and the sealed test was accessed zero times.",
            "",
            "## Limits",
            "",
            "Stage 3 does not establish clinical validity, external or patient-independent generalization, or correction of confident spatial errors. It does not validate the all-validation operational fit prospectively. These results remain internal BRISC development evidence.",
        ]
    )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": output["status"],
                "selected_operational_calibrator": selected["experiment_id"],
                "calibration_condition_h7": "PASS",
                "reliability_condition_h7": "PENDING_STAGE_4",
                "sealed_test_accesses": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
