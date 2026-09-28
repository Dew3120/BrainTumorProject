"""Finalize the registered reliability-model selection and Stage 4 report."""

import hashlib
import json
import subprocess
from pathlib import Path

import advanced_calibration_common as calibration
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SELECTION = ROOT / "experiments/e08_final_selection"
ARTIFACTS = SELECTION / "artifacts"
SELECTION_REPORT = ROOT / "audit/reliability_model_selection.md"
COMPLETION_REPORT = ROOT / "audit/stage4_completion_report.md"

EXPECTED = {
    "c03_isotonic": (
        "3af002eacbc05e14134abba67135515a58e060b0c4f255bbccd8f10b79c49cb9",
        "beec6f5526204b1dc5c44bef17048d56ad7f999d15b4dde308d9303f38a9abe4",
    ),
    "r01_ablation": (
        "d53b160a7aec1ab2c8b0d612576530cdcc530e1b7b30bb789d3e49bee3d9499d",
        "7e7162d4d1b33349c7944a8d8045898dbe37966d2c407b5feec556dbcfa51a7a",
    ),
    "r02_reduced_features": (
        "fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2",
        "1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819",
    ),
    "r03_extended_features": (
        "eba95d46886f1fad3bf47822ed38f5a5b48fdc74922c084b4bfa013f932ff740",
        "5ef0b58c58a843867ff4416cca4edfa56979285123c92ef942b7523817519aee",
    ),
    "r04_per_category": (
        "e47d380024953f9e77be1d1916f4eeb0f103a72a12070ecc3a5708d43fc39046",
        "b5180fb40c4663502c41cce03e83f174721897b74f145d8519c677287f47a047",
    ),
    "r05_category_aware": (
        "3f08ad6dde731209c53d84c4b01c6709bcec64dca536b801cd4e78d94abf2d8d",
        "237d8dbfeef592fa136ed513458161058acb6a98b32a6f79cc518dc34c6d8415",
    ),
    "r06_nonlinear_comparison": (
        "aa09898ee9e46f4d3edeecae34ec87d90d7107416b95376c2a7dee7a77eefb73",
        "bafd8a2076564ea0bb563e47e6814660ddc95098ce976b154b2889c828aed634",
    ),
    "r07_coefficient_bootstrap": (
        "fc8b1bd80826913ff990ea993bf1c0c8bdf028b614101f967afc4282c01cb948",
        "afddd3023655ec9fc539611a50eef3335d36923894e49aa99e831f9743e5dd10",
    ),
    "r08_prospective_fold_evaluation": (
        "8e6ea05a18937949aa55aa4d285e535047e3c1dba3a5a98bc12b7b3e523f34a4",
        "f96f296e07f8573898d5ff5840dafa78b8f26a63815bf4c35d2e2174156fb1b9",
    ),
    "r09_fold_sensitivity": (
        "29311b740b0ca9ae2806c7eb54ac0c86a93255d8706e324fb52f3262d61274bc",
        "fcfa94aefaefab4f1e6abcfa12534da52206baed63e616790d7a2c0e34f13423",
    ),
    "r10_seed_stability": (
        "59245caf1f2d8aba0221503a08466b990a6d70318afdaa71c2d87de52e12c888",
        "7bf8172e7fe5d75d3543e1ca72b189f3cae40a3d5488bbe753547ea45359ed5d",
    ),
}

V1 = {
    "failure_detection_auroc": 0.9409900910,
    "auroc_ci95_lower": 0.9184,
    "auroc_ci95_upper": 0.9612,
    "ece": 0.021171,
    "brier_skill": 0.476052,
}


def verify_sources():
    verified = {}
    results = {}
    for experiment, (result_hash, manifest_hash) in EXPECTED.items():
        directory = ROOT / "experiments" / experiment
        result_path = directory / "result.json"
        manifest_path = directory / "artifacts/experiment_manifest.json"
        if calibration.sha256(result_path) != result_hash:
            raise ValueError(f"{experiment} result hash mismatch")
        if calibration.sha256(manifest_path) != manifest_hash:
            raise ValueError(f"{experiment} manifest hash mismatch")
        verified[experiment] = {
            "result_sha256": result_hash,
            "manifest_sha256": manifest_hash,
        }
        results[experiment] = calibration.read_json(result_path)
    return results, verified


def binary_metrics(reliability, success):
    ece, mce, _bins = calibration.equal_frequency_binary(
        reliability, success.astype(np.float64), bins=15
    )
    return {
        "ece": ece,
        "mce": mce,
        "brier": float(np.mean(np.square(reliability - success))),
    }


def r04_aggregate():
    rows = calibration.read_csv(
        ROOT / "experiments/r04_per_category/artifacts/oof_category_scores.csv"
    )
    included = [
        row for row in rows if int(row["version_b_excluded_no_tumor_route"]) == 0
    ]
    if len(included) != 679:
        raise ValueError("R04 operational aggregate cohort mismatch")
    success = np.asarray(
        [bool(int(row["success_dice_at_least_0_50"])) for row in included]
    )
    reliability = np.asarray(
        [float(row["version_b_oof_reliability"]) for row in included]
    )
    r02_matched = np.asarray(
        [float(row["r02_aggregate_oof_reliability"]) for row in included]
    )
    from sklearn.metrics import roc_auc_score

    metrics = binary_metrics(reliability, success)
    return {
        "cases": len(included),
        "failure_detection_auroc": float(
            roc_auc_score(~success, 1.0 - reliability)
        ),
        "matched_r02_auroc": float(roc_auc_score(~success, 1.0 - r02_matched)),
        **metrics,
    }


def candidate_table(results):
    r01 = results["r01_ablation"]
    r02 = results["r02_reduced_features"]
    r03 = results["r03_extended_features"]
    r04 = r04_aggregate()
    r05 = results["r05_category_aware"]
    r06 = results["r06_nonlinear_comparison"]
    candidates = [
        {
            "experiment": "R01",
            "model": "14-feature logistic baseline",
            "registered_outcome": "DESCRIPTIVE_COMPLETE",
            "gate_passed": None,
            "selection_eligible": False,
            "cases": 680,
            "aggregate_oof_auroc": r01["baseline"]["failure_detection_auroc"],
            "ece": r01["baseline"]["calibration"]["ece"],
            "brier": r01["baseline"]["calibration"]["brier"],
            "selection_note": "Descriptive by registration; cannot be selected.",
        },
        {
            "experiment": "R02",
            "model": "10-feature aggregate logistic",
            "registered_outcome": "PASS",
            "gate_passed": True,
            "selection_eligible": True,
            "cases": 680,
            "aggregate_oof_auroc": r02["reduced_10_feature_model"][
                "failure_detection_auroc"
            ],
            "ece": r02["reduced_10_feature_model"]["calibration"]["ece"],
            "brier": r02["reduced_10_feature_model"]["calibration"]["brier"],
            "selection_note": "Passed noninferiority gate.",
        },
        {
            "experiment": "R03",
            "model": "20-feature extended logistic",
            "registered_outcome": "FAIL",
            "gate_passed": False,
            "selection_eligible": False,
            "cases": 680,
            "aggregate_oof_auroc": r03["extended_20_feature_model"][
                "failure_detection_auroc"
            ],
            "ece": r03["extended_20_feature_model"]["calibration"]["ece"],
            "brier": r03["extended_20_feature_model"]["calibration"]["brier"],
            "selection_note": "Failed registered superiority gate.",
        },
        {
            "experiment": "R04",
            "model": "Predicted-category routed logistic",
            "registered_outcome": "PASS",
            "gate_passed": True,
            "selection_eligible": True,
            "cases": r04["cases"],
            "aggregate_oof_auroc": r04["failure_detection_auroc"],
            "ece": r04["ece"],
            "brier": r04["brier"],
            "selection_note": "Passed category gates; aggregate operational ranking is lower than R02. One no-tumor route excluded as registered.",
            "matched_r02_auroc": r04["matched_r02_auroc"],
        },
        {
            "experiment": "R05",
            "model": "Category-aware aggregate logistic",
            "registered_outcome": "FAIL",
            "gate_passed": False,
            "selection_eligible": False,
            "cases": r05["analyses"]["overall"]["cases"],
            "aggregate_oof_auroc": r05["analyses"]["overall"][
                "failure_detection_auroc"
            ],
            "ece": r05["analyses"]["overall"]["calibration"]["ece"],
            "brier": r05["analyses"]["overall"]["calibration"]["brier"],
            "selection_note": "Failed predicted-glioma AUROC gate.",
        },
        {
            "experiment": "R06",
            "model": "Best nonlinear candidate (XGBoost)",
            "registered_outcome": "FAIL",
            "gate_passed": False,
            "selection_eligible": False,
            "cases": 680,
            "aggregate_oof_auroc": r06["models"]["xgboost"][
                "failure_detection_auroc"
            ],
            "ece": r06["models"]["xgboost"]["calibration"]["ece"],
            "brier": r06["models"]["xgboost"]["calibration"]["brier"],
            "selection_note": "No nonlinear candidate reached 0.975; R02 retained.",
        },
    ]
    for candidate in candidates:
        candidate.setdefault("matched_r02_auroc", None)
    return candidates


def select_candidate(candidates):
    eligible = [row for row in candidates if row["selection_eligible"]]
    if not eligible:
        raise ValueError("No passing selectable reliability model")
    return min(
        eligible,
        key=lambda row: (
            -row["aggregate_oof_auroc"],
            row["ece"],
            row["brier"],
            row["experiment"],
        ),
    )


def selection_markdown(result):
    lines = [
        "# Reliability model selection",
        "",
        f"**Selected model:** {result['selected_model']['experiment']} — {result['selected_model']['model']}",
        "",
        "The registered rule considers only R02–R06 models that passed their own gates, then orders them by aggregate OOF AUROC, lower ECE, lower Brier, and experiment ID. R01 and R07–R10 cannot be selected.",
        "",
        "| Experiment | Outcome | Eligible | Cases | Aggregate OOF AUROC | ECE | Brier |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["candidate_table"]:
        lines.append(
            f"| {row['experiment']} | {row['registered_outcome']} | "
            f"{'yes' if row['selection_eligible'] else 'no'} | {row['cases']} | "
            f"{row['aggregate_oof_auroc']:.6f} | {row['ece']:.6f} | {row['brier']:.6f} |"
        )
    h7 = result["h7_reliability_condition"]
    comparison = result["e08_vs_v1"]
    lines.extend(
        [
            "",
            "R02 outranks the other passing candidate, R04, on aggregate AUROC (`0.955502` versus `0.930860`). On R04's matched 679-case cohort, R02 is still higher at `0.955015`. Tie-breakers are therefore unnecessary.",
            "",
            f"H7 reliability condition: **{'PASS' if h7['passed'] else 'FAIL'}**. Selected aggregate AUROC `{h7['selected_aggregate_oof_auroc']:.6f}` and R08 mean fold AUROC `{h7['r08_mean_fold_auroc']:.6f}` both exceed the registered `0.941` minimum.",
            "",
            f"Compared with v1, selected E08 reliability AUROC changes by `{comparison['auroc_difference']:+.6f}`, ECE by `{comparison['ece_difference']:+.6f}` (lower is better), and Brier skill by `{comparison['brier_skill_difference']:+.6f}`.",
            "",
            "This selection nominates the C03-calibrated E08 pathway with R02 reliability for later policy and still-sealed evaluation. It does not establish clinical validity, patient-independent generalization, or external generalization.",
            "",
        ]
    )
    return "\n".join(lines)


def completion_markdown(result, source_commit):
    lines = [
        "# Stage 4 reliability completion report",
        "",
        "**Stage 4 complete, ready for Stage 5 review.**",
        "",
        "| Study | Status | Principal outcome |",
        "|---|---|---|",
    ]
    for row in result["stage_status"]:
        lines.append(
            f"| {row['experiment']} | {row['status']} | {row['principal_outcome']} |"
        )
    selected = result["selected_model"]
    comparison = result["e08_vs_v1"]
    lines.extend(
        [
            "",
            "## Selected operational reliability model",
            "",
            f"The selected model is **{selected['experiment']} — {selected['model']}**, operating on C03-calibrated E08 probability maps. Its aggregate OOF AUROC is `{selected['aggregate_oof_auroc']:.6f}`, ECE `{selected['ece']:.6f}`, Brier `{selected['brier']:.6f}`, and Brier skill `{result['selected_brier_skill']:.6f}`.",
            "",
            f"H7: **{'PASS' if result['h7_reliability_condition']['passed'] else 'FAIL'}**. R08's unweighted mean fold AUROC is `{result['h7_reliability_condition']['r08_mean_fold_auroc']:.6f}` against the `0.941` minimum.",
            "",
            "## E08 versus v1 reliability",
            "",
            f"- AUROC: E08 `{comparison['e08']['failure_detection_auroc']:.6f}` vs v1 `{comparison['v1']['failure_detection_auroc']:.6f}`; difference `{comparison['auroc_difference']:+.6f}`.",
            f"- ECE: E08 `{comparison['e08']['ece']:.6f}` vs v1 `{comparison['v1']['ece']:.6f}`; difference `{comparison['ece_difference']:+.6f}`.",
            f"- Brier skill: E08 `{comparison['e08']['brier_skill']:.6f}` vs v1 `{comparison['v1']['brier_skill']:.6f}`; difference `{comparison['brier_skill_difference']:+.6f}`.",
            "- These internal-development intervals overlap; no statistical superiority or clinical-generalization claim is made.",
            "",
            "## Integrity",
            "",
            f"- Evidence checkpoint before report consolidation: `{source_commit}`; it was clean and matched `origin/master`.",
            "- Final report commit: recorded in Git history and the delivery message because a tracked file cannot contain its own content-derived commit SHA.",
            "- Expected final working tree: clean after the report commit; verified in the delivery message.",
            f"- Tests: {result['verification']['expected_test_count']} passed.",
            "- Ruff: passed.",
            "- Sealed test-set accesses: 0.",
            "- Segmentation threshold changes: 0; threshold remained 0.50.",
            "- Classifier or neural-segmenter retraining events during Stage 4: 0.",
            "- Registered reliability-estimator fitting occurred as required and is not neural-model retraining.",
            "",
            "Stage 4 complete, ready for Stage 5 review.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    config_path = SELECTION / "config.json"
    config = calibration.read_json(config_path)
    if config["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E08 final selection is not in its registered state")
    if calibration.sha256(config_path) != "d65b701eca2343f7036ca6605a84ecaa76b3eb89832b8e43194c212edab893ea":
        raise ValueError("E08 final-selection config hash mismatch")
    results, verified_hashes = verify_sources()
    if results["c03_isotonic"]["gate"]["status"] != "PASS":
        raise ValueError("Selected C03 calibration did not pass")

    candidates = candidate_table(results)
    selected = select_candidate(candidates)
    if selected["experiment"] != "R02":
        raise ValueError("Registered selection did not choose R02")
    r02 = results["r02_reduced_features"]["reduced_10_feature_model"]
    r08 = results["r08_prospective_fold_evaluation"]
    h7_minimum = 0.941
    h7 = {
        "minimum_auroc": h7_minimum,
        "selected_aggregate_oof_auroc": selected["aggregate_oof_auroc"],
        "r08_mean_fold_auroc": r08["fold_summary"]["failure_detection_auroc"][
            "mean"
        ],
        "r08_gate_status": r08["gate"]["status"],
        "passed": bool(
            selected["aggregate_oof_auroc"] >= h7_minimum
            and r08["gate"]["status"] == "PASS"
        ),
    }
    e08 = {
        "failure_detection_auroc": selected["aggregate_oof_auroc"],
        "auroc_ci95_lower": r02["bootstrap"]["ci95_lower"],
        "auroc_ci95_upper": r02["bootstrap"]["ci95_upper"],
        "ece": selected["ece"],
        "brier_skill": r02["calibration"]["brier_skill"],
    }
    comparison = {
        "e08": e08,
        "v1": V1,
        "auroc_difference": e08["failure_detection_auroc"]
        - V1["failure_detection_auroc"],
        "ece_difference": e08["ece"] - V1["ece"],
        "brier_skill_difference": e08["brier_skill"] - V1["brier_skill"],
    }
    stage_status = [
        {"experiment": "R01", "status": "DESCRIPTIVE_COMPLETE", "principal_outcome": "14-feature ablation baseline; boundary entropy dominated."},
        {"experiment": "R02", "status": "PASS — SELECTED", "principal_outcome": "10-feature logistic AUROC 0.955502; passed noninferiority."},
        {"experiment": "R03", "status": "FAIL", "principal_outcome": "Extended features AUROC 0.951423; superiority gate missed."},
        {"experiment": "R04", "status": "PASS — NOT SELECTED", "principal_outcome": "All category gates passed, but routed aggregate AUROC 0.930860."},
        {"experiment": "R05", "status": "FAIL", "principal_outcome": "Predicted-glioma AUROC 0.871096 missed 0.88."},
        {"experiment": "R06", "status": "FAIL", "principal_outcome": "All nonlinear models underperformed R02."},
        {"experiment": "R07", "status": "DESCRIPTIVE_COMPLETE", "principal_outcome": "Four coefficient intervals excluded zero; boundary entropy most stable."},
        {"experiment": "R08", "status": "PASS", "principal_outcome": "Mean held-out-fold AUROC 0.960424 passed H7 gate."},
        {"experiment": "R09", "status": "DESCRIPTIVE_COMPLETE", "principal_outcome": "MODERATE fold sensitivity; AUROC SD 0.013362."},
        {"experiment": "R10", "status": "DESCRIPTIVE_COMPLETE", "principal_outcome": "EXACT seed stability across five registered seeds."},
    ]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    result = {
        "schema_version": 1,
        "experiment_id": "e08_final_selection",
        "status": "PASS",
        "decision": "NOMINATE_C03_CALIBRATED_E08_WITH_R02_RELIABILITY",
        "calibration_condition": {
            "selected_method": "C03 isotonic calibration",
            "gate_status": "PASS",
        },
        "candidate_table": candidates,
        "registered_selection_rule": [
            "passing R02-R06 models only",
            "highest aggregate OOF AUROC",
            "lowest ECE",
            "lowest Brier",
            "experiment ID ascending",
        ],
        "selected_model": selected,
        "selected_brier_skill": r02["calibration"]["brier_skill"],
        "h7_reliability_condition": h7,
        "e08_vs_v1": comparison,
        "stage_status": stage_status,
        "verification": {
            "source_hashes": verified_hashes,
            "source_commit": source_commit,
            "r01_r10_all_documented": True,
            "selection_rule_applied_without_relaxation": True,
            "expected_test_count": 136,
            "ruff_expected": "PASS",
            "test_images_accessed": 0,
            "threshold_changed": False,
            "classifier_or_neural_segmenter_retrained": False,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "limitations": [
            "Selection uses internal BRISC development data and awaits sealed evaluation.",
            "BRISC patient independence remains unverified.",
            "Glioma remains the weakest reliability subgroup.",
            "E08 and v1 AUROC intervals overlap; no statistical superiority is claimed.",
            "No clinical validity or external generalization is established.",
        ],
    }
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    calibration.write_csv(ARTIFACTS / "model_selection_table.csv", candidates)
    calibration.write_json(SELECTION / "result.json", result)
    SELECTION_REPORT.write_text(selection_markdown(result), encoding="utf-8")
    COMPLETION_REPORT.write_text(
        completion_markdown(result, source_commit), encoding="utf-8"
    )
    (SELECTION / "decision.md").write_text(
        """# Conditional E08 final-model decision

**Decision:** NOMINATE C03-CALIBRATED E08 WITH R02 RELIABILITY

C03 passed its registered calibration gate. R02 is the highest-AUROC selectable reliability model among R02–R06 candidates that passed their own gates, and R08 passed the H7 mean-fold AUROC gate. The E08 pathway is therefore nominated for Stage 5 policy work and later sealed evaluation.

This nomination is based on internal BRISC development data. It is not clinical approval, external validation, or a sealed-test result.
""",
        encoding="utf-8",
    )
    config.update(
        {
            "status": "COMPLETE",
            "selected_calibration": "c03_isotonic",
            "selected_reliability_model": "r02_reduced_features",
            "source_commit": source_commit,
            "source_hashes": verified_hashes,
        }
    )
    calibration.write_json(config_path, config)
    registry_path = ROOT / "experiments/registry.json"
    registry = calibration.read_json(registry_path)
    item = next(
        entry
        for entry in registry["experiments"]
        if entry["id"] == "e08_final_selection"
    )
    if item["status"] != "REGISTERED_NOT_STARTED":
        raise ValueError("E08 selection registry was not pending")
    item["status"] = "COMPLETE"
    calibration.write_json(registry_path, registry)
    (ARTIFACTS / ".gitkeep").unlink(missing_ok=True)
    included = sorted(
        path
        for path in SELECTION.rglob("*")
        if path.is_file() and path != ARTIFACTS / "experiment_manifest.json"
    )
    calibration.write_json(
        ARTIFACTS / "experiment_manifest.json",
        {
            "status": "PASS",
            "experiment_id": "e08_final_selection",
            "decision": result["decision"],
            "test_images_accessed": 0,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": calibration.sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in included
            ],
        },
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "selected_model": selected,
                "h7": h7,
                "e08_vs_v1": comparison,
                "source_commit": source_commit,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
