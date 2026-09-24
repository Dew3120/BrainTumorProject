"""Apply the preregistered Stage-2 model-selection rule to E01-E08."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_MD = ROOT / "audit/v2_model_selection.md"
OUTPUT_JSON = ROOT / "audit/v2_model_selection.json"
LABELS = ("glioma", "meningioma", "pituitary")
BASELINE = {
    "glioma": 0.608951638747171,
    "meningioma": 0.8702445032838158,
    "pituitary": 0.808789244003923,
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_result(experiment_id):
    path = ROOT / "experiments" / experiment_id / "result.json"
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("test_images_accessed") != 0:
        raise ValueError(f"{experiment_id} test-access posture changed")
    return result, sha256(path)


def row(experiment_id, name, dice, local_gate, pixel_ece=None, auroc=None):
    delta = {label: dice[label] - BASELINE[label] for label in LABELS}
    return {
        "experiment_id": experiment_id,
        "name": name,
        "dice": dice,
        "delta_vs_v1": delta,
        "macro_type_dice": float(np.mean([dice[label] for label in LABELS])),
        "category_harm_safe": (
            delta["meningioma"] >= -0.01 and delta["pituitary"] >= -0.01
        ),
        "local_gate_passed": local_gate,
        "accepted_pixel_foreground_ece": pixel_ece,
        "case_reliability_auroc": auroc,
    }


def fmt(value):
    return "—" if value is None else f"{value:.4f}"


def main():
    evidence = {}
    results = {}
    for experiment_id in (
        "e01_lcc_filter",
        "e02_morphology",
        "e03_tta",
        "e04_multiscale",
        "e05_glioma_weighted",
        "e06_boundary_loss",
        "e07_glioma_specialized",
        "e08_ensemble",
    ):
        results[experiment_id], evidence[experiment_id] = read_result(experiment_id)
    e01 = results["e01_lcc_filter"]
    e02 = results["e02_morphology"]
    e03 = results["e03_tta"]
    e04 = results["e04_multiscale"]
    e05 = results["e05_glioma_weighted"]
    e06 = results["e06_boundary_loss"]
    e07 = results["e07_glioma_specialized"]
    e08 = results["e08_ensemble"]
    rows = [
        row(
            "e01_lcc_filter",
            "Largest component",
            {label: e01["filtered_mean_dice"][label] for label in LABELS},
            e01["gate_passed"],
        ),
        row(
            "e02_morphology",
            "Morphology 3x3",
            e02["variants"]["kernel_3"]["mean_dice"],
            e02["variants"]["kernel_3"]["gate_passed"],
        ),
        row(
            "e03_tta",
            "Eight-view TTA",
            e03["tta_mean_dice"],
            e03["gate_passed"],
        ),
        row(
            "e04_multiscale",
            "128/256 multiscale",
            e04["multiscale_mean_dice"],
            e04["gate_passed"],
        ),
        row(
            "e05_glioma_weighted",
            "Glioma-weighted retraining",
            e05["candidate_mean_dice"],
            e05["gate_passed"],
            e05["pixel_calibration"]["oof_foreground_ece"],
            e05["case_reliability"]["failure_detection_auroc"],
        ),
        row(
            "e06_boundary_loss",
            "Boundary-loss retraining",
            e06["dice"]["candidate"],
            e06["gate_passed"],
            e06["pixel_calibration"]["oof_foreground_ece"],
            e06["case_reliability"]["failure_detection_auroc"],
        ),
        row(
            "e07_glioma_specialized",
            "Classifier-routed specialist",
            {
                label: e07["routed_validation"][f"val_dice_{label}"]
                for label in LABELS
            },
            e07["gate_passed"],
            e07["pixel_calibration"]["oof_foreground_ece"],
            e07["case_reliability"]["failure_detection_auroc"],
        ),
        row(
            "e08_ensemble",
            "Fixed E05-E07 ensemble",
            e08["mean_dice"]["ensemble"],
            e08["gate_passed"],
        ),
    ]
    eligible = [item for item in rows if item["category_harm_safe"]]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["dice"]["glioma"],
            (
                item["accepted_pixel_foreground_ece"]
                if item["accepted_pixel_foreground_ece"] is not None
                else float("inf")
            ),
            -(
                item["case_reliability_auroc"]
                if item["case_reliability_auroc"] is not None
                else float("-inf")
            ),
            -item["macro_type_dice"],
            item["experiment_id"],
        ),
    )
    selected = ranked[0]
    if selected["experiment_id"] != "e08_ensemble":
        raise ValueError("Predeclared ranking did not select E08")
    parent_delta = selected["delta_vs_v1"]["glioma"]
    parent_passed = parent_delta >= 0.05
    output = {
        "schema_version": 1,
        "status": "COMPLETE",
        "scope": "BRISC internal validation model selection; not clinical validation",
        "candidates": rows,
        "category_harm_eligible_ids": [item["experiment_id"] for item in ranked],
        "selected_experimental_v2": selected["experiment_id"],
        "selection_rule": (
            "filter category harm; then highest glioma Dice, lowest accepted pixel ECE, "
            "highest reliability AUROC, highest macro Dice, experiment ID"
        ),
        "parent_g1": {
            "required_glioma_dice_delta_over_v1": 0.05,
            "observed": parent_delta,
            "passed": parent_passed,
        },
        "accepted_final_model": (
            selected["experiment_id"] if parent_passed else "baseline_v1"
        ),
        "stage3_and_stage4_analysis_target": selected["experiment_id"],
        "result_sha256": evidence,
        "test_images_accessed": 0,
    }
    lines = [
        "# Controlled glioma intervention model selection",
        "",
        "**Scope:** BRISC internal validation only. Patient independence, external generalization, and clinical validity remain unverified.",
        "",
        "| ID | Intervention | Glioma Dice | Meningioma Dice | Pituitary Dice | Glioma Δ vs v1 | Harm-safe | Local gate | Pixel foreground ECE | Reliability AUROC |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|---:|---:|",
    ]
    for item in rows:
        lines.append(
            "| {experiment_id} | {name} | {glioma:.4f} | {meningioma:.4f} | "
            "{pituitary:.4f} | {delta:+.4f} | {harm} | {gate} | {ece} | "
            "{auroc} |".format(
                experiment_id=item["experiment_id"],
                name=item["name"],
                glioma=item["dice"]["glioma"],
                meningioma=item["dice"]["meningioma"],
                pituitary=item["dice"]["pituitary"],
                delta=item["delta_vs_v1"]["glioma"],
                harm="yes" if item["category_harm_safe"] else "no",
                gate="pass" if item["local_gate_passed"] else "fail",
                ece=fmt(item["accepted_pixel_foreground_ece"]),
                auroc=fmt(item["case_reliability_auroc"]),
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "E08 is the best category-harm-safe experimental candidate. It has the highest glioma Dice (`0.6324`) and improves meningioma and pituitary relative to v1. No calibration or reliability tie-break was needed because the glioma-Dice ranking was not tied.",
            "",
            "E08 does **not** replace frozen v1. Its glioma improvement over v1 is only `+0.0234`, below parent G1's required `+0.0500`. Frozen v1 remains the accepted final model. E08 is the locked experimental v2 target for the requested calibration and reliability comparisons; those analyses cannot retroactively change this model-selection decision.",
            "",
            "The ensemble's pass is compatible with all three component models failing locally: averaging can reduce uncorrelated errors. It does not show that any component is individually superior, and every result was obtained on the same internal validation partition used during development.",
            "",
            "No sealed-test image or prediction was accessed.",
            "",
        ]
    )
    OUTPUT_JSON.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
