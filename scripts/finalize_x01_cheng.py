"""Independently verify, document, and close X01 after visual review."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x01_cheng_overlap_control"
ARTIFACTS = EXPERIMENT / "artifacts"
RESULT_PATH = EXPERIMENT / "result.json"
REPORT_PATH = ROOT / "audit/x01_cheng_evaluation.md"
LABELS = ("glioma", "meningioma", "pituitary")
TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def assert_close(observed: float, expected: float, name: str) -> float:
    difference = abs(observed - expected)
    if difference > TOLERANCE:
        raise ValueError(f"Independent recomputation mismatch for {name}: {difference}")
    return difference


def verify_result(result: dict, rows: list[dict]) -> dict:
    if result["status"] != "EVALUATION_COMPLETE_VISUAL_REVIEW_PENDING":
        raise ValueError("X01 is not in the expected pre-finalization state")
    if len(rows) != 3064 or len({row["source_index"] for row in rows}) != 3064:
        raise ValueError("Per-case result membership changed")
    if len({row["patient_id"] for row in rows}) != 233:
        raise ValueError("Per-case patient membership changed")
    maximum = 0.0
    dice = np.asarray([float(row["calibrated_dice"]) for row in rows])
    iou = np.asarray([float(row["calibrated_iou"]) for row in rows])
    raw_dice = np.asarray([float(row["raw_dice"]) for row in rows])
    raw_iou = np.asarray([float(row["raw_iou"]) for row in rows])
    maximum = max(
        maximum,
        assert_close(
            float(dice.mean()),
            result["segmentation"]["slice_macro"]["dice"],
            "slice Dice",
        ),
        assert_close(
            float(iou.mean()), result["segmentation"]["slice_macro"]["iou"], "slice IoU"
        ),
        assert_close(
            float(raw_dice.mean()),
            result["segmentation"]["slice_macro"]["raw_dice"],
            "raw slice Dice",
        ),
        assert_close(
            float(raw_iou.mean()),
            result["segmentation"]["slice_macro"]["raw_iou"],
            "raw slice IoU",
        ),
    )
    patient_rows: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        patient_rows[row["patient_id"]].append(row)
    patient_dice = [
        float(np.mean([float(row["calibrated_dice"]) for row in members]))
        for members in patient_rows.values()
    ]
    maximum = max(
        maximum,
        assert_close(
            float(np.mean(patient_dice)),
            result["segmentation"]["subject_macro"]["dice"]["subject_macro_mean"],
            "subject-macro Dice",
        ),
    )
    for category in LABELS:
        members = [row for row in rows if row["category"] == category]
        category_dice = float(
            np.mean([float(row["calibrated_dice"]) for row in members])
        )
        maximum = max(
            maximum,
            assert_close(
                category_dice,
                result["segmentation"]["per_category"][category]["slice_mean_dice"],
                f"{category} slice Dice",
            ),
        )
        failures = sum(
            row["failure_dice_below_0_50"].lower() == "true" for row in members
        )
        if (
            failures
            != result["segmentation"]["per_category"][category]["failures_below_0_50"]
        ):
            raise ValueError(f"{category} failure count mismatch")
    failure = np.asarray(
        [row["failure_dice_below_0_50"].lower() == "true" for row in rows]
    )
    review = np.asarray([row["policy_status"] == "REVIEW_REQUIRED" for row in rows])
    if int(failure.sum()) != result["segmentation"]["dice_below_0_50"]:
        raise ValueError("Overall failure count mismatch")
    policy = result["reliability_transfer"]["policy"]
    independent_policy = {
        "review_required": int(review.sum()),
        "standard": int((~review).sum()),
        "failures_caught": int((review & failure).sum()),
        "failures_missed": int((~review & failure).sum()),
        "acceptable_reviewed": int((review & ~failure).sum()),
    }
    independent_policy["p01_total_cost"] = (
        independent_policy["failures_caught"]
        + independent_policy["acceptable_reviewed"]
        + 10 * independent_policy["failures_missed"]
    )
    for key, value in independent_policy.items():
        if value != policy[key]:
            raise ValueError(f"Independent policy mismatch: {key}")
    if result["verification"]["brisc_files_opened"]["sealed_test"] != 0:
        raise ValueError("X01 reports sealed BRISC test access")
    if any(
        result["verification"][key]
        for key in (
            "neural_retraining",
            "calibrator_refit",
            "reliability_refit",
            "threshold_changed",
        )
    ):
        raise ValueError("X01 violated a frozen-stack constraint")
    return {
        "per_case_rows": len(rows),
        "patients": len(patient_rows),
        "maximum_metric_difference": maximum,
        "policy": independent_policy,
    }


def manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x01_cheng_overlap_control",
        "status": "COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(paths)
        ],
    }


def main() -> None:
    result = read_json(RESULT_PATH)
    rows = read_csv(ARTIFACTS / "per_case_predictions.csv")
    verification = verify_result(result, rows)
    overlap = read_csv(ARTIFACTS / "brisc_overlap_candidates.csv")
    nearest_test = sum(row["nearest_brisc_original_split"] == "test" for row in overlap)
    exact_nearest_test = sum(
        row["nearest_brisc_original_split"] == "test"
        and int(row["nearest_phash_distance"]) == 0
        for row in overlap
    )
    result["status"] = "COMPLETE"
    result["decision"] = "FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY"
    result["visual_review"] = {
        "status": "PASS_NO_SYSTEMATIC_TECHNICAL_DEFECT",
        "finding": (
            "Reference and prediction overlays are aligned in the deterministic panel. "
            "Worst cases are genuine empty or mislocalized predictions; best cases show "
            "close overlap. No repeated transpose, flip, offset, stretch, or pairing defect "
            "was observed."
        ),
        "panel": "artifacts/visual_audit_panel.png",
    }
    result["independent_recomputation"] = verification
    result["source_equivalent_brisc_metadata_candidates"] = {
        "nearest_candidate_is_original_test": nearest_test,
        "exact_phash_and_nearest_candidate_is_original_test": exact_nearest_test,
        "brisc_test_files_opened": 0,
        "interpretation": "Content-overlap warning for X01; not sealed-test performance.",
    }
    write_json(ARTIFACTS / "evaluation_result.json", result)
    write_json(RESULT_PATH, result)

    classifier = result["classifier"]
    segmentation = result["segmentation"]
    reliability = result["reliability_transfer"]
    calibration = result["pixel_calibration"]
    taxonomy = result["failure_taxonomy"]
    gap = result["domain_gap"]
    report = f"""# X01 Cheng/Figshare overlap-control evaluation

**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY`  
**Audit role:** `ELIGIBLE_OVERLAP_CONTROL`  
**Independent external evidence:** No  
**Visual technical review:** Pass; no systematic alignment or rendering defect

## Population and scope

The frozen E08 ensemble, operational C03 isotonic mapping, R02 reliability model, and exact Stage 5 policy were applied once to 3,064 Cheng CE-T1 slices from 233 patient identifiers. Patient identifiers are the uncertainty-resampling unit. Cheng is a documented upstream source of BRISC, so these results are an overlap-control reproduction and cannot establish external generalization.

The pHash audit found {result["source_equivalent_brisc_metadata_candidates"]["exact_phash_and_nearest_candidate_is_original_test"]} Cheng slices whose nearest exact candidate in frozen BRISC metadata is in the original test split, and {result["source_equivalent_brisc_metadata_candidates"]["nearest_candidate_is_original_test"]} whose nearest candidate is in that split at any recorded distance. No BRISC test file was opened. This content-overlap warning is another reason X01 is not sealed-test evidence.

## Classifier

Patient-level accuracy was `{classifier["accuracy"]:.4f}` (95% patient-bootstrap CI `{classifier["bootstrap_ci95"]["accuracy"][0]:.4f}–{classifier["bootstrap_ci95"]["accuracy"][1]:.4f}`). Top-label ECE was `{classifier["top_label_ece"]:.4f}`. Recall was glioma `{classifier["recall"]["glioma"]:.4f}`, meningioma `{classifier["recall"]["meningioma"]:.4f}`, and pituitary `{classifier["recall"]["pituitary"]:.4f}`. All classifier reference gates passed. Tumor/no-tumor specificity and AUROC are not estimable because Cheng contains no no-tumor cases.

## Segmentation

| Category | Patients | Subject-macro Dice | 95% patient-bootstrap CI | Registered gate | Outcome |
|---|---:|---:|---:|---:|---|
| Glioma | {segmentation["per_category"]["glioma"]["dice"]["subjects"]} | {segmentation["per_category"]["glioma"]["dice"]["subject_macro_mean"]:.4f} | {segmentation["per_category"]["glioma"]["dice"]["bootstrap_ci95"][0]:.4f}–{segmentation["per_category"]["glioma"]["dice"]["bootstrap_ci95"][1]:.4f} | ≥0.55 | **Fail** |
| Meningioma | {segmentation["per_category"]["meningioma"]["dice"]["subjects"]} | {segmentation["per_category"]["meningioma"]["dice"]["subject_macro_mean"]:.4f} | {segmentation["per_category"]["meningioma"]["dice"]["bootstrap_ci95"][0]:.4f}–{segmentation["per_category"]["meningioma"]["dice"]["bootstrap_ci95"][1]:.4f} | ≥0.75 | Pass |
| Pituitary | {segmentation["per_category"]["pituitary"]["dice"]["subjects"]} | {segmentation["per_category"]["pituitary"]["dice"]["subject_macro_mean"]:.4f} | {segmentation["per_category"]["pituitary"]["dice"]["bootstrap_ci95"][0]:.4f}–{segmentation["per_category"]["pituitary"]["dice"]["bootstrap_ci95"][1]:.4f} | ≥0.75 | Pass |

Overall subject-macro Dice was `{segmentation["subject_macro"]["dice"]["subject_macro_mean"]:.4f}`. There were `{segmentation["dice_below_0_50"]}` Dice-below-0.50 slices and `{segmentation["zero_dice"]}` zero-Dice slices. Whole-volume Dice, tumor volume, and depth are `NOT_ESTIMABLE` because per-slice orientation and ordering are unavailable.

## Calibration and reliability

C03 improved lesion-centred pixel NLL from `{calibration["lesion_centered"]["raw"]["nll"]:.4f}` to `{calibration["lesion_centered"]["c03_operational"]["nll"]:.4f}`, Brier from `{calibration["lesion_centered"]["raw"]["brier"]:.4f}` to `{calibration["lesion_centered"]["c03_operational"]["brier"]:.4f}`, and ECE from `{calibration["lesion_centered"]["raw"]["ece"]:.4f}` to `{calibration["lesion_centered"]["c03_operational"]["ece"]:.4f}`. Whole-image ECE after C03 was `{calibration["whole_image"]["c03_operational"]["ece"]:.4f}`.

R02 retained strong failure ranking: AUROC `{reliability["failure_detection_auroc"]:.4f}` (95% patient-cluster bootstrap CI `{reliability["bootstrap_ci95"]["failure_detection_auroc"][0]:.4f}–{reliability["bootstrap_ci95"]["failure_detection_auroc"][1]:.4f}`). Its ECE was `{reliability["ece"]:.4f}`, above the registered `0.05` gate, so reliability calibration failed even though AUROC and Brier skill `{reliability["brier_skill"]:.4f}` passed. The full-precision policy sent `{reliability["policy"]["review_required"]}` slices (`{reliability["policy"]["review_rate"]:.1%}`) to review, caught `{reliability["policy"]["failures_caught"]}` failures, and missed `{reliability["policy"]["failures_missed"]}`.

## Failure taxonomy

Of the `{taxonomy["all"]["count"]}` weak slices, `{taxonomy["all"]["complete_miss"]["count"]}` (`{taxonomy["all"]["complete_miss"]["fraction"]:.1%}`) were complete misses, `{taxonomy["all"]["severe_under_prediction"]["count"]}` (`{taxonomy["all"]["severe_under_prediction"]["fraction"]:.1%}`) were severe under-predictions, and `{taxonomy["all"]["fragmented_prediction"]["count"]}` (`{taxonomy["all"]["fragmented_prediction"]["fraction"]:.1%}`) were fragmented. Glioma contributed `{taxonomy["glioma"]["count"]}` weak slices. The visual panel confirms these are model/task failures rather than a shared transform defect.

## Domain gap

Relative to BRISC development validation, the frozen classifier encoder centroid shifted by `{gap["encoder_representation_shift"]["centroid_euclidean"]:.4f}` and the mean per-dimension Wasserstein distance was `{gap["encoder_representation_shift"]["dimension_mean_wasserstein"]:.4f}`. R02 score Wasserstein distance was `{gap["reliability_score_wasserstein"]:.4f}`. Policy review rate rose from `{gap["policy_review_rate"]["brisc_operational_reference"]:.1%}` on the BRISC operational reference to `{gap["policy_review_rate"]["cheng"]:.1%}` on Cheng.

## What X01 establishes

- The frozen pipeline can be transported technically to the official Cheng v8 MATLAB release without retraining, refitting, threshold changes, or geometry failures.
- Classifier category recognition transported strongly within this overlap-control cohort.
- Meningioma and pituitary segmentation passed the registered reference gates.
- Glioma segmentation did not pass, and reliability scores ranked failures but were miscalibrated under the registered ECE criterion.

## What X01 does not establish

- It does not establish independent cross-dataset generalization because Cheng is an upstream BRISC source.
- It is not a sealed BRISC test evaluation; no BRISC test file was opened and no sealed-test claim is made.
- It does not establish performance on no-tumor scans, prospective patients, other scanners, or 3D volumes.
- It does not establish clinical validity, diagnostic accuracy, safety, or suitability for patient care.

## Verification

- Independent per-case recomputation maximum difference: `{verification["maximum_metric_difference"]}`.
- Bootstrap: 2,000 patient-cluster resamples, seed 42.
- Segmentation threshold: exactly `0.50`; reliability-policy threshold: exactly `0.9490250173235835`.
- BRISC development-validation files opened for registered domain reference: 680; sealed-test files opened: 0.
- Neural retraining, calibrator refit, reliability refit, and threshold changes: 0.
"""
    REPORT_PATH.write_text(report, encoding="utf-8", newline="\n")

    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if config["status"] != "AUDIT_COMPLETE":
        raise ValueError("X01 config lifecycle state changed")
    config["status"] = "COMPLETE"
    config["access_state"] = "EVALUATED_ONCE_FROZEN_STACK"
    config["test_images_accessed"] = 0
    config["model_retrained_or_refit"] = False
    write_json(config_path, config)

    decision_path = EXPERIMENT / "decision.md"
    decision_path.write_text(
        "# X01 decision\n\n"
        "**Decision:** `FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY`\n\n"
        "The classifier and two segmentation categories passed, but glioma subject-macro "
        "Dice and R02 reliability ECE failed their registered gates. The gate was not "
        "relaxed. Cheng remains an overlap control and supplies no independent external "
        "generalization evidence.\n",
        encoding="utf-8",
        newline="\n",
    )
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    matches = [
        item
        for item in registry["experiments"]
        if item["id"] == "x01_cheng_overlap_control"
    ]
    if len(matches) != 1 or matches[0]["status"] != "AUDIT_COMPLETE":
        raise ValueError("X01 registry lifecycle state changed")
    matches[0]["status"] = "COMPLETE"
    write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Reject X01 transport gates and retain overlap-control findings

- **Decision made:** Complete X01 as `FAIL_REGISTERED_TRANSPORT_GATES_OVERLAP_CONTROL_ONLY`. Accept the technically valid frozen run and visual review, but reject the registered transport gate because glioma Dice and R02 ECE failed.
- **Triggering artifact:** `experiments/x01_cheng_overlap_control/artifacts/evaluation_result.json`, SHA256 `{sha256(ARTIFACTS / "evaluation_result.json")}`; deterministic visual panel SHA256 `{sha256(ARTIFACTS / "visual_audit_panel.png")}`.
- **Rationale:** Glioma subject-macro Dice was `{segmentation["per_category"]["glioma"]["dice"]["subject_macro_mean"]:.6f}` against `0.55`. Reliability AUROC remained `{reliability["failure_detection_auroc"]:.6f}`, but ECE was `{reliability["ece"]:.6f}` against `0.05`. Meningioma and pituitary Dice and all classifier gates passed. Visual inspection found aligned references/predictions and no common transform defect.
- **Verification:** Independent recomputation over 3,064 rows had maximum metric difference `{verification["maximum_metric_difference"]}`. Frozen hashes matched; 2,000 patient-cluster bootstraps used seed 42. No BRISC sealed-test file was opened. Neural retraining, calibrator/reliability refit, segmentation-threshold changes, and gate relaxation were zero.
- **Alternatives considered:** Relax the glioma or ECE gates because ranking AUROC was strong; rejected. Present X01 as independent external validation; rejected because Cheng is an upstream BRISC source. Tune the C03 mapping or reliability threshold on Cheng; rejected as forbidden external-data fitting.
- **Who approved:** User approved Amendment 13, the frozen X01 overlap-control evaluation, and unchanged fail-closed gates.
"""
    with (ROOT / "audit/review_log.md").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(review_entry)

    paths = [
        ROOT / "audit/x01_cheng_dataset_audit.md",
        REPORT_PATH,
        ROOT / "scripts/audit_x01_cheng.py",
        ROOT / "scripts/evaluate_x01_cheng.py",
        ROOT / "scripts/finalize_x01_cheng.py",
        ROOT / "tests/test_x01_cheng_audit.py",
        ROOT / "tests/test_x01_cheng_evaluation.py",
        ROOT / "scripts/validate_experiment_tracking.py",
        ROOT / "tests/test_foundation_controls.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        RESULT_PATH,
        decision_path,
    ]
    paths.extend(
        path
        for path in ARTIFACTS.iterdir()
        if path.is_file() and path.name not in {".gitkeep", "experiment_manifest.json"}
    )
    write_json(ARTIFACTS / "experiment_manifest.json", manifest(paths))
    print(
        json.dumps(
            {
                "status": result["status"],
                "decision": result["decision"],
                "verification": verification,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
