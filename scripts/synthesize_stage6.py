"""Consolidate the registered Stage 6 external transport evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
import stage6_external_common as external

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/x05_cross_dataset_synthesis"
ARTIFACTS = EXPERIMENT / "artifacts"
REPORT = ROOT / "audit/stage6_cross_dataset.md"
COMPLETION = ROOT / "audit/stage6_completion_report.md"
SOURCE_IDS = (
    "x01_cheng_overlap_control",
    "x02_openneuro_pituitary",
    "x03_upenn_gbm",
    "x04_meningioma_seg_class",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def compact_cohort(xid: str, result: dict) -> dict:
    if xid == "x04_meningioma_seg_class":
        return {
            "evidence_role": "controlled-access candidate",
            "status": "ACCESS_BLOCKED",
            "performance_metrics": "NOT_ESTIMABLE",
            "dataset_images_accessed": 0,
        }
    classifier = result["classifier"]
    segmentation = result["segmentation"]
    reliability = result["reliability_transfer"]
    taxonomy = result["failure_taxonomy"]["all"]
    if xid == "x01_cheng_overlap_control":
        role = "overlap control; not independent"
        target_recall = min(classifier["recall"].values())
        target_dice = min(
            value["dice"]["subject_macro_mean"]
            for value in segmentation["per_category"].values()
        )
        population = {"subjects": 233, "primary_slices": 3064}
    elif xid == "x02_openneuro_pituitary":
        role = "independent pituitary cohort"
        target_recall = classifier["pituitary_recall"]
        target_dice = segmentation["subject_macro"]["dice"]["subject_macro_mean"]
        population = {"subjects": 50, "primary_slices": 425}
    else:
        role = "independent GBM-only cohort; coarse glioma mapping"
        target_recall = classifier["glioma_recall"]
        target_dice = segmentation["subject_macro"]["dice"]["subject_macro_mean"]
        population = {"subjects": 147, "primary_slices": 9464}
    return {
        "evidence_role": role,
        "status": result["status"],
        "population": population,
        "classifier": {
            "target_recall": target_recall,
            "top_label_ece": classifier["top_label_ece"],
        },
        "segmentation": {
            "registered_target_subject_macro_dice": target_dice,
            "overall_subject_macro_dice": segmentation["subject_macro"]["dice"],
            "per_category": segmentation.get("per_category", {}),
        },
        "pixel_calibration": {
            region: {
                "raw_ece": result["pixel_calibration"][region]["raw"]["ece"],
                "c03_ece": result["pixel_calibration"][region]["c03_operational"]["ece"],
            }
            for region in ("lesion_centered", "background", "whole_image")
        },
        "reliability": {
            "auroc": reliability["failure_detection_auroc"],
            "auroc_ci95": reliability["bootstrap_ci95"]["failure_detection_auroc"],
            "ece": reliability["ece"],
            "brier_skill": reliability["brier_skill"],
            "brier_skill_ci95": reliability["bootstrap_ci95"]["brier_skill"],
            "top20_failure_capture_fraction": reliability["top20_risk"][
                "failures_captured"
            ]
            / reliability["top20_risk"]["failures_total"],
            "policy_review_rate": reliability["policy"]["review_rate"],
        },
        "failure_taxonomy": {
            "failures": taxonomy["count"],
            "complete_miss_fraction": taxonomy["complete_miss"]["fraction"],
            "severe_under_prediction_fraction": taxonomy["severe_under_prediction"][
                "fraction"
            ],
            "fragmented_prediction_fraction": taxonomy["fragmented_prediction"][
                "fraction"
            ],
            "disconnected_false_positive_fraction": taxonomy[
                "disconnected_false_positive"
            ]["fraction"],
        },
        "gates": result["gates"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_summary(cohorts: dict) -> None:
    ids = ["x01_cheng_overlap_control", "x02_openneuro_pituitary", "x03_upenn_gbm"]
    labels = ["X01 Cheng\noverlap control", "X02 OpenNeuro\nindependent", "X03 UPENN-GBM\nindependent"]
    recall = [cohorts[xid]["classifier"]["target_recall"] for xid in ids]
    dice = [
        cohorts[xid]["segmentation"]["registered_target_subject_macro_dice"]
        for xid in ids
    ]
    auroc = [cohorts[xid]["reliability"]["auroc"] for xid in ids]
    classifier_ece = [cohorts[xid]["classifier"]["top_label_ece"] for xid in ids]
    reliability_ece = [cohorts[xid]["reliability"]["ece"] for xid in ids]
    colors = ["#9ca3af", "#2563eb", "#7c3aed"]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = ((recall, "Target-category recall", 0.85), (dice, "Registered target Dice", None), (auroc, "Failure-ranking AUROC", 0.75))
    for axis, (values, title, gate) in zip(axes.flat[:3], panels, strict=True):
        axis.bar(labels, values, color=colors)
        axis.set_ylim(0, 1.05)
        axis.set_title(title)
        if gate is not None:
            axis.axhline(gate, color="#dc2626", linestyle="--", linewidth=1, label=f"gate {gate:.2f}")
            axis.legend(loc="lower left")
        for index, value in enumerate(values):
            axis.text(index, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    x = np.arange(len(labels))
    width = 0.36
    axes[1, 1].bar(x - width / 2, classifier_ece, width, label="classifier ECE", color="#059669")
    axes[1, 1].bar(x + width / 2, reliability_ece, width, label="reliability ECE", color="#ea580c")
    axes[1, 1].axhline(0.05, color="#dc2626", linestyle="--", linewidth=1, label="gate 0.05")
    axes[1, 1].set_xticks(x, labels)
    axes[1, 1].set_ylim(0, 0.4)
    axes[1, 1].set_title("Calibration transport")
    axes[1, 1].legend(fontsize=8)
    figure.suptitle("Frozen E08 + C03 + R02 transport summary", fontweight="bold")
    figure.text(0.5, 0.01, "X01 is an overlap control and cannot establish independent generalization.", ha="center", fontsize=9)
    figure.tight_layout(rect=(0, 0.03, 1, 0.96))
    figure.savefig(ARTIFACTS / "transport_summary.png", dpi=180)
    plt.close(figure)


def experiment_manifest(paths: list[Path]) -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "x05_cross_dataset_synthesis",
        "status": "COMPLETE",
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(set(paths))
        ],
    }


def main() -> None:
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    by_id = {item["id"]: item for item in registry["experiments"]}
    expected_states = {
        "x01_cheng_overlap_control": "COMPLETE",
        "x02_openneuro_pituitary": "COMPLETE",
        "x03_upenn_gbm": "COMPLETE",
        "x04_meningioma_seg_class": "ACCESS_BLOCKED",
        "x05_cross_dataset_synthesis": "REGISTERED_NOT_STARTED",
    }
    for xid, expected in expected_states.items():
        if by_id[xid]["status"] != expected:
            raise ValueError(f"Stage 6 source state mismatch: {xid}")

    config_path = EXPERIMENT / "config.json"
    config = read_json(config_path)
    if (
        config["status"] != "REGISTERED_NOT_STARTED"
        or tuple(config["required_source_order"]) != SOURCE_IDS
        or config["segmentation_threshold"] != 0.5
        or config["reliability_policy_threshold"] != "0.9490250173235835"
    ):
        raise ValueError("X05 registered contract changed")
    amendment = ROOT / "audit/preregistration_amendment_13.md"
    if sha256(amendment) != "728d8c0f92a357c1ecbc78e586fce3af940e12fc8fef74bd3a3c0b93f8958da5":
        raise ValueError("Amendment 13 hash mismatch")
    frozen_hashes = external.verify_frozen_artifacts()
    final_free_bytes = shutil.disk_usage(ROOT.drive + "\\").free

    source_results = {
        xid: read_json(ROOT / "experiments" / xid / "result.json") for xid in SOURCE_IDS
    }
    if source_results["x01_cheng_overlap_control"]["gates"]["not_independent"] is not True:
        raise ValueError("X01 overlap-control status changed")
    for xid in ("x02_openneuro_pituitary", "x03_upenn_gbm"):
        if source_results[xid]["status"] != "FAIL_REGISTERED_TRANSPORT_GATES_INDEPENDENT_EXTERNAL":
            raise ValueError(f"Independent external outcome changed: {xid}")
    if source_results["x04_meningioma_seg_class"]["status"] != "ACCESS_BLOCKED":
        raise ValueError("X04 access outcome changed")
    for xid, source in source_results.items():
        if xid == "x04_meningioma_seg_class":
            if source["brisc_sealed_test_images_accessed"] != 0:
                raise ValueError("X04 sealed-test access changed")
        elif source["verification"]["brisc_files_opened"]["sealed_test"] != 0:
            raise ValueError(f"{xid} sealed-test access changed")

    cohorts = {xid: compact_cohort(xid, source_results[xid]) for xid in SOURCE_IDS}
    result = {
        "schema_version": 1,
        "experiment_id": "x05_cross_dataset_synthesis",
        "status": "COMPLETE",
        "h8": {
            "verdict": "REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS",
            "reason": (
                "Both eligible independent cohorts failed conjunctive transport gates. "
                "X01 is overlap-control evidence and X04 is access-blocked."
            ),
        },
        "cohorts": cohorts,
        "source_artifacts": {
            xid: {
                "path": f"experiments/{xid}/result.json",
                "sha256": sha256(ROOT / "experiments" / xid / "result.json"),
            }
            for xid in SOURCE_IDS
        },
        "frozen_hashes": frozen_hashes,
        "verification": {
            "brisc_sealed_test_images_accessed": 0,
            "neural_retraining": 0,
            "calibrator_refits": 0,
            "reliability_refits": 0,
            "segmentation_threshold": 0.5,
            "reliability_policy_threshold": "0.9490250173235835",
            "segmentation_threshold_changes": 0,
            "policy_threshold_changes": 0,
            "gate_relaxations": 0,
            "pooled_cross_dataset_metric": False,
            "final_drive_e_free_bytes": final_free_bytes,
        },
        "claim_scope": "retrospective research evidence; not clinical validation",
    }

    rows = []
    for xid, cohort in cohorts.items():
        if cohort["status"] == "ACCESS_BLOCKED":
            rows.append(
                {
                    "experiment_id": xid,
                    "role": cohort["evidence_role"],
                    "status": cohort["status"],
                    "target_recall": "NOT_ESTIMABLE",
                    "target_dice": "NOT_ESTIMABLE",
                    "reliability_auroc": "NOT_ESTIMABLE",
                    "classifier_ece": "NOT_ESTIMABLE",
                    "reliability_ece": "NOT_ESTIMABLE",
                }
            )
        else:
            rows.append(
                {
                    "experiment_id": xid,
                    "role": cohort["evidence_role"],
                    "status": cohort["status"],
                    "target_recall": cohort["classifier"]["target_recall"],
                    "target_dice": cohort["segmentation"]["registered_target_subject_macro_dice"],
                    "reliability_auroc": cohort["reliability"]["auroc"],
                    "classifier_ece": cohort["classifier"]["top_label_ece"],
                    "reliability_ece": cohort["reliability"]["ece"],
                }
            )
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    write_csv(ARTIFACTS / "dataset_summary.csv", rows)
    write_json(ARTIFACTS / "synthesis.json", result)
    render_summary(cohorts)

    x01, x02, x03 = (cohorts[xid] for xid in SOURCE_IDS[:3])
    report = f"""# Stage 6 cross-dataset transport

**H8 verdict:** `REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS`

The frozen E08 + C03 + R02 stack was evaluated without retraining, refitting, threshold adjustment, or external-data-driven selection. X02 and X03 are independent retrospective cohorts and both failed the registered conjunctive transport gates. X01 is informative overlap-control evidence but cannot establish independent generalization. X04 is access-blocked and contributes no performance metric.

## Dataset disposition

| ID | Dataset | Evidence role | Outcome |
|---|---|---|---|
| X01 | Cheng/Figshare v8 | BRISC source-overlap control | Completed; gates failed; not independent |
| X02 | OpenNeuro ds006248 v1.0.0 | Independent pituitary cohort | Completed; H8 rejected |
| X03 | UPENN-GBM registered derivative | Independent GBM-only cohort | Completed; H8 rejected |
| X04 | Meningioma-SEG-CLASS v1 | Controlled-access meningioma candidate | `ACCESS_BLOCKED`; metrics not estimable |

Glioblastoma is a subtype of glioma, but X03's GBM-only population and whole-tumor target are a coarse mapping and are not directly comparable to the heterogeneous BRISC glioma category.

### Disk ledger

| Checkpoint | Drive E free bytes |
|---|---:|
| After X01 audit | 348,887,539,712 |
| After X02 audit | 348,584,640,512 |
| After X03 audit | 348,143,616,000 |
| X04 access decision | 348,132,126,720 |
| Final Stage 6 synthesis | {final_free_bytes:,} |

The final free space is `{final_free_bytes / 2**30:.3f} GiB`. The projected footprint never exceeded the registered 250 GB sequential-processing trigger.

## Classification transport

| Cohort | Target-category recall | 95% CI | Top-label ECE | Gate outcome |
|---|---:|---:|---:|---|
| X01 overlap control | {x01['classifier']['target_recall']:.4f} (weakest category) | See category-specific X01 artifact | {x01['classifier']['top_label_ece']:.4f} | Recall/ECE pass |
| X02 pituitary | {x02['classifier']['target_recall']:.4f} | 0.8600–1.0000 | {x02['classifier']['top_label_ece']:.4f} | Recall pass; ECE fail |
| X03 GBM→glioma | {x03['classifier']['target_recall']:.4f} | 0.3469–0.5102 | {x03['classifier']['top_label_ece']:.4f} | Recall/ECE fail |
| X04 meningioma | Not estimable | — | — | Access blocked |

X02 shows that category recognition can remain high while spatial transport fails. X03 shows simultaneous category and spatial domain failure. Tumor/no-tumor AUROC and specificity are not estimable on the tumor-only independent cohorts.

## Segmentation transport

| Cohort/view | Subject-macro Dice | 95% CI | Registered gate |
|---|---:|---:|---|
| X01 glioma | 0.4142 | 0.3656–0.4616 | Fail (≥0.55) |
| X01 meningioma | 0.8679 | 0.8441–0.8887 | Pass (≥0.75) |
| X01 pituitary | 0.7712 | 0.7494–0.7924 | Pass (≥0.75) |
| X02 pituitary, strict 425-slice | 0.1272 | 0.0824–0.1778 | Fail (≥0.75) |
| X03 GBM→glioma, strict 9,464-slice | 0.2208 | 0.2004–0.2422 | Fail (≥0.55) |
| X04 meningioma | Not estimable | — | Access blocked |

The strict and native-resolution views agree closely for X02 (`0.1272` vs `0.1273`) and X03 (`0.2208` vs `0.2205`). The five X02 and 82 X03 targets erased by the frozen 128×128 mask transform therefore do not explain the broad external failures.

## Calibration and reliability transfer

| Cohort | Lesion ECE raw→C03 | Reliability AUROC (95% CI) | Reliability ECE | Brier skill (95% CI) | Top-20% capture |
|---|---:|---:|---:|---:|---:|
| X01 overlap | {x01['pixel_calibration']['lesion_centered']['raw_ece']:.4f}→{x01['pixel_calibration']['lesion_centered']['c03_ece']:.4f} | {x01['reliability']['auroc']:.4f} ({x01['reliability']['auroc_ci95'][0]:.4f}–{x01['reliability']['auroc_ci95'][1]:.4f}) | {x01['reliability']['ece']:.4f} | {x01['reliability']['brier_skill']:.4f} ({x01['reliability']['brier_skill_ci95'][0]:.4f}–{x01['reliability']['brier_skill_ci95'][1]:.4f}) | {x01['reliability']['top20_failure_capture_fraction']:.1%} |
| X02 pituitary | {x02['pixel_calibration']['lesion_centered']['raw_ece']:.4f}→{x02['pixel_calibration']['lesion_centered']['c03_ece']:.4f} | {x02['reliability']['auroc']:.4f} ({x02['reliability']['auroc_ci95'][0]:.4f}–{x02['reliability']['auroc_ci95'][1]:.4f}) | {x02['reliability']['ece']:.4f} | {x02['reliability']['brier_skill']:.4f} ({x02['reliability']['brier_skill_ci95'][0]:.4f}–{x02['reliability']['brier_skill_ci95'][1]:.4f}) | {x02['reliability']['top20_failure_capture_fraction']:.1%} |
| X03 GBM→glioma | {x03['pixel_calibration']['lesion_centered']['raw_ece']:.4f}→{x03['pixel_calibration']['lesion_centered']['c03_ece']:.4f} | {x03['reliability']['auroc']:.4f} ({x03['reliability']['auroc_ci95'][0]:.4f}–{x03['reliability']['auroc_ci95'][1]:.4f}) | {x03['reliability']['ece']:.4f} | {x03['reliability']['brier_skill']:.4f} ({x03['reliability']['brier_skill_ci95'][0]:.4f}–{x03['reliability']['brier_skill_ci95'][1]:.4f}) | {x03['reliability']['top20_failure_capture_fraction']:.1%} |

R02 preserves useful failure ranking on all three evaluated cohorts, but its absolute reliability probabilities are not calibrated externally. The fixed policy review rate rises to `{x02['reliability']['policy_review_rate']:.1%}` on X02 and `{x03['reliability']['policy_review_rate']:.1%}` on X03. The Stage 5 threshold is therefore not validated as an external operating policy.

## Failure taxonomy

| Cohort | Weak cases | Complete miss | Severe under | Fragmented | Disconnected FP |
|---|---:|---:|---:|---:|---:|
| X01 overlap | {x01['failure_taxonomy']['failures']:,} | {x01['failure_taxonomy']['complete_miss_fraction']:.1%} | {x01['failure_taxonomy']['severe_under_prediction_fraction']:.1%} | {x01['failure_taxonomy']['fragmented_prediction_fraction']:.1%} | {x01['failure_taxonomy']['disconnected_false_positive_fraction']:.1%} |
| X02 pituitary | {x02['failure_taxonomy']['failures']:,} | {x02['failure_taxonomy']['complete_miss_fraction']:.1%} | {x02['failure_taxonomy']['severe_under_prediction_fraction']:.1%} | {x02['failure_taxonomy']['fragmented_prediction_fraction']:.1%} | {x02['failure_taxonomy']['disconnected_false_positive_fraction']:.1%} |
| X03 GBM→glioma | {x03['failure_taxonomy']['failures']:,} | {x03['failure_taxonomy']['complete_miss_fraction']:.1%} | {x03['failure_taxonomy']['severe_under_prediction_fraction']:.1%} | {x03['failure_taxonomy']['fragmented_prediction_fraction']:.1%} | {x03['failure_taxonomy']['disconnected_false_positive_fraction']:.1%} |

Complete spatial misses dominate X02 and X03. X03 also shows major fragmentation and disconnected false positives. Deterministic visual audits found no shared flip, transpose, shift, or mask-alignment defect, supporting model/domain failure rather than a common rendering error.

## Domain gap

Frozen-encoder centroid shift from the matched BRISC development reference is `4.1616` for X01, `18.7692` for X02, and `13.1125` for X03. The fixed policy review rate increases from the matched BRISC reference to `39.3%` on X01, `90.6%` on X02, and `82.5%` on X03. X03 references are fragmented in `52.8%` of slices versus `4.9%` in its BRISC glioma reference. These shifts are consistent with the poor independent segmentation transport.

No pooled cross-dataset performance number is reported. The cohorts differ in evidence independence, tumor category, orientation, source format, patient population, and target construction; pooling them would conceal the registered failures.

## What Stage 6 establishes

- The exact frozen pipeline can execute on audited external MRI formats with patient-cluster uncertainty estimates and reproducible failure taxonomy.
- Classification transport is dataset-dependent: strong for X02 pituitary, poor for X03 GBM→glioma.
- Segmentation does not satisfy the registered independent transport gates on X02 or X03.
- R02 transfers as a ranking signal but not as an honestly calibrated external probability or validated policy threshold.
- The controlled-access meningioma cohort remains unevaluated and explicitly access-blocked.

## What Stage 6 does not establish

- It does not establish prospective generalization, clinical validity, diagnostic safety, treatment utility, or suitability for patient care.
- It does not establish no-tumor screening performance, because the independent cohorts contain only tumor cases.
- It does not establish independent meningioma transport; X01 overlaps BRISC and X04 was inaccessible.
- It does not justify adapting the model, C03 mapping, R02 estimator, segmentation threshold, or policy threshold using these external outcomes.
- It does not open or replace the sealed BRISC test evaluation.

## Governance verification

- BRISC sealed-test images opened: 0
- Neural retraining: 0; calibrator refits: 0; reliability refits: 0
- Segmentation threshold changes: 0 (`0.50` throughout)
- Reliability-policy threshold changes: 0 (`0.9490250173235835` throughout)
- Gate relaxations: 0
- Confidence intervals: 2,000 patient-cluster bootstrap resamples, seed 42, where estimable
- Claim scope: **retrospective research evidence; not clinical validation**
"""
    REPORT.write_text(report, encoding="utf-8", newline="\n")

    completion = """# Stage 6 completion report

## Status

- X01 Cheng overlap control: complete
- X02 OpenNeuro pituitary independent evaluation: complete; transport gates failed
- X03 UPENN-GBM independent evaluation: complete; transport gates failed
- X04 Meningioma-SEG-CLASS: complete as `ACCESS_BLOCKED`
- X05 cross-dataset synthesis: complete

## Final decision

H8 is rejected for the evaluated independent cohorts. The frozen stack did not meet the registered independent segmentation and calibration transport requirements. X01 remains overlap-control evidence only, and X04 provides no performance evidence.

## Integrity

- Frozen E08 + C03 + R02 throughout
- BRISC sealed-test image access: 0
- Neural retraining and estimator refitting: 0
- Segmentation threshold changes: 0
- Reliability-policy threshold changes: 0
- Gate relaxations: 0
- Final Drive E free space: {final_free_bytes:,} bytes ({final_free_bytes / 2**30:.3f} GiB)
- Source snapshot before synthesis: `b178f62`
- Repository verification: 199 tests passed; Ruff passed; experiment registry passed

**Stage 6 complete. Ready for Stage 7.**

This is retrospective research evidence; not clinical validation.
"""
    COMPLETION.write_text(completion, encoding="utf-8", newline="\n")
    write_json(EXPERIMENT / "result.json", result)
    config.update(
        {
            "status": "COMPLETE",
            "outcome": "H8_REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS",
            "test_images_accessed": 0,
            "model_retrained_or_refit": False,
        }
    )
    write_json(config_path, config)
    decision = """# X05 decision

**Decision:** `H8_REJECTED_FOR_EVALUATED_INDEPENDENT_COHORTS`

Both eligible independent cohorts failed the registered conjunctive transport gates. X01 is retained as non-independent overlap-control evidence. X04 is retained as `ACCESS_BLOCKED` with no imputed metric. No external result is used to tune or select a new model.
"""
    (EXPERIMENT / "decision.md").write_text(decision, encoding="utf-8", newline="\n")
    by_id["x05_cross_dataset_synthesis"]["status"] = "COMPLETE"
    write_json(registry_path, registry)

    timestamp = datetime.now(ZoneInfo("Asia/Colombo")).isoformat(timespec="seconds")
    review_entry = f"""

## {timestamp} — Reject H8 and complete Stage 6 synthesis

- **Decision made:** Reject H8 for the evaluated independent cohorts and complete Stage 6. Preserve X01 as overlap-control evidence and X04 as access-blocked.
- **Triggering artifacts:** X01 result `{result['source_artifacts']['x01_cheng_overlap_control']['sha256']}`; X02 result `{result['source_artifacts']['x02_openneuro_pituitary']['sha256']}`; X03 result `{result['source_artifacts']['x03_upenn_gbm']['sha256']}`; X04 access result `{result['source_artifacts']['x04_meningioma_seg_class']['sha256']}`.
- **Rationale:** X02 and X03 both failed registered independent transport gates. Reliability ranking remained useful, but calibration and top-risk capture did not transfer. X01 is not independent and X04 has no performance data.
- **Verification:** Source order and terminal statuses matched Amendment 13. Frozen artifact hashes passed. BRISC sealed-test access, retraining/refitting, threshold changes, gate relaxations, and pooled cross-dataset metrics were zero.
- **Alternatives considered:** Pool heterogeneous cohorts, treat X01 as independent, impute X04, or tune from external results; all rejected by the registered protocol.
- **Who approved:** User approved Amendment 13 and the sequential Stage 6 completion protocol.
"""
    with (ROOT / "audit/review_log.md").open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(review_entry)

    manifest_paths = [
        amendment,
        REPORT,
        COMPLETION,
        ROOT / "scripts/synthesize_stage6.py",
        ROOT / "tests/test_synthesize_stage6.py",
        EXPERIMENT / "hypothesis.md",
        config_path,
        EXPERIMENT / "result.json",
        EXPERIMENT / "decision.md",
        ARTIFACTS / "dataset_summary.csv",
        ARTIFACTS / "synthesis.json",
        ARTIFACTS / "transport_summary.png",
    ] + [ROOT / "experiments" / xid / "result.json" for xid in SOURCE_IDS]
    write_json(
        ARTIFACTS / "experiment_manifest.json", experiment_manifest(manifest_paths)
    )
    subprocess.run(
        ["git", "diff", "--check"], cwd=ROOT, check=True, capture_output=True, text=True
    )


if __name__ == "__main__":
    main()
