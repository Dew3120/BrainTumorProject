"""Run the registered P05 reliability-threshold sensitivity analysis."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/p05_threshold_sensitivity"
ARTIFACTS = EXPERIMENT / "artifacts"
OOF = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
AUDIT_REPORT = ROOT / "audit/threshold_sensitivity.md"

SELECTED_THRESHOLD = 0.9490250173235835
PERTURBATIONS = (-0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10)
COSTS = {"tp": 1, "fp": 1, "tn": 0, "fn": 10}
EXPECTED_HASHES = {
    "experiments/p01_cost_model/result.json": "d66042c2ce9ea848c6e98e8a06a26c9aace02a4f54b253de893b1ccafb531109",
    "experiments/p01_cost_model/artifacts/experiment_manifest.json": "ef52545bc645be0a453978046cdf094546408091d68c6ad8f15b5f99230bab50",
    "experiments/p02_threshold_selection/result.json": "ab86f12bb3116429bbfeb633cdef0ea0f636623757202d821e3f5b52a2fac05c",
    "experiments/p02_threshold_selection/artifacts/experiment_manifest.json": "80e1b9f3f655d43c451a194b7a9b5ddd6b6cc3de930cbd961855f880fe8196ef",
    "experiments/p03_decision_rule/result.json": "5b7573987e92b1a82461e92e37efd68ba46bdfc0bc0cc426c1128218e1944011",
    "experiments/p03_decision_rule/artifacts/experiment_manifest.json": "2a2bbc1198ae08e4b206ee91c0bdf9c0ac7d45c4d80ffff7eaf982a769038377",
    "experiments/p04_warning_copy/result.json": "a03b36297132aee490a74d03cf5cfa787495243a738ea2cef1f78ed1ee3f3fdb",
    "experiments/p04_warning_copy/artifacts/experiment_manifest.json": "655e4b2abaa9769c33295980bdcdb8e041ea2abb0fedf01198ba08160ff02d8a",
    "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv": "9b986435b0da0fdb340a16730209ebb895115ece694386c298830e3956b2fc6a",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_text(path, value):
    Path(path).write_text(value, encoding="utf-8", newline="\n")


def load_and_verify():
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"INVALID/STOP hash mismatch: {relative}: {actual}")

    config = read_json(EXPERIMENT / "config.json")
    if config["status"] not in {"REGISTERED", "COMPLETE"}:
        raise ValueError("INVALID/STOP invalid P05 lifecycle state")
    if config["selected_threshold"] != SELECTED_THRESHOLD:
        raise ValueError("INVALID/STOP selected threshold changed")
    if tuple(config["nominal_perturbations"]) != PERTURBATIONS:
        raise ValueError("INVALID/STOP perturbations changed")
    expected_costs = {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 10,
    }
    if config["costs"] != expected_costs:
        raise ValueError("INVALID/STOP cost vector changed")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("INVALID/STOP segmentation threshold changed")
    if any(
        config[key]
        for key in (
            "model_retrained_or_refit",
            "calibrator_retrained_or_refit",
            "reliability_model_retrained_or_refit",
        )
    ):
        raise ValueError("INVALID/STOP config permits training or refitting")

    p01 = read_json(ROOT / "experiments/p01_cost_model/result.json")
    p02 = read_json(ROOT / "experiments/p02_threshold_selection/result.json")
    p03 = read_json(ROOT / "experiments/p03_decision_rule/result.json")
    p04 = read_json(ROOT / "experiments/p04_warning_copy/result.json")
    if p01["cost_matrix"] != {
        "true_positive_reviewed_bad": 1,
        "false_positive_reviewed_acceptable": 1,
        "true_negative_accepted_acceptable": 0,
        "false_negative_accepted_bad": 10,
        "unit": "relative engineering cost",
        "mean_cost_formula": "(TP + FP + 10*FN) / N",
    }:
        raise ValueError("INVALID/STOP P01 cost vector changed")
    if p02["binary_selection"]["threshold"] != SELECTED_THRESHOLD:
        raise ValueError("INVALID/STOP P02 selected threshold changed")
    if p03["policy"]["threshold"] != SELECTED_THRESHOLD:
        raise ValueError("INVALID/STOP P03 policy threshold changed")
    if p04["catalogue"]["threshold"] != SELECTED_THRESHOLD:
        raise ValueError("INVALID/STOP P04 message threshold changed")

    with OOF.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 680:
        raise ValueError(f"INVALID/STOP expected 680 cases, found {len(rows)}")
    paths = [row["path"] for row in rows]
    if len(set(paths)) != 680:
        raise ValueError("INVALID/STOP OOF paths are not unique")
    if any("/test/" in path.replace("\\", "/").lower() for path in paths):
        raise ValueError("INVALID/STOP sealed-test path detected")

    reliability = np.asarray([float(row["reduced_oof_reliability"]) for row in rows])
    dice = np.asarray([float(row["calibrated_dice"]) for row in rows])
    acceptable = np.asarray(
        [bool(int(row["success_dice_at_least_0_50"])) for row in rows]
    )
    if not np.all(np.isfinite(reliability)) or np.any((reliability < 0) | (reliability > 1)):
        raise ValueError("INVALID/STOP invalid reliability score")
    if not np.array_equal(acceptable, dice >= 0.5):
        raise ValueError("INVALID/STOP target does not match Dice >= 0.50")
    if (int(acceptable.sum()), int((~acceptable).sum())) != (598, 82):
        raise ValueError("INVALID/STOP population mismatch")
    folds = sorted({int(row["fold"]) for row in rows})
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"INVALID/STOP fold mismatch: {folds}")

    return reliability, acceptable, {
        "status": "PASS",
        "hashes_verified": len(EXPECTED_HASHES),
        "cases": 680,
        "unique_cases": 680,
        "acceptable_cases": 598,
        "bad_cases": 82,
        "folds": folds,
        "scores_finite_and_bounded": True,
        "target_matches_dice_threshold_0_50": True,
        "cost_vector_unchanged": True,
        "selected_threshold_unchanged": True,
        "test_images_accessed": 0,
        "segmentation_threshold": 0.5,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
        "calibrator_retrained_or_refit": False,
        "reliability_model_retrained_or_refit": False,
    }


def evaluate(reliability, acceptable, threshold):
    threshold = float(threshold)
    bad = ~acceptable
    flagged = reliability < threshold
    tp = int(np.count_nonzero(flagged & bad))
    fp = int(np.count_nonzero(flagged & acceptable))
    tn = int(np.count_nonzero((~flagged) & acceptable))
    fn = int(np.count_nonzero((~flagged) & bad))
    total_cost = tp + fp + 10 * fn
    return {
        "threshold": threshold,
        "cases_flagged": tp + fp,
        "cases_accepted": tn + fn,
        "failures_caught": tp,
        "failures_missed": fn,
        "acceptable_cases_flagged": fp,
        "acceptable_cases_accepted": tn,
        "total_cost": total_cost,
        "mean_cost": total_cost / len(reliability),
    }


def requested_points(reliability, acceptable):
    selected = evaluate(reliability, acceptable, SELECTED_THRESHOLD)
    points = []
    for nominal in PERTURBATIONS:
        threshold = float(np.clip(SELECTED_THRESHOLD + nominal, 0.0, 1.0))
        row = evaluate(reliability, acceptable, threshold)
        applied = threshold - SELECTED_THRESHOLD
        row.update(
            {
                "label": "selected" if nominal == 0 else f"{nominal:+.2f}",
                "nominal_delta": nominal,
                "applied_delta": applied,
                "was_clipped": not np.isclose(applied, nominal, atol=1e-15),
                "cost_change": row["total_cost"] - selected["total_cost"],
                "cost_change_fraction": (
                    row["total_cost"] - selected["total_cost"]
                )
                / selected["total_cost"],
                "cost_change_per_0_01": (
                    None
                    if np.isclose(applied, 0.0, atol=1e-15)
                    else (row["total_cost"] - selected["total_cost"])
                    / (abs(applied) / 0.01)
                ),
            }
        )
        points.append(row)
    return points


def historical_points(reliability, acceptable):
    selected_cost = evaluate(reliability, acceptable, SELECTED_THRESHOLD)["total_cost"]
    output = []
    for label, threshold in (
        ("flag_nothing", 0.0),
        ("conventional_0.50", 0.5),
        ("selected", SELECTED_THRESHOLD),
        ("flag_everything", 1.0),
    ):
        row = evaluate(reliability, acceptable, threshold)
        row.update(
            {
                "label": label,
                "cost_delta_vs_selected": row["total_cost"] - selected_cost,
                "cost_delta_fraction_vs_selected": (
                    row["total_cost"] - selected_cost
                )
                / selected_cost,
            }
        )
        output.append(row)
    return output


def build_curve(reliability, acceptable, requested, historical):
    low = max(0.0, SELECTED_THRESHOLD - 0.10)
    high = min(1.0, SELECTED_THRESHOLD + 0.10)
    local_scores = np.unique(reliability[(reliability >= low) & (reliability <= high)])
    thresholds = {float(value) for value in local_scores}
    thresholds.update(float(row["threshold"]) for row in requested)
    thresholds.update(float(row["threshold"]) for row in historical)
    request_by_threshold = {}
    for row in requested:
        request_by_threshold.setdefault(row["threshold"], []).append(row["label"])
    historical_by_threshold = {}
    for row in historical:
        historical_by_threshold.setdefault(row["threshold"], []).append(row["label"])
    selected_cost = evaluate(reliability, acceptable, SELECTED_THRESHOLD)["total_cost"]
    output = []
    for threshold in sorted(thresholds):
        row = evaluate(reliability, acceptable, threshold)
        row.update(
            {
                "in_local_range": low <= threshold <= high,
                "is_observed_score": bool(np.any(reliability == threshold)),
                "requested_labels": "|".join(request_by_threshold.get(threshold, [])),
                "historical_labels": "|".join(historical_by_threshold.get(threshold, [])),
                "cost_change": row["total_cost"] - selected_cost,
                "cost_change_fraction": (row["total_cost"] - selected_cost)
                / selected_cost,
            }
        )
        output.append(row)
    return output


def independent_verify(curve_rows, requested, reliability, acceptable):
    def independent_counts(threshold):
        counts = {
            "cases_flagged": 0,
            "cases_accepted": 0,
            "failures_caught": 0,
            "failures_missed": 0,
            "acceptable_cases_flagged": 0,
            "acceptable_cases_accepted": 0,
        }
        for score, is_acceptable in zip(reliability.tolist(), acceptable.tolist(), strict=True):
            is_flagged = score < threshold
            counts["cases_flagged" if is_flagged else "cases_accepted"] += 1
            if is_acceptable:
                key = "acceptable_cases_flagged" if is_flagged else "acceptable_cases_accepted"
            else:
                key = "failures_caught" if is_flagged else "failures_missed"
            counts[key] += 1
        counts["total_cost"] = (
            counts["failures_caught"]
            + counts["acceptable_cases_flagged"]
            + 10 * counts["failures_missed"]
        )
        counts["mean_cost"] = counts["total_cost"] / len(reliability)
        return counts

    for row in curve_rows:
        expected = independent_counts(row["threshold"])
        for key, value in expected.items():
            if not np.isclose(row[key], value, atol=1e-15):
                raise ValueError(f"INVALID/STOP independent mismatch at {row['threshold']}: {key}")
    selected = next(row for row in requested if row["label"] == "selected")
    expected_p02 = {
        "cases_flagged": 195,
        "cases_accepted": 485,
        "failures_caught": 81,
        "failures_missed": 1,
        "acceptable_cases_flagged": 114,
        "acceptable_cases_accepted": 484,
        "total_cost": 205,
    }
    if any(selected[key] != value for key, value in expected_p02.items()):
        raise ValueError("INVALID/STOP selected threshold does not reproduce P02")
    return {
        "all_curve_rows_independently_recomputed": True,
        "selected_row_reproduces_p02": True,
        "curve_rows_verified": len(curve_rows),
    }


def save_curve_csv(rows):
    fieldnames = [
        "threshold",
        "in_local_range",
        "is_observed_score",
        "requested_labels",
        "historical_labels",
        "cases_flagged",
        "cases_accepted",
        "failures_caught",
        "failures_missed",
        "acceptable_cases_flagged",
        "acceptable_cases_accepted",
        "total_cost",
        "mean_cost",
        "cost_change",
        "cost_change_fraction",
    ]
    with (ARTIFACTS / "sensitivity_curve.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_figure(curve, requested, historical):
    local = [row for row in curve if row["in_local_range"]]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    axes[0].step(
        [row["threshold"] for row in local],
        [row["total_cost"] for row in local],
        where="post",
        color="#1f4e79",
        linewidth=2,
        label="Exact OOF cost",
    )
    axes[0].scatter(
        [row["threshold"] for row in requested],
        [row["total_cost"] for row in requested],
        color="#c43c39",
        zorder=3,
        label="Requested perturbations",
    )
    axes[0].axvline(SELECTED_THRESHOLD, color="#238b45", linestyle="--", label="Selected")
    axes[0].set(title="Local threshold sensitivity", xlabel="Reliability threshold", ylabel="Total cost")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = [row["label"].replace("_", "\n") for row in historical]
    costs = [row["total_cost"] for row in historical]
    colors = ["#777777", "#e69500", "#238b45", "#777777"]
    bars = axes[1].bar(labels, costs, color=colors)
    axes[1].bar_label(bars, padding=3)
    axes[1].set(title="Historical operating-point comparison", ylabel="Total cost")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("P05 reliability-threshold sensitivity (680 OOF cases)", fontsize=14)
    fig.savefig(ARTIFACTS / "sensitivity_curve.png", dpi=180)
    plt.close(fig)


def result_document(requested, historical, verification):
    by_label = {row["label"]: row for row in requested}
    selected = by_label["selected"]
    minus_002, plus_002 = by_label["-0.02"], by_label["+0.02"]
    minus_005, plus_005 = by_label["-0.05"], by_label["+0.05"]
    minus_010, plus_010 = by_label["-0.10"], by_label["+0.10"]
    local_worst = max(minus_002["total_cost"], plus_002["total_cost"])
    worst_005 = max(minus_005["total_cost"], plus_005["total_cost"])
    worst_010 = max(minus_010["total_cost"], plus_010["total_cost"])
    locally_flat_limit = selected["total_cost"] * 1.10
    robust_limit = selected["total_cost"] * 1.25
    locally_flat = local_worst <= locally_flat_limit
    robust = locally_flat and worst_005 <= robust_limit
    classification = "ROBUST" if robust else "FRAGILE"
    conclusion = (
        "Small perturbations in the threshold produce small changes in cost. "
        "The selected threshold is stable."
        if robust
        else "Small perturbations produce large cost changes. The threshold requires careful calibration."
    )
    return {
        "schema_version": 1,
        "experiment_id": "p05_threshold_sensitivity",
        "status": "PASS",
        "decision": f"THRESHOLD_SENSITIVITY_{classification}",
        "scope": "Internal BRISC development sensitivity; not clinical validation",
        "selected_threshold": SELECTED_THRESHOLD,
        "selected_operating_point": selected,
        "requested_perturbations": requested,
        "sensitivity": {
            "locally_flat": locally_flat,
            "classification": classification,
            "local_plus_or_minus_0_02_worst_cost": local_worst,
            "local_flat_cost_limit": locally_flat_limit,
            "plus_or_minus_0_05_worst_cost": worst_005,
            "robust_cost_limit": robust_limit,
            "plus_or_minus_0_10_worst_cost": worst_010,
            "positive_0_10_endpoint_clipped_to_one": plus_010["was_clipped"],
            "conclusion": conclusion,
        },
        "historical_comparison": historical,
        "verification": verification,
        "limitations": [
            "Sensitivity is measured on the same internal OOF development population used for policy selection.",
            "The P01 costs are engineering preferences, not clinically elicited utilities.",
            "The robustness labels are descriptive and do not change the selected threshold.",
            "BRISC patient independence remains unverified.",
        ],
    }


def markdown_report(result):
    selected = result["selected_operating_point"]
    sensitivity = result["sensitivity"]
    requested_lines = []
    for row in result["requested_perturbations"]:
        requested_lines.append(
            f"| {row['label']} | {row['threshold']:.15f} | {row['cases_flagged']} | "
            f"{row['failures_caught']} | {row['failures_missed']} | "
            f"{row['acceptable_cases_flagged']} | {row['total_cost']} | "
            f"{row['mean_cost']:.6f} | {row['cost_change']:+d} |"
        )
    historical_lines = []
    for row in result["historical_comparison"]:
        historical_lines.append(
            f"| {row['label']} | {row['threshold']:.15f} | {row['cases_flagged']} | "
            f"{row['failures_caught']} | {row['failures_missed']} | "
            f"{row['total_cost']} | {row['cost_delta_vs_selected']:+d} |"
        )
    return f"""# Reliability-threshold sensitivity

## Decision

**{sensitivity['classification']}**; locally flat: **{'YES' if sensitivity['locally_flat'] else 'NO'}**.

{sensitivity['conclusion']}

The selected threshold remains `{SELECTED_THRESHOLD}`. This descriptive analysis does not alter the policy.

## Requested perturbations

| Shift | Applied threshold | Flagged | Failures caught | Failures missed | Acceptable flagged | Total cost | Mean cost | Cost delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(requested_lines)}

At the selected threshold, {selected['cases_flagged']} cases are flagged, {selected['failures_caught']} failures are caught, {selected['failures_missed']} failure is missed, and total cost is {selected['total_cost']}.

The worst endpoint cost at `±0.05` is **{sensitivity['plus_or_minus_0_05_worst_cost']}**. The worst endpoint cost at `±0.10` is **{sensitivity['plus_or_minus_0_10_worst_cost']}**. The positive `+0.10` request is clipped to threshold `1.0` because the registered threshold domain is `[0,1]`.

Cost change per `0.01` is reported in `result.json` for each nonzero requested perturbation using the registered directional formula.

## Historical comparison

| Operating point | Threshold | Flagged | Failures caught | Failures missed | Total cost | Cost delta vs selected |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(historical_lines)}

## Integrity

- All {result['verification']['curve_rows_verified']} CSV rows were independently recomputed.
- The selected row reproduced P02 exactly.
- Population: 680 OOF cases, 598 acceptable and 82 failures.
- Cost vector: TP=1, FP=1, TN=0, FN=10.
- Segmentation threshold remained `0.50`.
- Sealed test-set access: 0.
- Model, calibrator, and reliability-model retraining/refitting: 0.

This is an internal development-policy sensitivity analysis. It does not establish clinical validity or cross-dataset generalization.
"""


def build_manifest(result):
    files = [
        "audit/threshold_sensitivity.md",
        "experiments/p05_threshold_sensitivity/hypothesis.md",
        "experiments/p05_threshold_sensitivity/config.json",
        "experiments/p05_threshold_sensitivity/result.json",
        "experiments/p05_threshold_sensitivity/decision.md",
        "experiments/p05_threshold_sensitivity/artifacts/sensitivity_curve.csv",
        "experiments/p05_threshold_sensitivity/artifacts/sensitivity_curve.png",
        "scripts/run_p05_threshold_sensitivity.py",
        "tests/test_run_p05_threshold_sensitivity.py",
    ]
    records = []
    for relative in files:
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"INVALID/STOP manifest file missing: {relative}")
        records.append({"path": relative, "sha256": sha256(path), "bytes": path.stat().st_size})
    return {
        "schema_version": 1,
        "experiment_id": "p05_threshold_sensitivity",
        "status": "PASS",
        "decision": result["decision"],
        "selected_threshold": SELECTED_THRESHOLD,
        "classification": result["sensitivity"]["classification"],
        "test_images_accessed": 0,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
        "files": records,
    }


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    reliability, acceptable, verification = load_and_verify()
    requested = requested_points(reliability, acceptable)
    historical = historical_points(reliability, acceptable)
    curve = build_curve(reliability, acceptable, requested, historical)
    verification.update(independent_verify(curve, requested, reliability, acceptable))

    result = result_document(requested, historical, verification)
    save_curve_csv(curve)
    save_figure(curve, requested, historical)
    write_json(EXPERIMENT / "result.json", result)
    decision = (
        "# P05 decision\n\n"
        f"**Decision: PASS — {result['sensitivity']['classification']}.**\n\n"
        f"{result['sensitivity']['conclusion']} The selected reliability threshold remains "
        f"`{SELECTED_THRESHOLD}`. The analysis is descriptive and does not change the policy.\n\n"
        "No sealed-test access, segmentation-threshold change, training, refitting, or gate relaxation occurred.\n"
    )
    write_text(EXPERIMENT / "decision.md", decision)
    write_text(AUDIT_REPORT, markdown_report(result))

    config = read_json(EXPERIMENT / "config.json")
    config["status"] = "COMPLETE"
    write_json(EXPERIMENT / "config.json", config)

    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    entry = next(item for item in registry["experiments"] if item["id"] == "p05_threshold_sensitivity")
    if entry["status"] not in {"REGISTERED", "COMPLETE"}:
        raise ValueError("INVALID/STOP registry has an invalid lifecycle state")
    entry["status"] = "COMPLETE"
    registry_path.write_bytes((json.dumps(registry, indent=2) + "\n").encode("utf-8"))

    manifest = build_manifest(result)
    write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
