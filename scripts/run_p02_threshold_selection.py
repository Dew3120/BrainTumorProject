"""Run the preregistered Stage 5 reliability-policy threshold sweep."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments/p02_threshold_selection"
ARTIFACTS = EXPERIMENT / "artifacts"
OOF = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
AUDIT_REPORT = ROOT / "audit/threshold_selection.md"

EXPECTED_HASHES = {
    "audit/preregistration_amendment_01.md": "62abb1d13d3ffd16d3fdf41b461ad48aafc746caf4639f18e4545467ebf8c0fb",
    "experiments/p01_cost_model/result.json": "d66042c2ce9ea848c6e98e8a06a26c9aace02a4f54b253de893b1ccafb531109",
    "experiments/p01_cost_model/artifacts/experiment_manifest.json": "ef52545bc645be0a453978046cdf094546408091d68c6ad8f15b5f99230bab50",
    "experiments/e08_final_selection/result.json": "dc2bb7f305d97830b499dc2599050e52c7119c4a3b1ae4997c240cc73edb11b9",
    "experiments/e08_final_selection/artifacts/experiment_manifest.json": "cc1be855c11b98378bf708504921e330030fc854d0ff5fc4dbdd772cfaf092b0",
    "experiments/r02_reduced_features/result.json": "fd680533caf1ac0a32c2d07de3f428cf31e0e85b12cb5f5347d30009394abbb2",
    "experiments/r02_reduced_features/artifacts/experiment_manifest.json": "1dc386a3cd342f14fcb827960c1ec89f749b2413962c768aec1c22bdf74cb819",
    "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv": "9b986435b0da0fdb340a16730209ebb895115ece694386c298830e3956b2fc6a",
    "experiments/c03_isotonic/result.json": "3af002eacbc05e14134abba67135515a58e060b0c4f255bbccd8f10b79c49cb9",
    "experiments/c03_isotonic/artifacts/experiment_manifest.json": "beec6f5526204b1dc5c44bef17048d56ad7f999d15b4dde308d9303f38a9abe4",
}

BINARY_COSTS = {"tp": 1, "fp": 1, "tn": 0, "fn": 10}
THREE_STATE_COSTS = {
    "review_acceptable": 1,
    "review_bad": 1,
    "warning_acceptable": 0.25,
    "warning_bad": 5,
    "standard_acceptable": 0,
    "standard_bad": 10,
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def write_text(path, value):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_and_load():
    for relative, expected in EXPECTED_HASHES.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise ValueError(f"INVALID/STOP hash mismatch: {relative}: {actual}")

    config = read_json(EXPERIMENT / "config.json")
    if config["cost_model"]["binary"] != {
        "true_positive": 1,
        "false_positive": 1,
        "true_negative": 0,
        "false_negative": 10,
    }:
        raise ValueError("INVALID/STOP binary costs changed")
    if config["cost_model"]["three_state"] != THREE_STATE_COSTS:
        raise ValueError("INVALID/STOP three-state costs changed")
    if config["segmentation_threshold"] != 0.5:
        raise ValueError("INVALID/STOP segmentation threshold changed")
    if config["model_retrained_or_refit"]:
        raise ValueError("INVALID/STOP config permits refitting")

    selection = read_json(ROOT / "experiments/e08_final_selection/result.json")
    if selection["selected_model"]["experiment"] != "R02":
        raise ValueError("INVALID/STOP R02 is no longer selected")
    if selection["calibration_condition"]["selected_method"] != "C03 isotonic calibration":
        raise ValueError("INVALID/STOP C03 is no longer selected")

    with OOF.open(newline="", encoding="utf-8-sig") as stream:
        source_rows = list(csv.DictReader(stream))
    if len(source_rows) != 680:
        raise ValueError(f"INVALID/STOP expected 680 cases, found {len(source_rows)}")
    paths = [row["path"] for row in source_rows]
    if len(set(paths)) != 680:
        raise ValueError("INVALID/STOP OOF paths are not unique")
    if any("/test/" in path.replace("\\", "/").lower() for path in paths):
        raise ValueError("INVALID/STOP sealed-test path detected")

    reliability = np.asarray(
        [float(row["reduced_oof_reliability"]) for row in source_rows]
    )
    dice = np.asarray([float(row["calibrated_dice"]) for row in source_rows])
    success = np.asarray(
        [bool(int(row["success_dice_at_least_0_50"])) for row in source_rows]
    )
    if not np.all(np.isfinite(reliability)) or np.any(reliability < 0) or np.any(reliability > 1):
        raise ValueError("INVALID/STOP invalid reliability value")
    if not np.array_equal(success, dice >= 0.5):
        raise ValueError("INVALID/STOP Dice target mismatch")
    if (int(success.sum()), int((~success).sum())) != (598, 82):
        raise ValueError("INVALID/STOP outcome population mismatch")
    folds = sorted({int(row["fold"]) for row in source_rows})
    if folds != [1, 2, 3, 4, 5]:
        raise ValueError(f"INVALID/STOP fold mismatch: {folds}")

    return source_rows, reliability, success, {
        "status": "PASS",
        "hashes_verified": len(EXPECTED_HASHES),
        "cases": len(source_rows),
        "unique_cases": len(set(paths)),
        "acceptable_cases": int(success.sum()),
        "bad_cases": int((~success).sum()),
        "folds": folds,
        "distinct_reliability_scores": int(np.unique(reliability).size),
        "minimum_reliability": float(reliability.min()),
        "maximum_reliability": float(reliability.max()),
        "scores_finite_and_bounded": True,
        "target_matches_dice_threshold_0_50": True,
        "binary_costs_unchanged": True,
        "three_state_costs_unchanged": True,
        "test_images_accessed": 0,
        "segmentation_threshold": 0.5,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
    }


def candidate_thresholds(reliability):
    return np.unique(np.concatenate(([0.0], reliability, [1.0])))


def binary_metrics(reliability, success, threshold):
    bad = ~success
    review = reliability < threshold
    tp = int(np.count_nonzero(review & bad))
    fp = int(np.count_nonzero(review & success))
    tn = int(np.count_nonzero((~review) & success))
    fn = int(np.count_nonzero((~review) & bad))
    total = int(tp + fp + 10 * fn)
    return {
        "threshold": float(threshold),
        "tp_bad_reviewed": tp,
        "fp_acceptable_reviewed": fp,
        "tn_acceptable_accepted": tn,
        "fn_bad_accepted": fn,
        "flagged_cases": tp + fp,
        "review_rate": (tp + fp) / len(success),
        "failure_capture_rate": tp / int(bad.sum()),
        "missed_failure_rate": fn / int(bad.sum()),
        "false_positive_rate": fp / int(success.sum()),
        "total_cost": total,
        "mean_cost": total / len(success),
    }


def binary_curve(reliability, success, thresholds):
    return [binary_metrics(reliability, success, threshold) for threshold in thresholds]


def select_binary(rows):
    return min(
        rows,
        key=lambda row: (
            row["total_cost"],
            row["fn_bad_accepted"],
            row["flagged_cases"],
            -row["threshold"],
        ),
    )


def three_state_grid(reliability, success, thresholds):
    acceptable_scores = np.sort(reliability[success])
    bad_scores = np.sort(reliability[~success])
    acceptable_below = np.searchsorted(acceptable_scores, thresholds, side="left")
    bad_below = np.searchsorted(bad_scores, thresholds, side="left")
    total_acceptable = acceptable_scores.size
    total_bad = bad_scores.size
    total_cases = reliability.size
    rows = []
    best = None
    best_key = None
    for low_index, theta_low in enumerate(thresholds):
        review_acceptable = int(acceptable_below[low_index])
        review_bad = int(bad_below[low_index])
        review_count = review_acceptable + review_bad
        for high_index in range(low_index, len(thresholds)):
            theta_high = float(thresholds[high_index])
            warning_acceptable = int(acceptable_below[high_index]) - review_acceptable
            warning_bad = int(bad_below[high_index]) - review_bad
            standard_acceptable = total_acceptable - int(acceptable_below[high_index])
            standard_bad = total_bad - int(bad_below[high_index])
            quarter_cost_units = (
                4 * (review_acceptable + review_bad)
                + warning_acceptable
                + 20 * warning_bad
                + 40 * standard_bad
            )
            total_cost = quarter_cost_units / 4
            row = {
                "theta_low": float(theta_low),
                "theta_high": theta_high,
                "review_acceptable": review_acceptable,
                "review_bad": review_bad,
                "warning_acceptable": warning_acceptable,
                "warning_bad": warning_bad,
                "standard_acceptable": standard_acceptable,
                "standard_bad": standard_bad,
                "review_cases": review_count,
                "warning_cases": warning_acceptable + warning_bad,
                "standard_cases": standard_acceptable + standard_bad,
                "review_rate": review_count / total_cases,
                "warning_rate": (warning_acceptable + warning_bad) / total_cases,
                "standard_rate": (standard_acceptable + standard_bad) / total_cases,
                "failures_not_review_required": warning_bad + standard_bad,
                "total_cost": total_cost,
                "mean_cost": total_cost / total_cases,
            }
            rows.append(row)
            key = (
                quarter_cost_units,
                standard_bad,
                warning_bad + standard_bad,
                review_count,
                -theta_high,
                -float(theta_low),
            )
            if best_key is None or key < best_key:
                best_key = key
                best = row
    return rows, best


def equal_frequency_bins(reliability, success, count=15):
    order = np.argsort(reliability, kind="mergesort")
    bins = []
    for index, members in enumerate(np.array_split(order, count), start=1):
        predicted = reliability[members]
        observed = success[members].astype(float)
        bins.append(
            {
                "bin": index,
                "count": int(members.size),
                "minimum_reliability": float(predicted.min()),
                "maximum_reliability": float(predicted.max()),
                "mean_predicted_reliability": float(predicted.mean()),
                "observed_success_frequency": float(observed.mean()),
                "absolute_gap": float(abs(predicted.mean() - observed.mean())),
            }
        )
    ece = sum(item["count"] * item["absolute_gap"] for item in bins) / len(success)
    mce = max(item["absolute_gap"] for item in bins)
    return bins, float(ece), float(mce)


def category_analysis(source_rows, reliability, success, global_threshold):
    labels = np.asarray([row["true_label"] for row in source_rows])
    output = []
    for category in ("glioma", "meningioma", "pituitary"):
        mask = labels == category
        category_scores = reliability[mask]
        category_success = success[mask]
        thresholds = candidate_thresholds(category_scores)
        optimum = select_binary(binary_curve(category_scores, category_success, thresholds))
        global_point = binary_metrics(category_scores, category_success, global_threshold)
        changed = int(
            np.count_nonzero(
                (category_scores < optimum["threshold"])
                != (category_scores < global_threshold)
            )
        )
        output.append(
            {
                "category": category,
                "cases": int(mask.sum()),
                "acceptable_cases": int(category_success.sum()),
                "bad_cases": int((~category_success).sum()),
                "category_optimal_threshold": optimum["threshold"],
                "category_optimal_total_cost": optimum["total_cost"],
                "category_optimal_mean_cost": optimum["mean_cost"],
                "category_optimal_flagged": optimum["flagged_cases"],
                "category_optimal_failures_caught": optimum["tp_bad_reviewed"],
                "category_optimal_failures_missed": optimum["fn_bad_accepted"],
                "global_threshold": float(global_threshold),
                "global_total_cost_in_category": global_point["total_cost"],
                "global_mean_cost_in_category": global_point["mean_cost"],
                "global_flagged_in_category": global_point["flagged_cases"],
                "global_failures_caught_in_category": global_point["tp_bad_reviewed"],
                "global_failures_missed_in_category": global_point["fn_bad_accepted"],
                "threshold_difference": optimum["threshold"] - global_threshold,
                "decision_assignments_changed": changed,
                "cost_reduction_vs_global": global_point["total_cost"] - optimum["total_cost"],
                "operationally_material_difference": bool(
                    changed > 0 and optimum["total_cost"] < global_point["total_cost"]
                ),
            }
        )
    return output


def render_figures(binary_rows, selected, reliability, success, bins):
    thresholds = np.asarray([row["threshold"] for row in binary_rows])
    costs = np.asarray([row["mean_cost"] for row in binary_rows])
    figure, axis = plt.subplots(figsize=(9, 5.5))
    axis.plot(thresholds, costs, color="#355c7d", linewidth=2)
    axis.scatter([selected["threshold"]], [selected["mean_cost"]], color="#c0392b", s=65, zorder=3)
    axis.axvline(selected["threshold"], color="#c0392b", linestyle="--", alpha=0.7)
    axis.set(xlabel="Reliability threshold", ylabel="Mean relative cost", title="Binary review-policy expected cost")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "expected_cost_curve.png", dpi=180)
    plt.close(figure)

    fpr = np.asarray([row["false_positive_rate"] for row in binary_rows])
    tpr = np.asarray([row["failure_capture_rate"] for row in binary_rows])
    auc = float(roc_auc_score(~success, 1.0 - reliability))
    figure, axis = plt.subplots(figsize=(6.5, 6.2))
    axis.plot(fpr, tpr, color="#355c7d", linewidth=2, label=f"OOF ROC AUC = {auc:.4f}")
    axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", linewidth=1)
    axis.scatter([selected["false_positive_rate"]], [selected["failure_capture_rate"]], color="#c0392b", s=70, label="Cost-optimal point", zorder=3)
    axis.set(xlabel="False-positive rate (acceptable masks reviewed)", ylabel="True-positive rate (bad masks reviewed)", title="Failure-detection ROC")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "roc_curve.png", dpi=180)
    plt.close(figure)

    predicted = [item["mean_predicted_reliability"] for item in bins]
    observed = [item["observed_success_frequency"] for item in bins]
    figure, axis = plt.subplots(figsize=(6.5, 6.2))
    axis.plot([0, 1], [0, 1], color="#888888", linestyle="--", label="Perfect calibration")
    axis.plot(predicted, observed, marker="o", color="#355c7d", linewidth=2, label="R02 OOF reliability")
    axis.axvline(selected["threshold"], color="#c0392b", linestyle="--", label="Selected policy threshold")
    axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted P(Dice ≥ 0.50)", ylabel="Observed success frequency", title="Reliability diagram (15 equal-frequency bins)")
    axis.legend(loc="lower right")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(ARTIFACTS / "reliability_diagram.png", dpi=180)
    plt.close(figure)
    return auc


def markdown_report(result):
    binary = result["binary_selection"]
    three = result["three_state_selection"]
    category_lines = []
    for row in result["category_analysis"]:
        category_lines.append(
            f"| {row['category']} | {row['cases']} | {row['bad_cases']} | "
            f"{row['category_optimal_threshold']:.12g} | {row['global_total_cost_in_category']} | "
            f"{row['category_optimal_total_cost']} | {row['decision_assignments_changed']} | "
            f"{row['operationally_material_difference']!s} |"
        )
    operating_lines = []
    for row in result["operating_points"]:
        operating_lines.append(
            f"| {row['name']} | {row['threshold']:.12g} | {row['flagged_cases']} | "
            f"{row['tp_bad_reviewed']} | {row['fn_bad_accepted']} | {row['fp_acceptable_reviewed']} | "
            f"{row['total_cost']} | {row['mean_cost']:.6f} |"
        )
    glioma = next(row for row in result["category_analysis"] if row["category"] == "glioma")
    material = "does" if glioma["operationally_material_difference"] else "does not"
    return f"""# Reliability threshold selection

## Scope

This internal-development analysis uses the frozen C03-calibrated E08 predictions and R02 out-of-fold reliability scores for 680 BRISC validation cases. It changes no model or segmentation threshold and does not establish clinical utility or external generalization.

## Binary cost-optimal threshold

Review is required when reliability is strictly below `{binary['threshold']:.17g}`. This threshold minimizes the fixed P01 cost `(TP + FP + 10×FN) / 680` over `0`, all distinct observed scores, and `1`.

- Flagged: **{binary['flagged_cases']} / 680** ({binary['review_rate']:.1%})
- Failures caught: **{binary['tp_bad_reviewed']} / 82** ({binary['failure_capture_rate']:.1%})
- Failures missed: **{binary['fn_bad_accepted']} / 82** ({binary['missed_failure_rate']:.1%})
- Acceptable masks reviewed: **{binary['fp_acceptable_reviewed']} / 598**
- Total cost: **{binary['total_cost']}**
- Mean cost: **{binary['mean_cost']:.6f}**

## Operating points

| Operating point | Threshold | Flagged | Failures caught | Failures missed | False alarms | Total cost | Mean cost |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(operating_lines)}

`0.50` is a conventional comparison, not a previously deployed reliability threshold.

## Three-state optimum

- `theta_low`: **{three['theta_low']:.17g}**
- `theta_high`: **{three['theta_high']:.17g}**
- Review required: **{three['review_cases']}** cases
- Warning: **{three['warning_cases']}** cases
- Standard: **{three['standard_cases']}** cases
- Bad masks in Standard: **{three['standard_bad']}**
- Bad masks outside mandatory review: **{three['failures_not_review_required']}**
- Total cost: **{three['total_cost']}**
- Mean cost: **{three['mean_cost']:.6f}**

The thresholds were obtained by exhaustively evaluating all `{result['three_state_candidate_pairs']}` valid observed-score pairs under the preregistered three-state costs.

## Descriptive category optima

| Category | Cases | Failures | Category optimum | Global cost | Category-optimal cost | Decisions changed | Material difference |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(category_lines)}

The glioma optimum {material} produce an operationally material difference under the declared definition: a different assignment plus lower subgroup cost than applying the global threshold. True-category optima are descriptive and are not deployment routing rules. The global threshold remains the operating candidate.

## Calibration and ranking context

- Failure-detection ROC AUC: **{result['roc_auc']:.6f}**
- Reliability ECE, 15 equal-frequency bins: **{result['reliability']['ece']:.6f}**
- Reliability MCE: **{result['reliability']['mce']:.6f}**

The reliability diagram is a calibration summary and does not become threshold-dependent merely because the selected operating point is marked.

## Verification and limits

All ten upstream hashes matched before results were written. The analysis used 680 unique OOF cases, 598 acceptable cases, 82 failures, and five original folds. Both cost vectors were unchanged. The neural segmentation threshold remained `0.50`; no sealed-test artifact was accessed; no classifier, segmenter, calibrator, or reliability estimator was retrained or refit.

Thresholds are optimized on internal development validation and may be optimistic. BRISC patient independence is unverified, category failure counts are small outside glioma, and real workflow costs or prevalence can change the preferred operating point.
"""


def main():
    source_rows, reliability, success, verification = verify_and_load()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    keep = ARTIFACTS / ".gitkeep"
    if keep.exists():
        keep.unlink()

    thresholds = candidate_thresholds(reliability)
    binary_rows = binary_curve(reliability, success, thresholds)
    selected = select_binary(binary_rows)
    operating_points = []
    for name, threshold in (
        ("flag_nothing", 0.0),
        ("flag_everything", 1.0),
        ("conventional_0.50_reference", 0.5),
        ("cost_optimal", selected["threshold"]),
    ):
        point = {"name": name, **binary_metrics(reliability, success, threshold)}
        operating_points.append(point)

    three_rows, three_selected = three_state_grid(reliability, success, thresholds)
    categories = category_analysis(source_rows, reliability, success, selected["threshold"])
    bins, ece, mce = equal_frequency_bins(reliability, success, count=15)

    write_csv(ARTIFACTS / "binary_threshold_curve.csv", binary_rows)
    write_csv(ARTIFACTS / "operating_points.csv", operating_points)
    write_csv(ARTIFACTS / "three_state_threshold_grid.csv", three_rows)
    write_csv(ARTIFACTS / "category_thresholds.csv", categories)
    auc = render_figures(binary_rows, selected, reliability, success, bins)

    result = {
        "schema_version": 1,
        "experiment_id": "p02_threshold_selection",
        "status": "PASS",
        "decision": "SELECT_INTERNAL_DEVELOPMENT_POLICY_THRESHOLDS",
        "scope": "Internal BRISC development policy selection; not clinical validation",
        "population": {"cases": 680, "acceptable": 598, "bad": 82},
        "costs": {"binary": BINARY_COSTS, "three_state": THREE_STATE_COSTS},
        "binary_candidate_thresholds": len(thresholds),
        "binary_selection": selected,
        "operating_points": operating_points,
        "three_state_candidate_pairs": len(three_rows),
        "three_state_selection": three_selected,
        "category_analysis": categories,
        "glioma_specific_note": next(row for row in categories if row["category"] == "glioma"),
        "roc_auc": auc,
        "reliability": {"bins": bins, "ece": ece, "mce": mce},
        "verification": {
            **verification,
            "source_hashes": EXPECTED_HASHES,
            "binary_selection_recomputed": True,
            "three_state_selection_recomputed": True,
            "script_sha256": sha256(__file__),
        },
        "limitations": [
            "Thresholds were optimized on internal development validation and await sealed evaluation.",
            "The P01 costs are engineering preferences, not clinically elicited utilities.",
            "BRISC patient independence remains unverified.",
            "True-category threshold optima are descriptive and cannot route deployment policy.",
            "Meningioma and pituitary subgroup optima are unstable because failures are sparse.",
        ],
    }

    config = read_json(EXPERIMENT / "config.json")
    config["status"] = "COMPLETE"
    write_json(EXPERIMENT / "config.json", config)
    write_json(EXPERIMENT / "result.json", result)
    write_text(
        EXPERIMENT / "decision.md",
        "# P02 decision\n\n"
        "**Decision: PASS — select the registered internal-development reliability-policy thresholds.**\n\n"
        f"The binary review threshold is `{selected['threshold']:.17g}`. "
        f"The three-state thresholds are `theta_low={three_selected['theta_low']:.17g}` and "
        f"`theta_high={three_selected['theta_high']:.17g}`. "
        "They minimize their respective preregistered costs with all registered tie breakers applied.\n\n"
        "These are internal-development operating candidates. The global threshold remains the policy candidate; "
        "true-category optima are descriptive only. No segmentation threshold changed, no test artifact was accessed, "
        "and no model was trained or refit. Stage 5.3 remains unstarted pending user approval.\n",
    )
    write_text(AUDIT_REPORT, markdown_report(result))

    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    matches = [item for item in registry["experiments"] if item["id"] == "p02_threshold_selection"]
    if len(matches) != 1:
        raise ValueError("P02 registry identity mismatch")
    matches[0]["status"] = "COMPLETE"
    write_json(registry_path, registry)

    manifest_paths = [
        "audit/threshold_selection.md",
        "experiments/p02_threshold_selection/hypothesis.md",
        "experiments/p02_threshold_selection/implementation_plan.md",
        "experiments/p02_threshold_selection/config.json",
        "experiments/p02_threshold_selection/result.json",
        "experiments/p02_threshold_selection/decision.md",
        "experiments/p02_threshold_selection/artifacts/binary_threshold_curve.csv",
        "experiments/p02_threshold_selection/artifacts/operating_points.csv",
        "experiments/p02_threshold_selection/artifacts/three_state_threshold_grid.csv",
        "experiments/p02_threshold_selection/artifacts/category_thresholds.csv",
        "experiments/p02_threshold_selection/artifacts/expected_cost_curve.png",
        "experiments/p02_threshold_selection/artifacts/roc_curve.png",
        "experiments/p02_threshold_selection/artifacts/reliability_diagram.png",
        "scripts/run_p02_threshold_selection.py",
        "tests/test_run_p02_threshold_selection.py",
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": "p02_threshold_selection",
        "status": "PASS",
        "decision": result["decision"],
        "binary_threshold": selected["threshold"],
        "theta_low": three_selected["theta_low"],
        "theta_high": three_selected["theta_high"],
        "test_images_accessed": 0,
        "segmentation_threshold_changed": False,
        "model_retrained_or_refit": False,
        "files": [
            {
                "path": relative,
                "sha256": sha256(ROOT / relative),
                "bytes": (ROOT / relative).stat().st_size,
            }
            for relative in manifest_paths
        ],
    }
    write_json(ARTIFACTS / "experiment_manifest.json", manifest)
    print(json.dumps({
        "status": "PASS",
        "binary": selected,
        "three_state": three_selected,
        "categories": categories,
        "manifest_sha256": sha256(ARTIFACTS / "experiment_manifest.json"),
        "result_sha256": sha256(EXPERIMENT / "result.json"),
        "audit_sha256": sha256(AUDIT_REPORT),
    }, indent=2))


if __name__ == "__main__":
    main()
