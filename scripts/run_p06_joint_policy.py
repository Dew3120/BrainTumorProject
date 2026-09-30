"""Run Amendment 21 P06 fixed O06+R02 joint-policy analysis."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/p06_joint_policy_risk_coverage"
ART = EXP / "artifacts"
FINAL = ROOT / "runs/final_stack"
O01 = ROOT / "experiments/o01_ood_benchmark/artifacts/benchmark_manifest.csv"
O05 = ROOT / "experiments/o05_ensemble_ood/artifacts/ensemble_scores.csv"
R02 = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
X02 = ROOT / "experiments/x02_openneuro_pituitary/artifacts/per_case_predictions.csv"
X03 = ROOT / "experiments/x03_upenn_gbm/artifacts/per_case_predictions.csv"
O07_ART = ROOT / "experiments/o07_ood_failure_correlation/artifacts"


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
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def joint_status(
    ood_score: float,
    reliability: float,
    ood_threshold: float,
    reliability_threshold: float,
) -> tuple[str, str]:
    if not np.isfinite([ood_score, reliability]).all() or not (
        0 <= ood_score <= 1 and 0 <= reliability <= 1
    ):
        return "REVIEW_REQUIRED", "INVALID_SCORE"
    if ood_score >= ood_threshold:
        return "REVIEW_REQUIRED", "OOD_REVIEW"
    if reliability < reliability_threshold:
        return "REVIEW_REQUIRED", "RELIABILITY_REVIEW"
    return "STANDARD", "BOTH_ACCEPT"


def summarize(rows: list[dict]) -> dict:
    standard = np.asarray([row["joint_status"] == "STANDARD" for row in rows])
    dice = np.asarray([row["calibrated_dice"] for row in rows], dtype=np.float64)
    failure = dice < 0.5
    missed = int(np.count_nonzero(failure & standard))
    caught = int(np.count_nonzero(failure & ~standard))
    standard_count = int(standard.sum())
    return {
        "cases": len(rows),
        "standard_count": standard_count,
        "standard_rate": standard_count / len(rows),
        "review_required_count": int((~standard).sum()),
        "review_required_rate": float((~standard).mean()),
        "standard_mean_dice": float(dice[standard].mean()) if standard_count else None,
        "standard_failure_count": missed,
        "standard_failure_rate": float(failure[standard].mean())
        if standard_count
        else None,
        "total_failures": int(failure.sum()),
        "failures_caught_for_review": caught,
        "failures_missed_in_standard": missed,
        "failure_review_sensitivity": caught / max(int(failure.sum()), 1),
        "review_reason_counts": {
            reason: sum(row["reason"] == reason for row in rows)
            for reason in (
                "BOTH_ACCEPT",
                "OOD_REVIEW",
                "RELIABILITY_REVIEW",
                "INVALID_SCORE",
            )
        },
    }


def bootstrap(rows: list[dict], repetitions=2000):
    groups = np.asarray([row["group_id"] for row in rows])
    unique = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(42)
    metrics = []
    for _ in range(repetitions):
        indices = np.concatenate(
            [members[group] for group in rng.choice(unique, len(unique), replace=True)]
        )
        item = summarize([rows[int(i)] for i in indices])
        metrics.append(item)
    fields = (
        "standard_rate",
        "review_required_rate",
        "standard_mean_dice",
        "standard_failure_rate",
        "failure_review_sensitivity",
    )
    output = {}
    for field in fields:
        values = [item[field] for item in metrics if item[field] is not None]
        output[field] = (
            [float(x) for x in np.percentile(values, [2.5, 97.5])] if values else None
        )
    return output


def brisc_rows() -> list[dict]:
    manifest = [
        row
        for row in read_csv(O01)
        if row["source"] == "brisc" and row["detector_split"] == "evaluation"
    ]
    score_by_id = {
        row["sample_id"]: float(row["selected_score"]) for row in read_csv(O05)
    }
    r02_by_path = {row["path"]: row for row in read_csv(R02)}
    if len(manifest) != 680 or len(r02_by_path) != 680:
        raise RuntimeError("P06 BRISC population mismatch")
    output = []
    for source in manifest:
        score_row = r02_by_path.get(source["path"])
        if score_row is None or source["sample_id"] not in score_by_id:
            raise RuntimeError("P06 BRISC join mismatch")
        output.append(
            {
                "cohort": "brisc_validation",
                "case_id": source["path"],
                "group_id": score_row["similarity_group"],
                "category": score_row["true_label"],
                "calibrated_dice": float(score_row["calibrated_dice"]),
                "ood_score": score_by_id[source["sample_id"]],
                "reliability": float(score_row["reduced_oof_reliability"]),
                "reliability_role": "out_of_fold",
            }
        )
    return output


def external_rows(cohort: str, source_path: Path, o07_path: Path) -> list[dict]:
    source = {row["case_id"]: row for row in read_csv(source_path)}
    ood = {row["case_id"]: row for row in read_csv(o07_path)}
    if set(source) != set(ood):
        raise RuntimeError(f"P06 {cohort} membership mismatch")
    return [
        {
            "cohort": cohort,
            "case_id": case_id,
            "group_id": row["subject_id"],
            "category": row["category"],
            "calibrated_dice": float(row["calibrated_dice"]),
            "ood_score": float(ood[case_id]["o05_score"]),
            "reliability": float(row["r02_reliability"]),
            "reliability_role": "frozen_full_population_operational",
        }
        for case_id, row in source.items()
    ]


def main() -> None:
    config = read_json(EXP / "config.json")
    if (
        config["status"] != "REGISTERED_NOT_STARTED"
        or sha256(ROOT / config["amendment_path"]) != config["amendment_sha256"]
    ):
        raise RuntimeError("P06 registration mismatch")
    sources = {
        "final_stack_manifest": FINAL / "manifest.json",
        "r02_oof_model_comparison": R02,
        "r02_deployable_manifest": ROOT
        / "experiments/x00_r02_deployable_model/artifacts/experiment_manifest.json",
        "o05_experiment_manifest": ROOT
        / "experiments/o05_ensemble_ood/artifacts/experiment_manifest.json",
        "o06_policy_assignments": ROOT
        / "experiments/o06_ood_policy/artifacts/policy_assignments.csv",
        "o06_experiment_manifest": ROOT
        / "experiments/o06_ood_policy/artifacts/experiment_manifest.json",
        "p03_oof_policy_assignments": ROOT
        / "experiments/p03_decision_rule/artifacts/oof_policy_assignments.csv",
        "p03_experiment_manifest": ROOT
        / "experiments/p03_decision_rule/artifacts/experiment_manifest.json",
        "x02_per_case_predictions": X02,
        "x03_per_case_predictions": X03,
    }
    for key, path in sources.items():
        if sha256(path) != config["source_hashes"][key]:
            raise RuntimeError(f"P06 source hash mismatch: {key}")
    o07_result = read_json(ROOT / "experiments/o07_ood_failure_correlation/result.json")
    if (
        o07_result["status"] != "COMPLETE"
        or not o07_result["verification"]["final_stack_fixtures_verified"]
    ):
        raise RuntimeError("P06 frozen external OOD scores are unverified")
    final_before = sha256(FINAL / "manifest.json")
    ood_threshold = float(config["thresholds"]["ood_review"])
    reliability_threshold = float(config["thresholds"]["reliability_standard"])
    cohorts = {
        "brisc_validation": brisc_rows(),
        "x02_openneuro": external_rows(
            "x02_openneuro", X02, O07_ART / "x02_frozen_ood_scores.csv"
        ),
        "x03_upenn_gbm": external_rows(
            "x03_upenn_gbm", X03, O07_ART / "x03_frozen_ood_scores.csv"
        ),
    }
    all_assignments = []
    results = {}
    for name, rows in cohorts.items():
        for row in rows:
            status, reason = joint_status(
                row["ood_score"],
                row["reliability"],
                ood_threshold,
                reliability_threshold,
            )
            row["joint_status"] = status
            row["reason"] = reason
            row["failure_dice_below_0_50"] = int(row["calibrated_dice"] < 0.5)
            all_assignments.append(row)
        point = summarize(rows)
        point["bootstrap_ci95"] = bootstrap(rows)
        point["per_category"] = {
            category: summarize([row for row in rows if row["category"] == category])
            for category in sorted({row["category"] for row in rows})
        }
        results[name] = point
    write_csv(ART / "joint_policy_assignments.csv", all_assignments)
    summary_rows = []
    for cohort, item in results.items():
        summary_rows.append(
            {
                "cohort": cohort,
                **{
                    key: value
                    for key, value in item.items()
                    if key
                    not in ("bootstrap_ci95", "per_category", "review_reason_counts")
                },
                **{
                    f"reason_{key.lower()}": value
                    for key, value in item["review_reason_counts"].items()
                },
            }
        )
    write_csv(ART / "cohort_summary.csv", summary_rows)
    result = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "status": "COMPLETE",
        "decision": "DESCRIPTIVE_NO_GATE",
        "exploratory_post_review": True,
        "fixed_rule": config["joint_standard_rule"],
        "cohorts": results,
        "gate": None,
        "interpretation": {
            "prevents_poor_masks_wording": "Report only observed failure-review sensitivity; this retrospective analysis does not establish safety or prospective protection.",
            "prospective_safety_established": False,
        },
        "verification": {
            "source_hashes_verified": True,
            "o07_final_stack_fixture_verification_inherited": True,
            "thresholds_exact": {
                "segmentation": config["thresholds"]["segmentation"],
                "reliability": config["thresholds"]["reliability_standard"],
                "ood_review": config["thresholds"]["ood_review"],
                "ood_refuse": None,
            },
            "final_stack_manifest_unchanged": sha256(FINAL / "manifest.json")
            == final_before,
            "model_fitting": 0,
            "threshold_tuning": 0,
            "brisc_test_images_accessed": 0,
            "neural_retraining": 0,
            "threshold_changes": 0,
        },
    }
    write_json(EXP / "result.json", result)
    config["status"] = "COMPLETE"
    write_json(EXP / "config.json", config)
    lines = [
        "# Decision — P06 joint policy risk coverage",
        "",
        "**Status:** COMPLETE — descriptive, no gate",
        "",
        f"The exact frozen rule was applied without tuning: `{config['joint_standard_rule']}`.",
        "",
    ]
    for name, item in results.items():
        mean_dice = (
            f"{item['standard_mean_dice']:.4f}"
            if item["standard_mean_dice"] is not None
            else "NOT_ESTIMABLE_NO_STANDARD_CASES"
        )
        failure_rate = (
            f"{item['standard_failure_rate']:.1%}"
            if item["standard_failure_rate"] is not None
            else "NOT_ESTIMABLE_NO_STANDARD_CASES"
        )
        lines.append(
            f"- **{name}:** STANDARD `{item['standard_count']}/{item['cases']}` ({item['standard_rate']:.1%}); accepted mean Dice `{mean_dice}`; accepted failure rate `{failure_rate}`; reviewed `{item['failures_caught_for_review']}/{item['total_failures']}` failures and missed `{item['failures_missed_in_standard']}`."
        )
    lines += [
        "",
        "These are retrospective development/external-cohort observations. They do not establish clinical safety, prospective protection, or a validated deployment policy.",
    ]
    (EXP / "decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    next(x for x in registry["experiments"] if x["id"] == EXP.name)["status"] = (
        "COMPLETE"
    )
    write_json(registry_path, registry)
    paths = [
        EXP / "config.json",
        EXP / "result.json",
        EXP / "decision.md",
        ART / "joint_policy_assignments.csv",
        ART / "cohort_summary.csv",
    ]
    write_json(
        ART / "experiment_manifest.json",
        {
            "schema_version": 1,
            "experiment_id": EXP.name,
            "files": [
                {
                    "path": path.relative_to(ROOT).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in paths
            ],
            "upstream_o07_scores": {
                "x02_sha256": sha256(O07_ART / "x02_frozen_ood_scores.csv"),
                "x03_sha256": sha256(O07_ART / "x03_frozen_ood_scores.csv"),
            },
        },
    )
    with (ROOT / "audit/review_log.md").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(
            f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Complete P06 joint policy analysis\n\n- **Decision made:** Close H13/P06 as a descriptive fixed-policy analysis with no gate.\n- **Triggering artifact:** `experiments/p06_joint_policy_risk_coverage/result.json`.\n- **Rationale:** O06 and R02 thresholds were applied unchanged and jointly to all registered populations.\n- **Verification:** Source hashes, memberships, score validity, and exact thresholds passed; zero fitting, tuning, sealed-test access, retraining, or threshold changes.\n- **Who approved:** User authorized completion of all Amendment 21 items.\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
