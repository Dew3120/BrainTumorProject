"""Run Amendment 21 R12 BRISC-fit one-feature external baselines."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import evaluate_x01_cheng as x01
import evaluate_x02_openneuro as x02
import evaluate_x03_upenn_gbm as x03
import numpy as np
import stage4_reliability_common as reliability_common
import stage6_external_common as external
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/r12_r02_external_baselines"
ART = EXP / "artifacts"
FINAL = ROOT / "runs/final_stack"
R01_FEATURES = ROOT / "experiments/r01_ablation/artifacts/oof_scores_and_features.csv"
R02_OOF = ROOT / "experiments/r02_reduced_features/artifacts/oof_model_comparison.csv"
X02_ROWS = (
    ROOT / "experiments/x02_openneuro_pituitary/artifacts/per_case_predictions.csv"
)
X03_ROWS = ROOT / "experiments/x03_upenn_gbm/artifacts/per_case_predictions.csv"
RELIABILITY_THRESHOLD = 0.9490250173235835


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


def make_comparator():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, penalty="l2", solver="lbfgs", max_iter=10000, random_state=42
        ),
    )


def fit_comparators_before_external_join(config: dict) -> dict:
    r01 = read_csv(R01_FEATURES)
    r02 = read_csv(R02_OOF)
    if len(r01) != 680 or len(r02) != 680:
        raise RuntimeError("R12 BRISC population mismatch")
    cache_config = {
        "c03_mappings_path": "experiments/c03_isotonic/artifacts/isotonic_mappings.json",
        "c03_mappings_sha256": sha256(
            ROOT / "experiments/c03_isotonic/artifacts/isotonic_mappings.json"
        ),
        "c03_result_sha256": sha256(ROOT / "experiments/c03_isotonic/result.json"),
        "calibration_cache_manifest_sha256": sha256(
            Path(
                (ROOT / "runs/segmentation_v2_ensemble/latest.txt").read_text().strip()
            )
            / "manifest.json"
        ),
        "fold_assignment_sha256": config["development_population"][
            "fold_assignment_sha256"
        ],
        "expected_c03_hard_mask_changes": {"pixels": 22891, "images": 679},
    }
    data = reliability_common.load_calibrated_e08(cache_config)
    by_path = {row["path"]: index for index, row in enumerate(data["rows"])}
    max_probability = data["probability"].max(axis=(1, 2))
    features = np.empty((680, 2), dtype=np.float64)
    success = np.empty(680, dtype=bool)
    dice = np.empty(680)
    groups = []
    for i, (feature_row, score_row) in enumerate(zip(r01, r02, strict=True)):
        if (
            feature_row["path"] != score_row["path"]
            or feature_row["path"] not in by_path
        ):
            raise RuntimeError("R12 BRISC row-order mismatch")
        cache_i = by_path[feature_row["path"]]
        if int(feature_row["fold"]) - 1 != int(data["assignment"][cache_i]):
            raise RuntimeError("R12 fold mismatch")
        features[i] = [
            float(feature_row["predicted_area_fraction"]),
            float(max_probability[cache_i]),
        ]
        success[i] = bool(int(feature_row["success_dice_at_least_0_50"]))
        dice[i] = float(feature_row["calibrated_dice"])
        groups.append(feature_row["similarity_group"])
        if abs(dice[i] - float(score_row["calibrated_dice"])) > 1e-12:
            raise RuntimeError("R12 BRISC Dice mismatch")
    assignment = np.asarray([int(row["fold"]) - 1 for row in r01])
    oof = np.empty((680, 2), dtype=np.float64)
    for fold in range(5):
        train = assignment != fold
        test = assignment == fold
        for column in range(2):
            model = make_comparator()
            model.fit(features[train, column, None], success[train])
            oof[test, column] = model.predict_proba(features[test, column, None])[:, 1]
    names = config["comparators"]
    frozen = {}
    for column, name in enumerate(names):
        model = make_comparator()
        model.fit(features[:, column, None], success)
        path = ART / f"{name}_brisc_frozen.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        frozen[name] = {"model": model, "path": path, "sha256": sha256(path)}
    write_csv(
        ART / "brisc_oof_comparator_scores.csv",
        [
            {
                "path": row["path"],
                "similarity_group": groups[i],
                "fold": int(assignment[i]) + 1,
                "calibrated_dice": dice[i],
                "success": int(success[i]),
                "r02_reliability": float(r02[i]["reduced_oof_reliability"]),
                "predicted_area_fraction": features[i, 0],
                "maximum_calibrated_foreground_probability": features[i, 1],
                "area_comparator_reliability": oof[i, 0],
                "max_probability_comparator_reliability": oof[i, 1],
            }
            for i, row in enumerate(r01)
        ],
    )
    return {
        "features": features,
        "success": success,
        "dice": dice,
        "groups": np.asarray(groups),
        "r02": np.asarray([float(row["reduced_oof_reliability"]) for row in r02]),
        "oof": oof,
        "frozen": frozen,
    }


def external_primary_rows(cohort: str):
    if cohort == "x02":
        all_rows = x02.build_evaluation_rows(
            x02.read_csv(x02.SUBJECT_MANIFEST), x02.read_csv(x02.TUMOR_SLICE_MANIFEST)
        )
        return [
            row for row in all_rows if row["reference_has_tumor"]
        ], x02.OpenNeuroDataset
    all_rows = x03.build_evaluation_rows(
        x03.read_csv(x03.SUBJECT_MANIFEST), x03.read_csv(x03.TUMOR_SLICE_MANIFEST)
    )
    return [row for row in all_rows if row["reference_has_tumor"]], x03.UpennDataset


def infer_external_feature(cohort: str, classifier, models, device) -> list[dict]:
    rows, dataset_type = external_primary_rows(cohort)
    dataset = dataset_type(rows)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    output = [None] * len(rows)
    with torch.inference_mode():
        for batch_no, (seg_images, cls_images, targets, indices) in enumerate(
            loader, start=1
        ):
            original_batch = len(seg_images)
            if original_batch < 16:
                # Preserve the Stage 6 fp16 batch geometry at the hard-mask
                # boundary even though this analysis selects tumor slices only.
                padding = 16 - original_batch
                seg_images = torch.cat(
                    (seg_images, seg_images[:1].repeat(padding, 1, 1, 1))
                )
                cls_images = torch.cat(
                    (cls_images, cls_images[:1].repeat(padding, 1, 1, 1))
                )
            seg_images = seg_images.to(device, non_blocking=True)
            cls_images = cls_images.to(device, non_blocking=True)
            context = (
                torch.autocast("cuda", dtype=torch.float16)
                if device.type == "cuda"
                else nullcontext()
            )
            with context:
                class_logits = classifier(cls_images)
                e05 = models["e05"](seg_images)
                e06 = models["e06"](seg_images)
                specialist = models["e07_specialist"](seg_images)
                fallback = models["v1_fallback"](seg_images)
            class_probability = (
                class_logits.float().softmax(1).cpu().numpy()[:original_batch]
            )
            predicted = class_probability.argmax(1)
            route = predicted == 0
            routed = fallback.float().sigmoid()
            if np.any(route):
                route_index = torch.from_numpy(np.flatnonzero(route)).to(device)
                routed[route_index] = specialist.float().sigmoid()[route_index]
            raw = (
                ((e05.float().sigmoid() + e06.float().sigmoid() + routed) / 3.0)[
                    :original_batch, 0
                ]
                .cpu()
                .numpy()
            )
            # Match the registered Stage 6 operational path, which persisted
            # C03 output as float32 before feature extraction.
            calibrated = external.apply_operational_c03(raw).astype(np.float32)
            hard = calibrated >= 0.5
            refs = targets[:, 0].numpy() >= 0.5
            intersection = (hard & refs).sum(axis=(1, 2))
            denominator = hard.sum(axis=(1, 2)) + refs.sum(axis=(1, 2))
            dice = 2 * intersection / np.maximum(denominator, 1)
            for pos, index in enumerate(indices.tolist()):
                if not refs[pos].any():
                    dice[pos] = 0.0
                output[index] = {
                    "case_id": rows[index]["case_id"],
                    "subject_id": rows[index]["subject_id"],
                    "category": rows[index]["label"],
                    "calibrated_dice": float(dice[pos]),
                    "predicted_pixels": int(hard[pos].sum()),
                    "predicted_area_fraction": float(hard[pos].mean()),
                    "maximum_calibrated_foreground_probability": float(
                        calibrated[pos].max()
                    ),
                    "r02_reliability_recomputed": float(
                        external.operational_reliability(calibrated[pos])["score"]
                    ),
                }
            if batch_no % 100 == 0 or batch_no == len(loader):
                print(
                    f"R12 {cohort} frozen inference {batch_no}/{len(loader)}",
                    flush=True,
                )
    return output


def bootstrap_indices(groups: np.ndarray, repetitions=2000, seed=42):
    unique = np.asarray(sorted(set(groups)))
    members = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [members[g] for g in rng.choice(unique, len(unique), replace=True)]
        )
        for _ in range(repetitions)
    ]


def metric_summary(success, reliability, groups, samples=None):
    failure = ~success
    risk = 1 - reliability
    if samples is None:
        samples = bootstrap_indices(groups)
    aucs, aps = [], []
    for sample in samples:
        if len(np.unique(failure[sample])) < 2:
            continue
        aucs.append(roc_auc_score(failure[sample], risk[sample]))
        aps.append(average_precision_score(failure[sample], risk[sample]))
    return {
        "failure_prevalence": float(failure.mean()),
        "failure_detection_auroc": float(roc_auc_score(failure, risk)),
        "failure_detection_auroc_ci95": [
            float(x) for x in np.percentile(aucs, [2.5, 97.5])
        ],
        "failure_average_precision": float(average_precision_score(failure, risk)),
        "failure_average_precision_ci95": [
            float(x) for x in np.percentile(aps, [2.5, 97.5])
        ],
    }


def coverage_rows(cohort, method, reliability, dice, case_ids, fixed_count):
    order = np.lexsort(
        (np.asarray([value.casefold() for value in case_ids]), -reliability)
    )
    rows = []
    for coverage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        count = max(1, math.ceil(coverage * len(order)))
        selected = order[:count]
        rows.append(
            {
                "cohort": cohort,
                "method": method,
                "selection": f"coverage_{coverage:.1f}",
                "accepted_count": count,
                "accepted_coverage": count / len(order),
                "accepted_mean_dice": float(dice[selected].mean()),
                "accepted_failure_rate": float((dice[selected] < 0.5).mean()),
            }
        )
    selected = order[:fixed_count]
    rows.append(
        {
            "cohort": cohort,
            "method": method,
            "selection": "r02_fixed_policy_accepted_count",
            "accepted_count": fixed_count,
            "accepted_coverage": fixed_count / len(order),
            "accepted_mean_dice": float(dice[selected].mean()) if fixed_count else None,
            "accepted_failure_rate": float((dice[selected] < 0.5).mean())
            if fixed_count
            else None,
        }
    )
    return rows


def main() -> None:
    config = read_json(EXP / "config.json")
    if (
        config["status"] != "REGISTERED_NOT_STARTED"
        or sha256(ROOT / config["amendment_path"]) != config["amendment_sha256"]
    ):
        raise RuntimeError("R12 registration mismatch")
    checks = {
        "final_stack_manifest": FINAL / "manifest.json",
        "r01_oof_features": R01_FEATURES,
        "r02_oof_model_comparison": R02_OOF,
        "r02_deployable_manifest": ROOT
        / "experiments/x00_r02_deployable_model/artifacts/experiment_manifest.json",
        "x02_per_case_predictions": X02_ROWS,
        "x02_experiment_manifest": ROOT
        / "experiments/x02_openneuro_pituitary/artifacts/experiment_manifest.json",
        "x03_per_case_predictions": X03_ROWS,
        "x03_experiment_manifest": ROOT
        / "experiments/x03_upenn_gbm/artifacts/experiment_manifest.json",
    }
    for key, path in checks.items():
        if sha256(path) != config["source_hashes"][key]:
            raise RuntimeError(f"R12 source hash mismatch: {key}")
    frozen_manifest_before = sha256(FINAL / "manifest.json")
    # This call performs and freezes every authorized comparator fit before any
    # external outcome table is loaded or joined.
    development = fit_comparators_before_external_join(config)
    frozen_at = datetime.now().astimezone().isoformat(timespec="seconds")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, models = x01.load_models(device)
    external_features = {}
    for cohort in ("x02", "x03"):
        cache_path = ART / f"{cohort}_frozen_feature_cache.csv"
        if cache_path.exists():
            external_features[cohort] = read_csv(cache_path)
            for row in external_features[cohort]:
                for field in (
                    "calibrated_dice",
                    "predicted_area_fraction",
                    "maximum_calibrated_foreground_probability",
                    "r02_reliability_recomputed",
                ):
                    row[field] = float(row[field])
                row["predicted_pixels"] = int(row["predicted_pixels"])
        else:
            external_features[cohort] = infer_external_feature(
                cohort, classifier, models, device
            )
            write_csv(cache_path, external_features[cohort])
    source_tables = {"x02": read_csv(X02_ROWS), "x03": read_csv(X03_ROWS)}
    expected = {"x02": 425, "x03": 9464}
    cohort_results = {}
    all_score_rows = []
    all_coverage = []
    all_bootstrap = []
    for cohort in ("x02", "x03"):
        inferred = external_features[cohort]
        committed = source_tables[cohort]
        if len(inferred) != expected[cohort] or len(committed) != expected[cohort]:
            raise RuntimeError(f"R12 {cohort} population mismatch")
        by_id = {row["case_id"]: row for row in committed}
        dice = []
        groups = []
        r02 = []
        matrix = []
        ids = []
        for row in inferred:
            old = by_id.get(row["case_id"])
            if old is None:
                raise RuntimeError(f"R12 {cohort} membership mismatch")
            if (
                abs(row["calibrated_dice"] - float(old["calibrated_dice"])) > 1e-12
                or row["predicted_pixels"] != int(old["predicted_pixels"])
                or abs(
                    row["r02_reliability_recomputed"] - float(old["r02_reliability"])
                )
                > 1e-12
            ):
                raise RuntimeError(
                    f"R12 {cohort} inference reproduction mismatch: {row['case_id']}"
                )
            ids.append(row["case_id"])
            groups.append(row["subject_id"])
            dice.append(row["calibrated_dice"])
            r02.append(float(old["r02_reliability"]))
            matrix.append(
                [
                    row["predicted_area_fraction"],
                    row["maximum_calibrated_foreground_probability"],
                ]
            )
        dice = np.asarray(dice)
        success = dice >= 0.5
        r02 = np.asarray(r02)
        matrix = np.asarray(matrix)
        groups = np.asarray(groups)
        comparator = np.column_stack(
            [
                development["frozen"][name]["model"].predict_proba(
                    matrix[:, col, None]
                )[:, 1]
                for col, name in enumerate(config["comparators"])
            ]
        )
        samples = bootstrap_indices(groups)
        method_scores = {
            "r02": r02,
            "predicted_area_fraction": comparator[:, 0],
            "maximum_calibrated_foreground_probability": comparator[:, 1],
        }
        summaries = {
            name: metric_summary(success, score, groups, samples)
            for name, score in method_scores.items()
        }
        differences = {}
        for name in config["comparators"]:
            values = []
            for sample in samples:
                failure = ~success[sample]
                if len(np.unique(failure)) < 2:
                    continue
                values.append(
                    roc_auc_score(failure, 1 - r02[sample])
                    - roc_auc_score(failure, 1 - method_scores[name][sample])
                )
            point = (
                summaries["r02"]["failure_detection_auroc"]
                - summaries[name]["failure_detection_auroc"]
            )
            differences[name] = {
                "r02_minus_comparator_auroc": point,
                "paired_ci95": [float(x) for x in np.percentile(values, [2.5, 97.5])],
                "gate_passed": point >= 0.03,
            }
            for j, value in enumerate(values):
                all_bootstrap.append(
                    {
                        "cohort": cohort,
                        "comparator": name,
                        "resample": j + 1,
                        "r02_minus_comparator_auroc": value,
                    }
                )
        fixed_count = int(np.count_nonzero(r02 >= RELIABILITY_THRESHOLD))
        for name, score in method_scores.items():
            all_coverage += coverage_rows(cohort, name, score, dice, ids, fixed_count)
        for i, case_id in enumerate(ids):
            all_score_rows.append(
                {
                    "cohort": cohort,
                    "case_id": case_id,
                    "subject_id": groups[i],
                    "calibrated_dice": dice[i],
                    "failure_dice_below_0_50": int(not success[i]),
                    "predicted_area_fraction": matrix[i, 0],
                    "maximum_calibrated_foreground_probability": matrix[i, 1],
                    "r02_reliability": r02[i],
                    "area_comparator_reliability": comparator[i, 0],
                    "max_probability_comparator_reliability": comparator[i, 1],
                }
            )
        cohort_results[cohort] = {
            "cases": len(ids),
            "patients": len(set(groups)),
            "methods": summaries,
            "paired_comparisons": differences,
            "r02_fixed_policy_accepted_count": fixed_count,
            "r02_fixed_policy_accepted_coverage": fixed_count / len(ids),
        }
    gate_checks = [
        value["gate_passed"]
        for cohort in cohort_results.values()
        for value in cohort["paired_comparisons"].values()
    ]
    passed = all(gate_checks)
    write_csv(ART / "external_scores.csv", all_score_rows)
    write_csv(ART / "risk_coverage.csv", all_coverage)
    write_csv(ART / "paired_bootstrap_differences.csv", all_bootstrap)
    model_states = []
    for name, item in development["frozen"].items():
        scaler = item["model"].named_steps["standardscaler"]
        logistic = item["model"].named_steps["logisticregression"]
        model_states.append(
            {
                "comparator": name,
                "pickle_path": item["path"].relative_to(ROOT).as_posix(),
                "pickle_sha256": item["sha256"],
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "coefficient": logistic.coef_[0].tolist(),
                "intercept": logistic.intercept_.tolist(),
                "frozen_before_external_outcome_join": True,
                "frozen_at": frozen_at,
            }
        )
    write_json(
        ART / "comparator_states.json", {"schema_version": 1, "states": model_states}
    )
    result = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "status": "PASS" if passed else "FAIL",
        "decision": "PASS" if passed else "FAIL",
        "exploratory_post_review": True,
        "development": {
            "cases": 680,
            "comparator_oof_metrics": {
                name: metric_summary(
                    development["success"],
                    development["oof"][:, i],
                    development["groups"],
                )
                for i, name in enumerate(config["comparators"])
            },
            "r02_oof_metrics": metric_summary(
                development["success"], development["r02"], development["groups"]
            ),
        },
        "external": cohort_results,
        "gate": {
            **config["gate"],
            "checks_passed": sum(gate_checks),
            "checks_total": 4,
            "status": "PASS" if passed else "FAIL",
        },
        "comparator_states": model_states,
        "verification": {
            "comparator_fit_completed_before_external_outcome_join": True,
            "source_hashes_verified": True,
            "external_inference_reproduced_committed_dice_pixels_and_r02_within_1e-12": True,
            "feature_contract": "prediction-only frozen E08+C03 outputs",
            "final_stack_manifest_unchanged": sha256(FINAL / "manifest.json")
            == frozen_manifest_before,
            "brisc_test_images_accessed": 0,
            "neural_retraining": 0,
            "r02_refitting": 0,
            "threshold_changes": 0,
            "segmentation_threshold": 0.5,
        },
    }
    write_json(EXP / "result.json", result)
    config["status"] = "COMPLETE"
    write_json(EXP / "config.json", config)
    lines = [
        "# Decision — R12 external baselines",
        "",
        f"**Status:** {result['status']}",
        "",
        "The two one-feature comparators were fitted only on BRISC development cases, frozen, and then applied unchanged to each external cohort. The conjunctive four-comparison gate was not relaxed.",
        "",
    ]
    for cohort in ("x02", "x03"):
        lines.append(
            f"- **{cohort.upper()}:** R02 AUROC `{cohort_results[cohort]['methods']['r02']['failure_detection_auroc']:.6f}`; area comparator `{cohort_results[cohort]['methods']['predicted_area_fraction']['failure_detection_auroc']:.6f}`; maximum-probability comparator `{cohort_results[cohort]['methods']['maximum_calibrated_foreground_probability']['failure_detection_auroc']:.6f}`."
        )
    lines += [
        "",
        "This is a post-review exploratory baseline analysis on previously analyzed external cohorts. It is not untouched confirmation or clinical validation.",
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
        ART / "brisc_oof_comparator_scores.csv",
        ART / "external_scores.csv",
        ART / "risk_coverage.csv",
        ART / "paired_bootstrap_differences.csv",
        ART / "comparator_states.json",
        ART / "x02_frozen_feature_cache.csv",
        ART / "x03_frozen_feature_cache.csv",
    ] + [item["path"] for item in development["frozen"].values()]
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
        },
    )
    with (ROOT / "audit/review_log.md").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(
            f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Complete R12 external reliability baselines\n\n- **Decision made:** Record H11/R12 as `{result['status']}` under the four-way conjunctive gate.\n- **Triggering artifact:** `experiments/r12_r02_external_baselines/result.json`.\n- **Rationale:** BRISC-only one-feature comparators were frozen before external outcome joins and applied unchanged.\n- **Verification:** All source, membership, fold, and inference reproduction checks passed; zero sealed-test access, neural retraining, R02 refitting, or threshold changes.\n- **Who approved:** User authorized completion of all Amendment 21 items.\n"
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "external": cohort_results,
                "gate": result["gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
