"""Run Amendment 21 O07 frozen OOD-score/failure association analysis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import evaluate_x01_cheng as x01
import evaluate_x02_openneuro as x02
import evaluate_x03_upenn_gbm as x03
import final_stack_runtime as runtime
import numpy as np
import stage6_external_common as external
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/o07_ood_failure_correlation"
ART = EXP / "artifacts"
FINAL = ROOT / "runs/final_stack"
R12 = ROOT / "experiments/r12_r02_external_baselines/artifacts"
X02_SOURCE = (
    ROOT / "experiments/x02_openneuro_pituitary/artifacts/per_case_predictions.csv"
)
X03_SOURCE = ROOT / "experiments/x03_upenn_gbm/artifacts/per_case_predictions.csv"


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


def native_slice(dataset, row: dict, cohort: str) -> np.ndarray:
    image, _mask = dataset._volume(row)
    if cohort == "x02":
        return image[:, int(row["coronal_index_ras"]), :].T
    return image[:, :, int(row["axial_index_ras"])].T


def frozen_states():
    o02 = read_json(FINAL / "ood/o02_state.json")
    o05 = read_json(FINAL / "ood/o05_state.json")
    archive = np.load(FINAL / "ood/o03_mahalanobis_state.npz")
    refs = np.load(FINAL / "ood/o05_normalization_reference.npz")
    return (
        o02,
        o05,
        archive["location"].astype(np.float64),
        archive["precision"].astype(np.float64),
        {name: refs[name].astype(np.float64) for name in runtime.O05_SCORE_ORDER},
    )


def external_rows(cohort: str):
    if cohort == "x02":
        rows = x02.build_evaluation_rows(
            x02.read_csv(x02.SUBJECT_MANIFEST), x02.read_csv(x02.TUMOR_SLICE_MANIFEST)
        )
        return [row for row in rows if row["reference_has_tumor"]], x02.OpenNeuroDataset
    rows = x03.build_evaluation_rows(
        x03.read_csv(x03.SUBJECT_MANIFEST), x03.read_csv(x03.TUMOR_SLICE_MANIFEST)
    )
    return [row for row in rows if row["reference_has_tumor"]], x03.UpennDataset


def infer_ood(cohort: str, classifier, models, device) -> list[dict]:
    rows, dataset_type = external_rows(cohort)
    dataset = dataset_type(rows)
    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    o02_state, o05_state, location, precision, references = frozen_states()
    module_name = read_json(FINAL / "manifest.json")["ood"]["o03_bottleneck_module"]
    module = dict(models["v1_fallback"].named_modules()).get(module_name)
    if module is None:
        raise RuntimeError("O07 O03 bottleneck module missing")
    captured = []
    hook = module.register_forward_hook(
        lambda _m, _i, output: captured.append(
            output.detach().float().mean(dim=(2, 3)).cpu().numpy()
        )
    )
    output = [None] * len(rows)
    try:
        with torch.inference_mode():
            for batch_no, (seg_images, cls_images, _targets, indices) in enumerate(
                loader, start=1
            ):
                original_batch = len(seg_images)
                if original_batch < 16:
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
                embedding = captured.pop(0)[:original_batch].astype(np.float64)
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
                    .astype(np.float64)
                )
                for pos, index in enumerate(indices.tolist()):
                    display, _window = external.display_uint8(
                        native_slice(dataset, rows[index], cohort)
                    )
                    buffer = io.BytesIO()
                    Image.fromarray(display, mode="L").save(
                        buffer, format="PNG", optimize=False
                    )
                    image_features = runtime.o02_features(
                        display, len(buffer.getvalue())
                    )
                    o02_vector = [image_features[name] for name in runtime.O02_FEATURES]
                    o02_score = runtime._manual_logistic(
                        o02_vector,
                        o02_state["scaler_mean"],
                        o02_state["scaler_scale"],
                        o02_state["coefficients"],
                        o02_state["intercept"],
                    )
                    centered = embedding[pos] - location
                    o03_score = float(centered @ precision @ centered)
                    o04_classifier = float(1 - class_probability[pos].max())
                    o04_segmenter = float(runtime.binary_entropy(raw[pos]).mean())
                    components = {
                        "o02_probability": o02_score,
                        "o03_mahalanobis": o03_score,
                        "o04_classifier": o04_classifier,
                        "o04_segmenter_entropy": o04_segmenter,
                    }
                    normalized = [
                        float(
                            np.searchsorted(
                                references[name], components[name], side="right"
                            )
                            / len(references[name])
                        )
                        for name in runtime.O05_SCORE_ORDER
                    ]
                    score = runtime._manual_logistic(
                        normalized,
                        o05_state["scaler_mean"],
                        o05_state["scaler_scale"],
                        o05_state["coefficients"],
                        o05_state["intercept"],
                    )
                    output[index] = {
                        "cohort": cohort,
                        "case_id": rows[index]["case_id"],
                        "subject_id": rows[index]["subject_id"],
                        "category": rows[index]["label"],
                        **components,
                        **{
                            f"normalized_{name}": normalized[i]
                            for i, name in enumerate(runtime.O05_SCORE_ORDER)
                        },
                        "o05_score": score,
                    }
                if batch_no % 100 == 0 or batch_no == len(loader):
                    print(
                        f"O07 {cohort} frozen inference {batch_no}/{len(loader)}",
                        flush=True,
                    )
    finally:
        hook.remove()
    if captured:
        raise RuntimeError("O07 bottleneck capture imbalance")
    return output


def cohort_metrics(score, dice, case_ids):
    failure = dice < 0.5
    rank = np.lexsort((np.asarray([value.casefold() for value in case_ids]), -score))
    count = math.ceil(0.2 * len(score))
    high = rank[:count]
    prevalence = float(failure.mean())
    high_prevalence = float(failure[high].mean())
    return {
        "cases": len(score),
        "failures": int(failure.sum()),
        "failure_prevalence": prevalence,
        "spearman_ood_vs_dice": float(spearmanr(score, dice).statistic),
        "failure_detection_auroc": float(roc_auc_score(failure, score)),
        "failure_average_precision": float(average_precision_score(failure, score)),
        "highest_ood_quintile": {
            "cases": count,
            "failure_prevalence": high_prevalence,
            "enrichment_ratio": high_prevalence / prevalence,
            "failures_captured": int(failure[high].sum()),
            "failure_capture_rate": float(failure[high].sum() / failure.sum()),
        },
    }


def bootstrap(score, dice, case_ids, groups, repetitions=2000):
    unique = np.asarray(sorted(set(groups)))
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(42)
    records = []
    for iteration in range(repetitions):
        chosen = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in chosen])
        metric = cohort_metrics(
            score[indices], dice[indices], [case_ids[i] for i in indices]
        )
        records.append(
            {
                "resample": iteration + 1,
                "spearman_ood_vs_dice": metric["spearman_ood_vs_dice"],
                "failure_detection_auroc": metric["failure_detection_auroc"],
                "failure_average_precision": metric["failure_average_precision"],
                "highest_quintile_failure_prevalence": metric["highest_ood_quintile"][
                    "failure_prevalence"
                ],
                "highest_quintile_enrichment_ratio": metric["highest_ood_quintile"][
                    "enrichment_ratio"
                ],
                "highest_quintile_failure_capture_rate": metric["highest_ood_quintile"][
                    "failure_capture_rate"
                ],
            }
        )
    return records


def main() -> None:
    config = read_json(EXP / "config.json")
    if (
        config["status"] != "REGISTERED_NOT_STARTED"
        or sha256(ROOT / config["amendment_path"]) != config["amendment_sha256"]
    ):
        raise RuntimeError("O07 registration mismatch")
    paths = {
        "final_stack_manifest": FINAL / "manifest.json",
        "o05_ensemble_scores": ROOT
        / "experiments/o05_ensemble_ood/artifacts/ensemble_scores.csv",
        "o05_experiment_manifest": ROOT
        / "experiments/o05_ensemble_ood/artifacts/experiment_manifest.json",
        "x02_per_case_predictions": X02_SOURCE,
        "x02_experiment_manifest": ROOT
        / "experiments/x02_openneuro_pituitary/artifacts/experiment_manifest.json",
        "x03_per_case_predictions": X03_SOURCE,
        "x03_experiment_manifest": ROOT
        / "experiments/x03_upenn_gbm/artifacts/experiment_manifest.json",
    }
    for key, path in paths.items():
        if sha256(path) != config["source_hashes"][key]:
            raise RuntimeError(f"O07 source hash mismatch: {key}")
    final_before = sha256(FINAL / "manifest.json")
    verification = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_final_stack.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if verification.returncode:
        raise RuntimeError(
            "O07 final-stack fixture verification failed: " + verification.stderr
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier, models = x01.load_models(device)
    score_rows = []
    result_cohorts = {}
    bootstrap_rows = []
    for cohort, source_path, expected in (
        ("x02", X02_SOURCE, 425),
        ("x03", X03_SOURCE, 9464),
    ):
        cache = ART / f"{cohort}_frozen_ood_scores.csv"
        inferred = (
            read_csv(cache)
            if cache.exists()
            else infer_ood(cohort, classifier, models, device)
        )
        if not cache.exists():
            write_csv(cache, inferred)
        committed = {row["case_id"]: row for row in read_csv(source_path)}
        if len(inferred) != expected or len(committed) != expected:
            raise RuntimeError(f"O07 {cohort} membership count mismatch")
        case_ids = []
        groups = []
        score = []
        dice = []
        for row in inferred:
            old = committed.get(row["case_id"])
            if old is None:
                raise RuntimeError(f"O07 {cohort} case mismatch")
            value = float(row["o05_score"])
            current_dice = float(old["calibrated_dice"])
            if not np.isfinite([value, current_dice]).all():
                raise RuntimeError("O07 non-finite required value")
            case_ids.append(row["case_id"])
            groups.append(row["subject_id"])
            score.append(value)
            dice.append(current_dice)
            score_rows.append(
                {
                    **row,
                    "calibrated_dice": current_dice,
                    "failure_dice_below_0_50": int(current_dice < 0.5),
                }
            )
        score = np.asarray(score)
        dice = np.asarray(dice)
        groups = np.asarray(groups)
        metrics = cohort_metrics(score, dice, case_ids)
        boots = bootstrap(score, dice, case_ids, groups)
        fields = [key for key in boots[0] if key != "resample"]
        metrics["patient_bootstrap_ci95"] = {
            field: [
                float(x)
                for x in np.percentile([row[field] for row in boots], [2.5, 97.5])
            ]
            for field in fields
        }
        result_cohorts[cohort] = metrics
        for row in boots:
            bootstrap_rows.append({"cohort": cohort, **row})
    write_csv(ART / "external_ood_scores_with_outcomes.csv", score_rows)
    write_csv(ART / "patient_bootstrap_metrics.csv", bootstrap_rows)
    result = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "status": "COMPLETE",
        "decision": "DESCRIPTIVE_NO_GATE",
        "exploratory_post_review": True,
        "cohorts": result_cohorts,
        "gate": None,
        "verification": {
            "source_hashes_verified": True,
            "final_stack_fixtures_verified": True,
            "fixture_verifier_stdout": verification.stdout.strip(),
            "final_stack_manifest_unchanged": sha256(FINAL / "manifest.json")
            == final_before,
            "model_fitting": 0,
            "threshold_selection": 0,
            "brisc_test_images_accessed": 0,
            "neural_retraining": 0,
            "threshold_changes": 0,
        },
    }
    write_json(EXP / "result.json", result)
    config["status"] = "COMPLETE"
    write_json(EXP / "config.json", config)
    lines = [
        "# Decision — O07 OOD/failure association",
        "",
        "**Status:** COMPLETE — descriptive, no gate",
        "",
        "Frozen O05 scores were recomputed without fitting or threshold selection. Results are associations on previously analyzed external cohorts, not evidence of causal protection or prospective safety.",
        "",
    ]
    for cohort in ("x02", "x03"):
        item = result_cohorts[cohort]
        lines.append(
            f"- **{cohort.upper()}:** Spearman OOD-versus-Dice `{item['spearman_ood_vs_dice']:.4f}`; failure AUROC `{item['failure_detection_auroc']:.4f}`; top-quintile enrichment `{item['highest_ood_quintile']['enrichment_ratio']:.3f}×`; captured `{item['highest_ood_quintile']['failures_captured']}/{item['failures']}` failures."
        )
    (EXP / "decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    registry_path = ROOT / "experiments/registry.json"
    registry = read_json(registry_path)
    next(x for x in registry["experiments"] if x["id"] == EXP.name)["status"] = (
        "COMPLETE"
    )
    write_json(registry_path, registry)
    artifacts = [
        EXP / "config.json",
        EXP / "result.json",
        EXP / "decision.md",
        ART / "x02_frozen_ood_scores.csv",
        ART / "x03_frozen_ood_scores.csv",
        ART / "external_ood_scores_with_outcomes.csv",
        ART / "patient_bootstrap_metrics.csv",
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
                for path in artifacts
            ],
        },
    )
    with (ROOT / "audit/review_log.md").open(
        "a", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(
            f"\n## {datetime.now().astimezone().isoformat(timespec='seconds')} — Complete O07 OOD/failure association\n\n- **Decision made:** Close H12/O07 as a descriptive analysis with no gate.\n- **Triggering artifact:** `experiments/o07_ood_failure_correlation/result.json`.\n- **Rationale:** Frozen O05 scores were recomputed on X02 and X03 and associated with Dice/failure without tuning.\n- **Verification:** Final-stack fixtures and all hashes passed; zero fitting, threshold selection, sealed-test access, neural retraining, or threshold changes.\n- **Who approved:** User authorized completion of all Amendment 21 items.\n"
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
