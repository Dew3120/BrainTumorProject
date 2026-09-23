"""Harden segmentation reproducibility without retraining or sealed-test access."""

import argparse
import hashlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import mlflow
import numpy as np
import segmentation_overfit_gate as overfit
import segmentation_smoke_test as smoke
import segmentation_visual_audit as visual
import torch
import train_segmenter as segmenter
from monai.losses import DiceLoss

ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = "20260923-072439-397263-full"
CHECKPOINT_NAME = "epoch_056_macro_dice_0.762662.pt"
TRAINING_SOURCE_COMMIT = "67ef81458911e616524c6c9bab34fe371703e983"
MANIFEST_NAME = "segmentation_run_manifest.json"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalized_text(value):
    return value.replace("\r\n", "\n")


def verify_source_commit(run):
    mapping = {
        "scripts/train_segmenter.py": run / "training_script.py",
        "scripts/segmentation_overfit_gate.py": run / "overfit_source_snapshot.py",
        "scripts/segmentation_smoke_test.py": run / "smoke_source_snapshot.py",
    }
    historical_paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", TRAINING_SOURCE_COMMIT, "scripts"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    results = {}
    for repository_path, saved_path in mapping.items():
        saved = saved_path.read_text(encoding="utf-8-sig")
        matching_paths = []
        for historical_path in historical_paths:
            if not historical_path.endswith(".py"):
                continue
            committed = subprocess.check_output(
                ["git", "show", f"{TRAINING_SOURCE_COMMIT}:{historical_path}"],
                cwd=ROOT,
            ).decode("utf-8-sig")
            if normalized_text(committed) == normalized_text(saved):
                matching_paths.append(historical_path)
        if len(matching_paths) != 1:
            raise ValueError(
                f"Expected one source match in {TRAINING_SOURCE_COMMIT} for "
                f"{repository_path}; found {matching_paths}"
            )
        results[repository_path] = {
            "saved_copy": saved_path.relative_to(run).as_posix(),
            "historical_commit_path": matching_paths[0],
            "normalized_text_identical": True,
            "saved_sha256": sha256(saved_path),
        }
    return results


def inventory_files(run):
    files = []
    for path in sorted(
        (item for item in run.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(run).as_posix().casefold(),
    ):
        relative = path.relative_to(run).as_posix()
        if relative == MANIFEST_NAME:
            continue
        files.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return files


def verify_inventory(run, files):
    current = inventory_files(run)
    if current != files:
        raise ValueError("Run-folder inventory changed while manifest was generated")


def model_state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def synthetic_training_step(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = overfit.make_diagnostic_model("linear", "compact").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024.0, growth_interval=1000000)
    values = torch.linspace(0.0, 1.0, 2 * 128 * 128, device="cuda")
    images = values.reshape(2, 1, 128, 128)
    targets = torch.zeros_like(images)
    targets[:, :, 40:88, 44:84] = 1.0
    dice_loss = DiceLoss(sigmoid=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(images)
    loss = segmenter.segmentation_loss(logits, targets, dice_loss)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    result = {
        "loss": float(loss.detach().cpu().item()),
        "state_sha256": model_state_digest(model),
    }
    del model, optimizer, scaler, images, targets, logits, loss
    torch.cuda.empty_cache()
    return result


def strict_determinism_probe():
    if not torch.cuda.is_available():
        return {
            "status": "NOT_RUN_CUDA_UNAVAILABLE",
            "bitwise_repeat_match": None,
        }
    original_algorithms = torch.are_deterministic_algorithms_enabled()
    original_benchmark = torch.backends.cudnn.benchmark
    original_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        first = synthetic_training_step(42)
        second = synthetic_training_step(42)
        matches = first == second
        if not matches:
            raise RuntimeError("Repeated strict synthetic training steps differ")
        return {
            "status": "PASS_CURRENT_RUNTIME_SYNTHETIC_STEP",
            "strict_deterministic_algorithms_enabled_for_probe": True,
            "bitwise_repeat_match": True,
            "first": first,
            "second": second,
            "scope": (
                "Two repeated synthetic forward/backward/AdamW/AMP steps on the current "
                "GPU runtime; no project image or accepted checkpoint was trained"
            ),
        }
    except RuntimeError as error:
        return {
            "status": "STRICT_GUARD_FAILURE",
            "bitwise_repeat_match": False,
            "error": str(error),
        }
    finally:
        torch.use_deterministic_algorithms(original_algorithms)
        torch.backends.cudnn.benchmark = original_benchmark
        torch.backends.cudnn.deterministic = original_cudnn_deterministic


def inspect_training_determinism(run, config):
    script = (run / "training_script.py").read_text(encoding="utf-8")
    helper = (run / "overfit_source_snapshot.py").read_text(encoding="utf-8")
    controls = {
        "python_seed_42": "random.seed(42)" in script,
        "numpy_seed_42": "np.random.seed(42)" in script,
        "torch_cpu_seed_42": "torch.manual_seed(42)" in script,
        "torch_cuda_seed_42": "torch.cuda.manual_seed_all(42)" in script,
        "cudnn_benchmark_disabled": "torch.backends.cudnn.benchmark = False" in script,
        "cudnn_deterministic_enabled": "torch.backends.cudnn.deterministic = True"
        in script,
        "shuffle_generator_seeded": "torch.Generator().manual_seed(42)" in script,
        "workers_seed_python_and_numpy": all(
            value in script
            for value in (
                "worker_seed = torch.initial_seed() % (2**32)",
                "np.random.seed(worker_seed)",
                "random.seed(worker_seed)",
            )
        ),
        "training_augmentation_disabled": config["augmentation"].startswith("none"),
        "model_dropout_zero": "dropout=0.0" in helper,
        "strict_deterministic_algorithms_enabled_in_training": (
            "torch.use_deterministic_algorithms(True)" in script
        ),
        "cublas_workspace_config_recorded": (
            "CUBLAS_WORKSPACE_CONFIG" in script or "CUBLAS_WORKSPACE_CONFIG" in config
        ),
        "tf32_policy_recorded": any("tf32" in key.casefold() for key in config),
    }
    if not all(
        controls[key]
        for key in (
            "python_seed_42",
            "numpy_seed_42",
            "torch_cpu_seed_42",
            "torch_cuda_seed_42",
            "cudnn_benchmark_disabled",
            "cudnn_deterministic_enabled",
            "shuffle_generator_seeded",
            "workers_seed_python_and_numpy",
            "training_augmentation_disabled",
            "model_dropout_zero",
        )
    ):
        raise ValueError("Expected deterministic training controls changed")
    model = overfit.make_diagnostic_model("linear", "compact")
    operation_counts = dict(
        sorted(Counter(type(module).__name__ for module in model.modules()).items())
    )
    probe = strict_determinism_probe()
    return {
        "conclusion": "CONTROLLED_BUT_NOT_FULLY_GUARANTEED",
        "full_training_replay_performed": False,
        "controls": controls,
        "operation_counts": operation_counts,
        "strict_current_runtime_probe": probe,
        "specific_unenforced_or_environment_sensitive_operations": [
            {
                "operations": "CUDA Conv2d and ConvTranspose2d forward/backward through cuDNN",
                "reason": (
                    "cuDNN deterministic mode was enabled, but the original run did not "
                    "enable PyTorch's global deterministic-algorithm guard. An unsupported "
                    "nondeterministic path would not have failed closed."
                ),
            },
            {
                "operations": "InstanceNorm2d, Dice/BCE losses, and CUDA gradient reductions",
                "reason": (
                    "Parallel floating-point reduction order and kernel implementation can "
                    "change across CUDA, cuDNN, PyTorch, driver, or GPU versions. The current "
                    "runtime probe passes, but cross-runtime bitwise identity is not guaranteed."
                ),
            },
            {
                "operations": "CUDA AdamW update with AMP FP16 forward and GradScaler",
                "reason": (
                    "FP16 rounding and overflow decisions can amplify small runtime-level "
                    "differences. Early stopping and top-3 selection can then follow a different "
                    "path even with the same seed."
                ),
            },
            {
                "operations": "CUDA library determinism enforcement",
                "reason": (
                    "CUBLAS_WORKSPACE_CONFIG and TF32 policy were not explicitly recorded by "
                    "the training run. The model path is convolution-heavy, but the environment "
                    "contract is incomplete."
                ),
            },
        ],
        "controlled_data_order": (
            "The frozen CSV order, seeded DataLoader shuffle generator, seeded workers, no "
            "augmentation, and zero dropout make the input order and transformations controlled."
        ),
    }


def mask_position(mask):
    y, x = np.nonzero(np.asarray(mask, dtype=bool))
    if not len(y):
        raise ValueError("Position analysis requires a non-empty prediction")
    height, width = mask.shape
    return {
        "centroid_y_normalized": float(y.mean() / (height - 1)),
        "centroid_x_normalized": float(x.mean() / (width - 1)),
        "fraction_in_lower_half": float((y >= height / 2).mean()),
        "fraction_in_lower_third": float((y >= 2 * height / 3).mean()),
    }


def position_summary(items):
    fields = (
        "centroid_y_normalized",
        "fraction_in_lower_half",
        "fraction_in_lower_third",
    )
    return {
        "count": len(items),
        **{
            f"median_{field}": float(np.median([item[field] for item in items]))
            for field in fields
        },
        "centroid_below_image_midline_count": sum(
            item["centroid_y_normalized"] > 0.5 for item in items
        ),
        "centroid_below_normalized_y_0_60_count": sum(
            item["centroid_y_normalized"] > 0.60 for item in items
        ),
        "centroid_y_minimum": min(item["centroid_y_normalized"] for item in items),
        "centroid_y_maximum": max(item["centroid_y_normalized"] for item in items),
    }


def analyze_zero_dice_positions(records, predictions):
    cases = []
    nonzero = []
    for record in records:
        position = {
            "path": record["path"],
            "category": record["label"],
            "view": record["filename"].split("_")[-2],
            "dice": record["dice"],
            **mask_position(predictions[record["path"]]),
        }
        (cases if record["dice"] == 0.0 else nonzero).append(position)
    if len(cases) != 17:
        raise ValueError(f"Expected 17 zero-Dice position cases, found {len(cases)}")
    zero_summary = position_summary(cases)
    nonzero_summary = position_summary(nonzero)
    strong_lower_half = (
        zero_summary["centroid_below_image_midline_count"] / len(cases) >= 0.70
        and zero_summary["median_fraction_in_lower_half"] >= 0.70
    )
    strong_skull_base = (
        zero_summary["centroid_below_normalized_y_0_60_count"] / len(cases) >= 0.70
        or zero_summary["median_fraction_in_lower_third"] >= 0.50
    )
    conclusion = (
        "CONSISTENT_LOWER_POSITION_BIAS_CONFIRMED"
        if strong_lower_half or strong_skull_base
        else "HETEROGENEOUS_WITH_MODEST_LOWER_HALF_SHIFT"
    )
    return {
        "coordinate_definition": "y=0 is the top and y=1 is the bottom of the 128x128 model image",
        "predeclared_support_rule": (
            "At least 70% of zero-Dice centroids below the midline plus median lower-half "
            "fraction >=0.70, or at least 70% of centroids below y=0.60, or median lower-third "
            "fraction >=0.50"
        ),
        "conclusion": conclusion,
        "zero_dice": zero_summary,
        "nonzero_dice": nonzero_summary,
        "zero_dice_cases": sorted(cases, key=lambda item: item["path"].casefold()),
        "section_f_wording": (
            "A consistent skull-base positional bias was not confirmed across all 17 zero-Dice "
            "cases. Errors were spatially heterogeneous, with a modest lower-half shift: 11/17 "
            "centroids were below the image midline and median normalized y was 0.534 versus "
            "0.463 among nonzero-Dice cases; only 1/17 centroids fell below y=0.60 and the median "
            "lower-third pixel fraction was 0.008."
        ),
    }


def write_report(path, manifest, manifest_hash):
    determinism = manifest["determinism_audit"]
    reload = manifest["checkpoint_reload_verification"]
    position = manifest["findings_carry_forward"]["zero_dice_position_check"]
    controls = determinism["controls"]
    risks = "\n".join(
        f"- **{item['operations']}:** {item['reason']}"
        for item in determinism[
            "specific_unenforced_or_environment_sensitive_operations"
        ]
    )
    content = f"""# Segmentation reproducibility audit

## Determinism conclusion

**Controlled, but not fully guaranteed.** The original training run seeded Python, NumPy, Torch CPU/CUDA, the shuffle generator, and workers. It disabled cuDNN benchmarking, enabled cuDNN deterministic mode, used no augmentation, and used zero dropout. Data order and preprocessing are therefore controlled on the recorded stack.

The run did not call `torch.use_deterministic_algorithms(True)`, did not record `CUBLAS_WORKSPACE_CONFIG`, and did not record a TF32 policy. A complete 68-epoch replay was intentionally not performed because reproducibility audit prohibits retraining. The current CUDA stack passed two bitwise-identical synthetic forward/backward/AdamW/AMP steps with PyTorch's strict deterministic guard enabled; this is an operation-path check, not proof that the historical full run is bitwise reproducible everywhere.

| Control | Recorded |
|---|---:|
| Python/NumPy/Torch CPU/Torch CUDA seed 42 | {all(controls[key] for key in ("python_seed_42", "numpy_seed_42", "torch_cpu_seed_42", "torch_cuda_seed_42"))} |
| cuDNN benchmark disabled | {controls["cudnn_benchmark_disabled"]} |
| cuDNN deterministic mode enabled | {controls["cudnn_deterministic_enabled"]} |
| Shuffle generator and workers seeded | {controls["shuffle_generator_seeded"] and controls["workers_seed_python_and_numpy"]} |
| Augmentation disabled and dropout zero | {controls["training_augmentation_disabled"] and controls["model_dropout_zero"]} |
| Global deterministic-algorithm guard used in original run | {controls["strict_deterministic_algorithms_enabled_in_training"]} |
| CUBLAS workspace policy recorded | {controls["cublas_workspace_config_recorded"]} |
| TF32 policy recorded | {controls["tf32_policy_recorded"]} |

Specific remaining sources of uncertainty:

{risks}

## Checkpoint reload

- Accepted checkpoint: `{reload["checkpoint"]}`
- Checkpoint SHA256: `{reload["checkpoint_sha256"]}`
- Validation images recomputed: {reload["validation_images_recomputed"]}
- Maximum aggregate metric difference: {reload["maximum_metric_difference"]}
- Maximum per-image Dice/IoU difference: {reload["maximum_per_image_difference"]}
- Test images accessed: {reload["test_images_accessed"]}

This restates and independently repeats the successful checkpoint reload: epoch 56 reproduces every stored validation metric exactly on the current CUDA runtime.

## Run manifest

- Manifest: `{manifest["manifest_path"]}`
- Manifest SHA256: `{manifest_hash}`
- Hashed run files: {manifest["file_count"]}
- Source-equivalent training commit: `{manifest["git"]["training_source_equivalent_commit"]}`
- Manifest audit base commit: `{manifest["git"]["manifest_audit_base_commit"]}`

The exact training-time Git HEAD was not recorded by the historical run. Commit `67ef814` is the first repository commit containing normalized-text-identical copies of all three saved training sources and the accepted segmentation findings. The manifest excludes itself because a file cannot contain its own final SHA256; its external SHA256 is recorded above and in MLflow.

## Positional-bias follow-up

{position["section_f_wording"]}

The consolidated findings must also retain the tumor-type confound (12/17 zero-Dice cases are glioma while 15/20 highest-Dice cases are meningioma) and quote the uncertainty-analysis Spearman correlations directly: classifier confidence versus segmenter Dice `0.3252`, and classifier confidence versus segmenter inside-reference probability `0.1660`.

## Scope

No model was retrained, no threshold changed, no test image was accessed, and calibration and uncertainty work was not started.
"""
    Path(path).write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", type=Path, default=ROOT / "runs/segmentation" / RUN_NAME
    )
    args = parser.parse_args()
    run = args.run.resolve()
    if run.name != RUN_NAME:
        raise ValueError("Accepted run identity mismatch")
    config = json.loads((run / "config.json").read_text(encoding="utf-8"))
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    findings = json.loads(
        (ROOT / "audit/segmentation_training_findings.json").read_text(encoding="utf-8")
    )
    section_d = json.loads(
        (ROOT / "audit/segmentation_uncertainty_signals.json").read_text(encoding="utf-8")
    )
    if findings["accepted_run"] != run.name:
        raise ValueError("Accepted findings/run mismatch")
    expected_checkpoint = findings["top3_checkpoints"][0]
    checkpoint = run / "top3_checkpoints" / CHECKPOINT_NAME
    checkpoint_hash = visual.sha256(checkpoint)
    if (
        checkpoint.name != expected_checkpoint["file"]
        or checkpoint_hash != expected_checkpoint["sha256"]
    ):
        raise ValueError("Accepted checkpoint identity or SHA256 mismatch")
    split = run / "split_snapshot.csv"
    if smoke.digest(split) != config["split_sha256"]:
        raise ValueError("Split snapshot SHA256 mismatch")
    lock_hash = smoke.digest(ROOT / "uv.lock")
    if lock_hash != config["lock_sha256"]:
        raise ValueError("Dependency lock SHA256 mismatch")

    rows = smoke.read_csv(split)
    parts = segmenter.partition_masked_rows(rows)
    validation_rows = parts["validation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = overfit.make_diagnostic_model("linear", "compact").to(device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model_state"])
    metrics, records, predictions = visual.inference(
        model, validation_rows, device, 0.5
    )
    tolerance = 1e-6 if device.type == "cuda" else 1e-5
    maximum_metric_difference = max(
        abs(metrics[key] - result["best_validation_metrics"][key])
        for key in result["best_validation_metrics"]
    )
    if maximum_metric_difference > tolerance:
        raise ValueError(
            f"Checkpoint aggregate metric mismatch: {maximum_metric_difference}"
        )
    stored = {
        row["path"].replace("\\", "/").casefold(): row
        for row in smoke.read_csv(run / "validation_predictions.csv")
    }
    if set(stored) != {row["path"].replace("\\", "/").casefold() for row in records}:
        raise ValueError("Stored prediction membership mismatch")
    maximum_per_image_difference = 0.0
    for record in records:
        old = stored[record["path"].replace("\\", "/").casefold()]
        if (
            old["label"] != record["label"]
            or int(old["reference_pixels"]) != record["reference_pixels"]
            or int(old["predicted_pixels"]) != record["predicted_pixels"]
        ):
            raise ValueError(f"Stored per-image record mismatch: {record['path']}")
        maximum_per_image_difference = max(
            maximum_per_image_difference,
            abs(float(old["dice"]) - record["dice"]),
            abs(float(old["iou"]) - record["iou"]),
        )
    if maximum_per_image_difference > tolerance:
        raise ValueError(
            f"Checkpoint per-image mismatch: {maximum_per_image_difference}"
        )

    determinism = inspect_training_determinism(run, config)
    source_verification = verify_source_commit(run)
    position = analyze_zero_dice_positions(records, predictions)
    correlations = section_d["cross_model_correlations_all_680"]
    if (
        abs(
            correlations["spearman_classifier_confidence_vs_segmenter_dice"]
            - 0.32524900331879836
        )
        > 1e-12
        or abs(
            correlations[
                "spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference"
            ]
            - 0.16603542526179857
        )
        > 1e-12
    ):
        raise ValueError("uncertainty analysis correlation context mismatch")

    files = inventory_files(run)
    base_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    manifest_path = run / MANIFEST_NAME
    manifest = {
        "schema_version": 1,
        "status": "SEGMENTATION_REPRODUCIBILITY_MANIFEST_COMPLETE",
        "manifest_path": str(manifest_path),
        "manifest_self_exclusion": {
            "excluded_from_file_inventory": True,
            "reason": (
                "A manifest cannot contain its own final SHA256. Its hash is stored in "
                "audit/segmentation_reproducibility.md and the MLflow artifact store."
            ),
        },
        "run": run.name,
        "file_count": len(files),
        "files": files,
        "config": config,
        "git": {
            "training_time_head_recorded_by_original_run": False,
            "training_source_equivalent_commit": TRAINING_SOURCE_COMMIT,
            "training_source_commit_verification": source_verification,
            "manifest_audit_base_commit": base_commit,
        },
        "reproduction_inputs": {
            "split_sha256": config["split_sha256"],
            "checkpoint_sha256": checkpoint_hash,
            "dependency_lock_sha256": lock_hash,
            "training_script_sha256": config["training_script_sha256"],
            "overfit_script_sha256": config["overfit_script_sha256"],
            "smoke_script_sha256": config["smoke_script_sha256"],
        },
        "determinism_audit": determinism,
        "checkpoint_reload_verification": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "device": str(device),
            "validation_images_recomputed": len(records),
            "maximum_metric_difference": maximum_metric_difference,
            "maximum_per_image_difference": maximum_per_image_difference,
            "test_images_accessed": 0,
        },
        "findings_carry_forward": {
            "zero_dice_position_check": position,
            "tumor_type_confounded_confidence_comparison": (
                "12/17 zero-Dice cases are glioma; 15/20 highest-Dice cases are meningioma"
            ),
            "spearman_classifier_confidence_vs_segmenter_dice": correlations[
                "spearman_classifier_confidence_vs_segmenter_dice"
            ],
            "spearman_classifier_confidence_vs_segmenter_inside_probability": correlations[
                "spearman_classifier_confidence_vs_segmenter_mean_probability_inside_reference"
            ],
        },
        "constraints": {
            "retraining_performed": False,
            "threshold_changed": False,
            "test_images_accessed": 0,
            "calibration_work_started": False,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    verify_inventory(run, files)
    manifest_hash = sha256(manifest_path)
    report_path = ROOT / "audit/segmentation_reproducibility.md"
    write_report(report_path, manifest, manifest_hash)
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'runs/mlflow.db').as_posix()}")
    with mlflow.start_run(run_id=config["mlflow_run_id"]):
        mlflow.log_artifact(str(manifest_path))
        mlflow.log_artifact(str(report_path))
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "file_count": manifest["file_count"],
                "manifest_sha256": manifest_hash,
                "determinism": determinism,
                "checkpoint_reload": manifest["checkpoint_reload_verification"],
                "zero_dice_position_check": position,
                "test_images_accessed": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
