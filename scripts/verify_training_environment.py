import gc
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import monai
import psutil
import torch
import torchvision
from monai.networks.nets import UNet
from torchvision.models import resnet18

PROJECT = Path(__file__).resolve().parents[1]


def benchmark(name, factory, segmentation=False):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = factory().cuda()
    model.train()

    batch_size = 2 if segmentation else 4
    channels = 1 if segmentation else 3

    inputs = torch.randn(
        batch_size, channels, 256, 256, device="cuda"
    )

    if segmentation:
        targets = torch.randint(
            0, 2, (batch_size, 1, 256, 256), device="cuda"
        ).float()
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        targets = torch.arange(batch_size, device="cuda") % 4
        criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")

    first_parameter = next(model.parameters())
    original_parameter = first_parameter.detach().clone()

    step_times = []
    final_loss = None

    for step in range(4):
        torch.cuda.synchronize()
        started = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda", dtype=torch.float16
        ):
            predictions = model(inputs)
            loss = criterion(predictions, targets)

        if not torch.isfinite(loss).item():
            raise RuntimeError(f"{name}: non-finite loss")

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        # Exclude the first warm-up step from timing.
        if step > 0:
            step_times.append(elapsed)

        final_loss = float(loss.detach().cpu())

    changed = not torch.equal(
        original_parameter, first_parameter.detach()
    )
    if not changed:
        raise RuntimeError(f"{name}: weights did not update")

    result = {
        "batch_size": batch_size,
        "image_size": 256,
        "mean_step_seconds": sum(step_times) / len(step_times),
        "peak_allocated_vram_gib": (
            torch.cuda.max_memory_allocated() / 1024**3
        ),
        "peak_reserved_vram_gib": (
            torch.cuda.max_memory_reserved() / 1024**3
        ),
        "synthetic_loss": final_loss,
        "weights_updated": changed,
    }

    print(name, json.dumps(result, indent=2), flush=True)
    return result


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Fix installation first.")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    output = (
        PROJECT
        / "audit"
        / "environment"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)

    report = {
        "purpose": "Synthetic environment test; not model accuracy",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "monai": monai.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "system_ram_available_gib": (
            psutil.virtual_memory().available / 1024**3
        ),
    }

    report["classifier"] = benchmark(
        "classifier",
        lambda: resnet18(weights=None, num_classes=4),
    )

    report["segmenter"] = benchmark(
        "segmenter",
        lambda: UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(16, 32, 64),
            strides=(2, 2),
            num_res_units=1,
        ),
        segmentation=True,
    )

    runs = PROJECT / "runs"
    runs.mkdir(exist_ok=True)
    artifact_directory = runs / "mlflow_artifacts"
    artifact_directory.mkdir(exist_ok=True)

    database = (runs / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{database}")

    experiment_name = "environment-checks"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            artifact_location=artifact_directory.as_uri(),
        )
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(
        experiment_id=experiment_id,
        run_name="synthetic-gpu-check",
    ) as run:
        mlflow.log_params({
            "gpu": report["gpu"],
            "torch": report["torch"],
            "monai": report["monai"],
            "data": "synthetic",
        })

        for model_name in ("classifier", "segmenter"):
            for metric in (
                "mean_step_seconds",
                "peak_allocated_vram_gib",
                "peak_reserved_vram_gib",
                "synthetic_loss",
            ):
                mlflow.log_metric(
                    f"{model_name}_{metric}",
                    report[model_name][metric],
                )

        report["mlflow_run_id"] = run.info.run_id
        report["status"] = "PASS"
        result_file = output / "environment_check.json"
        result_file.write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(result_file))

    print("\nENVIRONMENT CHECK PASSED")
    print("Saved results:", result_file)
    print("This does not establish BRISC model accuracy.")


if __name__ == "__main__":
    main()
