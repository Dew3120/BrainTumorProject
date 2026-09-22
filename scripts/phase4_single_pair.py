"""Print one complete batch, then optionally run a bounded one-image diagnostic."""

import argparse
import csv
import json
import shutil
from datetime import UTC, datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import phase4_segmentation_smoke as smoke
import torch
from monai.losses import DiceLoss


def stats(tensor):
    value = tensor.detach()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "min": value.min().item(),
        "max": value.max().item(),
        "mean": value.mean().item(),
        "finite": torch.isfinite(value).all().item(),
    }


def breakdown(logits, target):
    foreground = target.bool()
    per_pixel = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    foreground_contribution = per_pixel[foreground].sum() / target.numel()
    background_contribution = per_pixel[~foreground].sum() / target.numel()
    dice_loss = DiceLoss(sigmoid=True)(logits, target)
    probability = logits.sigmoid()
    manual_dice = 1 - (2 * (probability * target).sum() + 1e-5) / (
        probability.sum() + target.sum() + 1e-5
    )
    torch.testing.assert_close(dice_loss, manual_dice, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        foreground_contribution + background_contribution, per_pixel.mean()
    )
    dice, _ = smoke.mask_scores(logits, target)
    metrics = {
        "dice": dice.item(),
        "dice_loss": dice_loss.item(),
        "bce_foreground_mean": per_pixel[foreground].mean().item(),
        "bce_background_mean": per_pixel[~foreground].mean().item(),
        "bce_foreground_contribution": foreground_contribution.item(),
        "bce_background_contribution": background_contribution.item(),
        "bce_total": per_pixel.mean().item(),
        "total_loss": (dice_loss + per_pixel.mean()).item(),
        "predicted_foreground_pixels": int((logits >= 0).sum().item()),
        "foreground_probability_mean": probability[foreground].mean().item(),
        "background_probability_mean": probability[~foreground].mean().item(),
    }
    return (
        dice_loss + per_pixel.mean(),
        metrics,
        {
            "dice": dice_loss,
            "bce_foreground_contribution": foreground_contribution,
            "bce_background_contribution": background_contribution,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run only after reviewing the printed batch",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    source = smoke.ROOT / "runs/segmentation/20260921-174907-092984-smoke"
    row = next(
        r
        for r in smoke.read_csv(source / "selected_pairs.csv")
        if r["draft_split"] == "train"
    )
    if (
        smoke.digest(smoke.dataset_path(row["path"])) != row["sha256"]
        or smoke.digest(smoke.dataset_path(row["mask_path"]))
        != row["mask_sha256_verified"]
    ):
        raise ValueError("Input pair changed since smoke test")
    dataset = smoke.Pairs([row])
    image, target = dataset[0]
    image, target = image.unsqueeze(0).cuda(), target.unsqueeze(0).cuda()
    assert image.shape == target.shape == (1, 1, 128, 128)
    assert image.dtype == target.dtype == torch.float32
    assert torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
    assert set(target.unique().tolist()) == {0.0, 1.0}
    assert 0 < target.sum() < target.numel()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = smoke.make_model().cuda()
    model.eval()
    logits = model(image)
    assert logits.shape == target.shape and torch.isfinite(logits).all()
    _, metrics, components = breakdown(logits, target)
    gradients = {}
    for name, component in components.items():
        parameter_gradients = torch.autograd.grad(
            component, tuple(model.parameters()), retain_graph=True
        )
        if not all(torch.isfinite(g).all() for g in parameter_gradients):
            raise RuntimeError("Non-finite component gradient")
        gradients[name + "_parameter_gradient_l2"] = (
            sum(g.square().sum().item() for g in parameter_gradients) ** 0.5
        )
    preflight = {
        "pair": row["filename"],
        "split": "train",
        "batch_size": 1,
        "image": stats(image),
        "mask": stats(target),
        "mask_unique_values": target.unique().tolist(),
        "foreground_pixels": int(target.sum().item()),
        "background_pixels": int((target == 0).sum().item()),
        "raw_model_logits": stats(logits),
        "sigmoid_probabilities": stats(logits.sigmoid()),
        "initial_losses": metrics,
        "component_gradients": gradients,
        "checks": "PASS: shapes/dtypes/ranges/finiteness/binary mask/source hashes/manual Dice and BCE decomposition",
        "test_or_validation_images_used": False,
    }
    print(json.dumps(preflight, indent=2), flush=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")
    output = (
        smoke.ROOT
        / "runs/segmentation"
        / (stamp + ("-single-pair" if args.train else "-batch-inspection"))
    )
    output.mkdir(parents=True, exist_ok=False)
    smoke.save_json(output / "batch_inspection.json", preflight)
    print("RUN DIRECTORY:", output, flush=True)
    if not args.train:
        return
    rule = {
        "max_updates": 300,
        "pass_dice": 0.90,
        "consecutive_passes": 5,
        "plateau_check_start_update": 150,
        "plateau_window": 60,
        "low_dice_ceiling": 0.20,
        "plateau_range_maximum": 0.02,
    }
    config = {
        "pair": row,
        "rule": rule,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "seed": 42,
        "precision": "FP32",
        "architecture": "original smoke UNet",
        "loss": "unweighted BCE + foreground soft Dice; unchanged",
        "image_size": 128,
        "batch_size": 1,
        "augmentation": "none",
        "script_sha256": smoke.digest(__file__),
        "smoke_script_sha256": smoke.digest(smoke.__file__),
        "test_or_validation_images_used": False,
    }
    smoke.save_json(output / "config.json", config)
    shutil.copy2(__file__, output / "single_pair_script.py")
    shutil.copy2(smoke.__file__, output / "phase4_segmentation_smoke.py")
    print("PREDECLARED RULE:", json.dumps(rule), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    tracking = smoke.ROOT / "runs"
    artifacts = tracking / "mlflow_artifacts/segmentation-single-pair"
    artifacts.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{(tracking / 'mlflow.db').as_posix()}")
    exp = mlflow.get_experiment_by_name("brisc-segmentation-single-pair")
    eid = (
        exp.experiment_id
        if exp
        else mlflow.create_experiment(
            "brisc-segmentation-single-pair", artifact_location=artifacts.as_uri()
        )
    )
    history = [{"update": 0, **metrics}]
    consecutive = 0
    with mlflow.start_run(experiment_id=eid, run_name=output.name) as run:
        config["mlflow_run_id"] = run.info.run_id
        smoke.save_json(output / "config.json", config)
        mlflow.log_params({k: str(v) for k, v in config.items()})
        mlflow.log_artifact(str(output / "batch_inspection.json"))
        mlflow.log_metrics(metrics, step=0)
        status = "BUDGET_EXHAUSTED_DIAGNOSE"
        for step in range(1, 301):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_logits = model(image)
            objective, _, _ = breakdown(train_logits, target)
            if not torch.isfinite(objective):
                raise RuntimeError("Non-finite loss")
            objective.backward()
            if any(
                not torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise RuntimeError("Non-finite gradient")
            optimizer.step()
            model.eval()
            with torch.inference_mode():
                final_logits = model(image)
                _, metrics, _ = breakdown(final_logits, target)
            history.append({"update": step, **metrics})
            mlflow.log_metrics(metrics, step=step)
            consecutive = consecutive + 1 if metrics["dice"] >= 0.90 else 0
            if step % 25 == 0 or consecutive >= 5:
                print(
                    f"Update {step}: Dice={metrics['dice']:.4f}, BCE fg/bg contributions={metrics['bce_foreground_contribution']:.5f}/{metrics['bce_background_contribution']:.5f}, predicted foreground={metrics['predicted_foreground_pixels']}",
                    flush=True,
                )
            if consecutive >= 5:
                status = "SINGLE_PAIR_MEMORIZATION_PASS"
                break
            recent = [r["dice"] for r in history[-60:]]
            if step >= 150 and max(recent) < 0.20 and max(recent) - min(recent) < 0.02:
                status = "STOP_AND_DIAGNOSE_LOW_PLATEAU"
                break
        with (output / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        torch.save(
            {
                "model_state": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "config": config,
            },
            output / "diagnostic_model.pt",
        )
        smoke.contact_sheet([row], dataset, final_logits.cpu(), output / "overlay.png")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        steps = [r["update"] for r in history]
        axes[0].plot(steps, [r["dice"] for r in history])
        axes[0].axhline(0.90, color="gray", linestyle="--")
        axes[0].set(
            xlabel="Optimizer updates", ylabel="One-image training Dice", ylim=(0, 1)
        )
        for key in (
            "dice_loss",
            "bce_foreground_contribution",
            "bce_background_contribution",
        ):
            axes[1].plot(steps, [r[key] for r in history], label=key)
        axes[1].set(xlabel="Optimizer updates", ylabel="Contribution to objective")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "loss_breakdown.png", dpi=160)
        plt.close(fig)
        result = {
            "status": status,
            "updates": step,
            "rule": rule,
            "initial_metrics": history[0],
            "final_metrics": history[-1],
            "test_or_validation_images_used": False,
            "full_training_started": False,
            "interpretation": "One-image diagnostic only; loss contributions do not by themselves establish a root cause.",
        }
        smoke.save_json(output / "result.json", result)
        for name in (
            "history.csv",
            "result.json",
            "loss_breakdown.png",
            "overlay.png",
            "config.json",
        ):
            mlflow.log_artifact(str(output / name))
        mlflow.set_tag("diagnostic_outcome", status)
    (smoke.ROOT / "runs/segmentation/latest_single_pair.txt").write_text(
        str(output), encoding="utf-8"
    )
    print(status, flush=True)


if __name__ == "__main__":
    main()
