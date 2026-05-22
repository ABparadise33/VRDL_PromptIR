"""Train the unmodified PromptIR baseline on the HW4 rain/snow dataset."""

import argparse
import csv
import os
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from promptir.data import (
    HW4RainSnowTrainDataset,
    HW4RainSnowValDataset,
    build_paired_samples,
    split_train_val_samples,
)
from promptir.model import PromptIR


METRIC_FIELDS = ["epoch", "train_l1", "val_psnr", "lr", "steps"]


class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Official PromptIR warmup + cosine scheduler behavior."""

    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"]
                + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        return [
            (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            / (
                1
                + np.cos(
                    np.pi
                    * (self.last_epoch - self.warmup_epochs - 1)
                    / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="hw4_realse_dataset")
    parser.add_argument("--output-dir", default="runs/promptir_baseline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--val-max-images", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
    )
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    return parser.parse_args()


def resolve_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best_psnr):
    checkpoint = {
        "epoch": epoch,
        "best_psnr": best_psnr,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, path)


def load_metric_history(path, before_epoch):
    if not path.exists():
        return []

    history = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            epoch = int(row["epoch"])
            if epoch < before_epoch:
                history.append(
                    {
                        "epoch": epoch,
                        "train_l1": float(row["train_l1"]),
                        "val_psnr": (
                            float(row["val_psnr"])
                            if row.get("val_psnr") not in (None, "")
                            else None
                        ),
                        "lr": float(row["lr"]),
                        "steps": int(row["steps"]),
                    }
                )
    return history


def save_metric_history(path, history):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(history)


def plot_loss_curve(path, history):
    cache_dir = path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped loss curve plotting.")
        return

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    losses = [row["train_l1"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Train L1 Loss")
    plt.title("PromptIR Training Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_psnr_curve(path, history):
    cache_dir = path.parent / ".matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped PSNR curve plotting.")
        return

    points = [
        (row["epoch"], row.get("val_psnr"))
        for row in history
        if row.get("val_psnr") is not None
    ]
    if not points:
        return

    epochs, psnrs = zip(*points)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, psnrs, marker="o", linewidth=1.8)
    plt.xlabel("Epoch")
    plt.ylabel("Validation PSNR (dB)")
    plt.title("PromptIR Validation PSNR")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def pad_to_multiple(input_tensor, multiple=8):
    height, width = input_tensor.shape[-2:]
    padded_h = ((height + multiple - 1) // multiple) * multiple
    padded_w = ((width + multiple - 1) // multiple) * multiple
    pad_h = padded_h - height
    pad_w = padded_w - width
    if pad_h == 0 and pad_w == 0:
        return input_tensor, height, width
    padded = torch.nn.functional.pad(input_tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, height, width


def batch_psnr(restored, clean):
    restored = restored.detach().clamp(0.0, 1.0)
    clean = clean.detach().clamp(0.0, 1.0)
    mse = torch.mean((restored - clean) ** 2, dim=(1, 2, 3))
    psnr = torch.where(
        mse == 0,
        torch.full_like(mse, float("inf")),
        -10.0 * torch.log10(mse),
    )
    return psnr


def validate(model, loader, device, max_images=0):
    if loader is None:
        return None

    model.eval()
    psnr_sum = 0.0
    image_count = 0
    with torch.no_grad():
        progress = tqdm(loader, desc="Validate", leave=False)
        for names, degraded, clean in progress:
            del names
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            padded, height, width = pad_to_multiple(degraded)
            restored = model(padded)[..., :height, :width]
            psnrs = batch_psnr(restored, clean)
            psnr_sum += psnrs.sum().item()
            image_count += psnrs.numel()
            progress.set_postfix(psnr=f"{psnr_sum / image_count:.2f}")
            if 0 < max_images <= image_count:
                break

    model.train()
    if image_count == 0:
        return None
    return psnr_sum / image_count


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = build_paired_samples(args.dataset_root)
    train_samples, val_samples = split_train_val_samples(
        all_samples,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    dataset = HW4RainSnowTrainDataset(
        args.dataset_root,
        patch_size=args.patch_size,
        augment=not args.no_augment,
        samples=train_samples,
    )
    val_dataset = HW4RainSnowValDataset(val_samples) if val_samples else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

    model = PromptIR(decoder=True).to(device)
    loss_fn = nn.L1Loss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
    )

    start_epoch = 1
    if args.resume is not None:
        try:
            checkpoint = torch.load(
                args.resume,
                map_location=device,
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1

    metrics_path = output_dir / "metrics.csv"
    loss_curve_path = output_dir / "loss_curve.png"
    psnr_curve_path = output_dir / "psnr_curve.png"
    metric_history = load_metric_history(metrics_path, before_epoch=start_epoch)
    if args.resume is not None:
        best_psnr = checkpoint.get("best_psnr")
    else:
        best_psnr = None
    if best_psnr is None and metric_history:
        val_psnrs = [
            row["val_psnr"]
            for row in metric_history
            if row.get("val_psnr") is not None
        ]
        best_psnr = max(val_psnrs) if val_psnrs else None

    print(f"Device: {device}")
    print(f"Training pairs: {len(dataset)}")
    print(f"Validation pairs: {len(val_dataset) if val_dataset is not None else 0}")
    print(f"Checkpoints: {output_dir}")
    print(f"Metrics: {metrics_path}")
    print(f"Loss curve: {loss_curve_path}")
    print(f"PSNR curve: {psnr_curve_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        step_count = 0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, (_, degraded, clean) in enumerate(progress, start=1):
            step_count = step
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            restored = model(degraded)
            loss = loss_fn(restored, clean)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")
            if 0 < args.max_steps_per_epoch <= step:
                break

        scheduler.step()
        epoch_loss = running_loss / max(1, step_count)
        val_psnr = validate(
            model,
            val_loader,
            device,
            max_images=args.val_max_images,
        )
        lr = optimizer.param_groups[0]["lr"]
        val_text = f", val_psnr={val_psnr:.4f}" if val_psnr is not None else ""
        print(f"Epoch {epoch}: train_l1={epoch_loss:.6f}{val_text}, lr={lr:.8f}")

        metric_history.append(
            {
                "epoch": epoch,
                "train_l1": epoch_loss,
                "val_psnr": val_psnr,
                "lr": lr,
                "steps": step_count,
            }
        )
        save_metric_history(metrics_path, metric_history)
        plot_loss_curve(loss_curve_path, metric_history)
        plot_psnr_curve(psnr_curve_path, metric_history)

        is_best = val_psnr is not None and (
            best_psnr is None or val_psnr > best_psnr
        )
        if is_best:
            best_psnr = val_psnr
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                best_psnr,
            )

        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            best_psnr,
        )
        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / f"epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                best_psnr,
            )


if __name__ == "__main__":
    main()
