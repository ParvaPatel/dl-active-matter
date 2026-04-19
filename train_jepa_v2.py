"""Video-JEPA v2 training entry point.

Improvements over train_jepa.py:
    - Cosine LR schedule with linear warmup
    - EMA momentum cosine warmup (0.996 → 0.999)
    - Gradient clipping
    - Full VICReg logging (variance + covariance losses)
"""

import os
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

try:
    import wandb
except ImportError:
    wandb = None

from data.dataset import ActiveMatterDataset
from models.encoder import SpatioTemporalViT, count_parameters
from models.jepa_v2 import VideoJEPAv2
from utils.training import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train Video-JEPA v2 on active_matter")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def get_ema_momentum(epoch, total_epochs, start=0.996, end=0.999):
    """Cosine schedule for EMA momentum: ramps from start → end over training.

    Early: lower momentum → target tracks online encoder faster (more exploration)
    Late:  higher momentum → target stabilizes (more exploitation)
    """
    return end - (end - start) * (math.cos(math.pi * epoch / total_epochs) + 1) / 2


def get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr):
    """Linear warmup + cosine decay LR schedule."""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr):
    """Set learning rate for all param groups."""
    for g in optimizer.param_groups:
        g["lr"] = lr


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch, ema_momentum, grad_clip):
    model.train()
    total_loss_accum = 0.0
    pred_loss_accum = 0.0
    var_loss_accum = 0.0
    cov_loss_accum = 0.0
    num_batches = 0

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            total_loss, pred_loss, var_loss, cov_loss, target_std = model(x)

        scaler.scale(total_loss).backward()

        # Gradient clipping (unscale first for accurate norm computation)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # EMA update for target encoder
        model.update_target_encoder(momentum=ema_momentum)

        total_loss_accum += total_loss.item()
        pred_loss_accum += pred_loss.item()
        var_loss_accum += var_loss.item()
        cov_loss_accum += cov_loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch} | Batch {batch_idx} | "
                f"Total: {total_loss.item():.4f} | Pred: {pred_loss.item():.4f} | "
                f"Var: {var_loss.item():.4f} | Cov: {cov_loss.item():.4f} | "
                f"TargetStd: {target_std:.4f} | LR: {current_lr:.2e} | "
                f"EMA_m: {ema_momentum:.4f}",
                flush=True,
            )
            if wandb and wandb.run:
                wandb.log({
                    "train/batch_total_loss": total_loss.item(),
                    "train/batch_pred_loss": pred_loss.item(),
                    "train/batch_var_loss": var_loss.item(),
                    "train/batch_cov_loss": cov_loss.item(),
                    "train/target_std": target_std,
                    "train/lr": current_lr,
                    "train/ema_momentum": ema_momentum,
                    "epoch": epoch,
                    "batch": batch_idx,
                })

    n = max(num_batches, 1)
    return (
        total_loss_accum / n,
        pred_loss_accum / n,
        var_loss_accum / n,
        cov_loss_accum / n,
    )


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss_accum = 0.0
    num_batches = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda"):
            total_loss, pred_loss, var_loss, cov_loss, target_std = model(x)
        total_loss_accum += total_loss.item()
        num_batches += 1

    return total_loss_accum / max(num_batches, 1)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Expand shell variables
    for key in ["data_dir", "checkpoint_dir"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expandvars(cfg[key])

    # Experiment directory
    experiment_name = cfg.get("experiment_name", "jepa_v2")
    checkpoint_dir = os.path.join(
        cfg.get("checkpoint_dir", "checkpoints"), experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_every = cfg.get("checkpoint_every", 5)

    print("=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Checkpoints → {checkpoint_dir}")
    print("=" * 60)

    set_seed(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    from torch.utils.data import DataLoader, random_split

    train_ds = ActiveMatterDataset(cfg["data_dir"], split="train",
                                    n_frames=cfg.get("n_frames", 32))
    try:
        val_ds = ActiveMatterDataset(cfg["data_dir"], split="val",
                                      n_frames=cfg.get("n_frames", 32))
    except FileNotFoundError:
        print("Validation split not found — holding out 10% of training samples")
        n_val = max(1, int(0.1 * len(train_ds)))
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds,
            [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    encoder = SpatioTemporalViT(
        in_channels=cfg.get("in_channels", 11),
        patch_size=tuple(cfg.get("patch_size", [2, 16, 16])),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("encoder_depth", 6),
        num_heads=cfg.get("num_heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
        drop_rate=cfg.get("drop_rate", 0.0),
    )

    model = VideoJEPAv2(
        encoder=encoder,
        predictor_dim=cfg.get("predictor_dim", 256),
        predictor_depth=cfg.get("predictor_depth", 3),
        predictor_heads=cfg.get("predictor_heads", 4),
        context_frames=cfg.get("context_frames", 16),
        var_weight=cfg.get("var_weight", 5.0),
        cov_weight=cfg.get("cov_weight", 0.04),
        var_gamma=cfg.get("var_gamma", 1.0),
    )
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Total parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    assert param_count < 100_000_000, f"Model exceeds 100M params: {param_count:,}"

    # Optimizer (no scheduler — we manually set LR each epoch)
    base_lr = cfg.get("lr", 1.5e-4)
    min_lr = cfg.get("min_lr", 1e-6)
    warmup_epochs = cfg.get("warmup_epochs", 10)
    total_epochs = cfg.get("epochs", 100)
    grad_clip = cfg.get("grad_clip", 1.0)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=cfg.get("weight_decay", 0.05),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
    )
    scaler = GradScaler("cuda")

    # EMA momentum schedule params
    ema_start = cfg.get("ema_momentum_start", 0.996)
    ema_end = cfg.get("ema_momentum_end", 0.999)

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from epoch {start_epoch}")
    else:
        auto_ckpt = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(auto_ckpt):
            start_epoch = load_checkpoint(auto_ckpt, model, optimizer, scaler)
            print(f"Auto-resumed from epoch {start_epoch}")

    # W&B
    use_wandb = wandb is not None
    if use_wandb:
        try:
            wandb.init(
                project=cfg.get("wandb_project", "dl-active-matter"),
                config=cfg,
                name=experiment_name,
                resume="allow",
            )
            wandb.log({"param_count": param_count})
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")
            use_wandb = False

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, total_epochs):
        # --- Set LR for this epoch (cosine with warmup) ---
        current_lr = get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr)
        set_lr(optimizer, current_lr)

        # --- EMA momentum for this epoch (cosine warmup) ---
        ema_momentum = get_ema_momentum(epoch, total_epochs, ema_start, ema_end)

        # --- Train ---
        train_total, train_pred, train_var, train_cov = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, ema_momentum, grad_clip
        )
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch} | Train: {train_total:.4f} "
            f"(pred={train_pred:.4f}, var={train_var:.4f}, cov={train_cov:.4f}) | "
            f"Val: {val_loss:.4f} | LR: {current_lr:.2e} | EMA: {ema_momentum:.4f}",
            flush=True,
        )
        if use_wandb and wandb.run:
            wandb.log({
                "train/loss": train_total,
                "train/pred_loss": train_pred,
                "train/var_loss": train_var,
                "train/cov_loss": train_cov,
                "val/loss": val_loss,
                "epoch": epoch,
            })

        # Save latest
        ckpt_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_total,
            "val_loss": val_loss,
            "config": cfg,
            "experiment_name": experiment_name,
        }
        save_checkpoint(ckpt_state, checkpoint_dir, "latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
                "experiment_name": experiment_name,
            }, checkpoint_dir, "best.pt")
            print(f"  -> New best val loss: {val_loss:.4f}")

        # Periodic checkpoint (every 5 epochs for finer analysis)
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ckpt_state, checkpoint_dir, f"epoch_{epoch+1}.pt")
            print(f"  -> Saved periodic checkpoint: epoch_{epoch+1}.pt")

    if use_wandb and wandb.run:
        wandb.finish()
    print(f"Training complete. Checkpoints in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
