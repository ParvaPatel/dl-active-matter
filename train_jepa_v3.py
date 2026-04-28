"""Video-JEPA v3 training entry point.

Combines v1's meaningful prediction loss with v2's training stabilizers,
plus performance optimizations:
    - Cosine LR schedule with linear warmup (from v2)
    - EMA momentum cosine warmup 0.996 → 0.999 (from v2)
    - Gradient clipping (from v2)
    - Raw MSE prediction loss (from v1 — the critical fix)
    - Full VICReg logging (variance + covariance losses)
    - [NEW] Early stopping (patience-based on val_loss)
    - [NEW] Gradient accumulation (larger effective batch without OOM)
    - [NEW] torch.compile() for kernel fusion speedup
    - [NEW] persistent_workers for faster DataLoader restarts
    - [NEW] GPU memory logging
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
from models.jepa_v3 import VideoJEPAv3
from utils.training import set_seed, save_checkpoint, load_checkpoint, log_gpu_memory


def parse_args():
    parser = argparse.ArgumentParser(description="Train Video-JEPA v3 on active_matter")
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


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch,
                    ema_momentum, grad_clip, grad_accum_steps=1):
    model.train()
    total_loss_accum = 0.0
    pred_loss_accum = 0.0
    var_loss_accum = 0.0
    cov_loss_accum = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)

        with autocast("cuda"):
            total_loss, pred_loss, var_loss, cov_loss, target_std = model(x)
            # Scale loss by accumulation steps for correct gradient magnitude
            scaled_loss = total_loss / grad_accum_steps

        # .item() called OUTSIDE autocast/compiled region to avoid torch.compile graph break
        target_std_val = target_std.item()

        scaler.scale(scaled_loss).backward()

        # Step optimizer every grad_accum_steps batches (or at end of epoch)
        if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            # Gradient clipping (unscale first for accurate norm computation)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # EMA update for target encoder (once per optimizer step, not per batch)
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
                f"TargetStd: {target_std_val:.4f} | LR: {current_lr:.2e} | "
                f"EMA_m: {ema_momentum:.4f}",
                flush=True,
            )
            if wandb and wandb.run:
                wandb.log({
                    "train/batch_total_loss": total_loss.item(),
                    "train/batch_pred_loss": pred_loss.item(),
                    "train/batch_var_loss": var_loss.item(),
                    "train/batch_cov_loss": cov_loss.item(),
                    "train/target_std": target_std_val,
                    "train/lr": current_lr,
                    "train/ema_momentum": ema_momentum,
                    "epoch": epoch,
                    "batch": batch_idx,
                })


        # Log GPU memory after first batch (once per training run)
        if epoch == 0 and batch_idx == 0:
            log_gpu_memory(tag="(after first batch)")

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
    experiment_name = cfg.get("experiment_name", "jepa_v3")
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

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,  # Keep workers alive across epochs
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
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

    model = VideoJEPAv3(
        encoder=encoder,
        predictor_dim=cfg.get("predictor_dim", 256),
        predictor_depth=cfg.get("predictor_depth", 3),
        predictor_heads=cfg.get("predictor_heads", 4),
        context_frames=cfg.get("context_frames", 16),
        var_weight=cfg.get("var_weight", 1.0),
        cov_weight=cfg.get("cov_weight", 0.01),
        var_gamma=cfg.get("var_gamma", 1.0),
    )
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Total parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    assert param_count < 100_000_000, f"Model exceeds 100M params: {param_count:,}"

    # torch.compile() — fuses operations for 10-30% speedup (PyTorch 2.x)
    use_compile = cfg.get("compile", True)
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)
        print("  ✓ Model compiled")
    elif use_compile:
        print("torch.compile() not available (requires PyTorch 2.x), skipping")

    # Optimizer (no scheduler — we manually set LR each epoch)
    base_lr = float(cfg.get("lr", 1.5e-4))
    min_lr = float(cfg.get("min_lr", 1e-6))
    warmup_epochs = cfg.get("warmup_epochs", 10)
    total_epochs = cfg.get("epochs", 100)
    grad_clip = cfg.get("grad_clip", 1.0)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    early_stopping_patience = cfg.get("early_stopping_patience", 15)
    # Don't start early stopping until past warmup + a grace period.
    # Warmup causes val loss to temporarily worsen (LR is ramping up) —
    # counting those epochs as "no improvement" fires early stopping prematurely.
    early_stopping_start_epoch = cfg.get("early_stopping_start_epoch", warmup_epochs + 5)

    effective_batch = cfg["batch_size"] * grad_accum_steps
    print(f"Batch size: {cfg['batch_size']} × {grad_accum_steps} accum = {effective_batch} effective")
    print(f"Early stopping: patience={early_stopping_patience} epochs (active from epoch {early_stopping_start_epoch})")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(base_lr),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
        betas=tuple(float(b) for b in cfg.get("betas", [0.9, 0.95])),
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
    best_val_loss = float("inf")   # global best → controls best.pt
    es_best_val_loss = float("inf")  # post-warmup best → controls early stopping
    # NOTE: es_best_val_loss is intentionally NOT saved/loaded from checkpoint.
    # On any resume it resets to inf, so the first post-warmup epoch becomes the
    # new baseline. This correctly handles warmup-phase val_loss artifacts.
    epochs_without_improvement = 0

    for epoch in range(start_epoch, total_epochs):
        # --- Set LR for this epoch (cosine with warmup) ---
        current_lr = get_lr(epoch, total_epochs, warmup_epochs, base_lr, min_lr)
        set_lr(optimizer, current_lr)

        # --- EMA momentum for this epoch (cosine warmup) ---
        ema_momentum = get_ema_momentum(epoch, total_epochs, ema_start, ema_end)

        # --- Train ---
        train_total, train_pred, train_var, train_cov = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            ema_momentum, grad_clip, grad_accum_steps
        )
        val_loss = validate(model, val_loader, device)

        in_warmup = epoch < early_stopping_start_epoch
        print(
            f"Epoch {epoch} | Train: {train_total:.4f} "
            f"(pred={train_pred:.4f}, var={train_var:.4f}, cov={train_cov:.4f}) | "
            f"Val: {val_loss:.4f} | LR: {current_lr:.2e} | EMA: {ema_momentum:.4f}"
            + (" [warmup]" if in_warmup else ""),
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

        # --- Global best: saves best.pt (all epochs, including warmup) ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
                "experiment_name": experiment_name,
            }, checkpoint_dir, "best.pt")
            print(f"  -> New global best val loss: {val_loss:.4f} (saved best.pt)")

        # --- Early stopping: tracks improvement only AFTER warmup ---
        # Uses es_best_val_loss (separate from best_val_loss) so warmup-phase
        # artifacts (e.g., low val at epoch 0 due to tiny warmup LR) don't
        # prevent the patience counter from ever resetting during real training.
        if not in_warmup:
            if val_loss < es_best_val_loss:
                es_best_val_loss = val_loss
                epochs_without_improvement = 0
                print(f"  -> New post-warmup best: {val_loss:.4f} (early stopping counter reset)")
            else:
                epochs_without_improvement += 1
                print(f"  -> No improvement for {epochs_without_improvement}/{early_stopping_patience} epochs "
                      f"(post-warmup best: {es_best_val_loss:.4f})")
        else:
            print(f"  -> Warmup phase (epoch {epoch+1}/{early_stopping_start_epoch}) — early stopping not active")

        # Periodic checkpoint (every 5 epochs for finer analysis)
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ckpt_state, checkpoint_dir, f"epoch_{epoch+1}.pt")
            print(f"  -> Saved periodic checkpoint: epoch_{epoch+1}.pt")

        # Early stopping check — only fires after warmup+grace period
        if not in_warmup and epochs_without_improvement >= early_stopping_patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING at epoch {epoch} — no post-warmup improvement for {early_stopping_patience} epochs")
            print(f"Global best val: {best_val_loss:.4f} | Post-warmup best: {es_best_val_loss:.4f}")
            print(f"{'='*60}")
            break

    if use_wandb and wandb.run:
        wandb.finish()
    print(f"Training complete. Checkpoints in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
