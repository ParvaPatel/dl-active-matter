"""VideoMAE training entry point.

Optimized for speed:
  - torch.compile() for kernel fusion
  - cudnn.benchmark for auto-tuned convolutions
  - Gradient accumulation for effective batch size > physical batch size
  - persistent_workers for faster data loading
  - Gradient clipping for stability
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.amp import GradScaler, autocast

try:
    import wandb
except ImportError:
    wandb = None

from data.dataset import ActiveMatterDataset
from models.encoder import SpatioTemporalViT, count_parameters
from models.mae import VideoMAE
from utils.training import set_seed, save_checkpoint, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Train VideoMAE on active_matter")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch,
                    grad_clip=1.0, grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)

        with autocast("cuda"):
            loss, pred, mask = model(x)
            scaled_loss = loss / grad_accum_steps

        scaler.scale(scaled_loss).backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    # Handle leftover gradients
    if num_batches % grad_accum_steps != 0:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        with autocast("cuda"):
            loss, pred, mask = model(x)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Expand shell variables like ${USER} in config values
    for key in ["data_dir", "checkpoint_dir"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expandvars(cfg[key])

    # Experiment-specific checkpoint directory
    experiment_name = cfg.get("experiment_name", "default")
    checkpoint_dir = os.path.join(
        cfg.get("checkpoint_dir", "checkpoints"), experiment_name
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_every = cfg.get("checkpoint_every", 10)

    print("=" * 60)
    print(f"EXPERIMENT: {experiment_name}")
    print(f"Checkpoints → {checkpoint_dir}")
    print("=" * 60)

    set_seed(cfg.get("seed", 42))

    # Performance: auto-tune convolution algorithms
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    from torch.utils.data import DataLoader, random_split

    train_ds = ActiveMatterDataset(cfg["data_dir"], split="train")

    try:
        val_ds = ActiveMatterDataset(cfg["data_dir"], split="val")
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
        persistent_workers=num_workers > 0,
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

    model = VideoMAE(
        encoder=encoder,
        decoder_dim=cfg.get("decoder_dim", 192),
        decoder_depth=cfg.get("decoder_depth", 2),
        decoder_heads=cfg.get("decoder_heads", 3),
        mask_ratio=cfg.get("mask_ratio", 0.75),
    )
    model = model.to(device)

    encoder_params = count_parameters(encoder)
    total_params = count_parameters(model)
    print(f"Encoder parameters: {encoder_params:,} ({encoder_params/1e6:.2f}M)")
    print(f"Total parameters:   {total_params:,} ({total_params/1e6:.2f}M)")
    assert encoder_params < 100_000_000, f"Encoder exceeds 100M params: {encoder_params:,}"

    # torch.compile for kernel fusion speedup
    use_compile = cfg.get("compile", True)
    if use_compile:
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed ({e}), continuing without it")

    # Training config
    grad_clip = cfg.get("grad_clip", 1.0)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)

    effective_batch = cfg["batch_size"] * grad_accum_steps
    print(f"Batch size: {cfg['batch_size']} × {grad_accum_steps} accum = {effective_batch} effective")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 1.5e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
        betas=tuple(float(b) for b in cfg.get("betas", [0.9, 0.95])),
    )

    scaler = GradScaler("cuda")

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
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")
            use_wandb = False

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, cfg.get("epochs", 100)):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch,
            grad_clip, grad_accum_steps
        )
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
              flush=True)
        if use_wandb and wandb.run:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epoch": epoch,
            })

        # Save latest
        ckpt_state = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
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

        # Periodic checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ckpt_state, checkpoint_dir, f"epoch_{epoch+1}.pt")
            print(f"  -> Saved periodic checkpoint: epoch_{epoch+1}.pt")

    if use_wandb and wandb.run:
        wandb.finish()
    print(f"Training complete. Checkpoints in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
