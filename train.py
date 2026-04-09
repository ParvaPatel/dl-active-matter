"""VideoMAE training entry point."""

import os
import argparse
import yaml
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

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


def train_one_epoch(model, dataloader, optimizer, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda"):
            loss, pred, mask = model(x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")
            if wandb and wandb.run:
                wandb.log({"train/batch_loss": loss.item(), "epoch": epoch, "batch": batch_idx})

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
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

    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    from torch.utils.data import DataLoader, random_split

    train_ds = ActiveMatterDataset(cfg["data_dir"], split="train")

    # Try to load a dedicated validation split; if missing, carve from train
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

    model = VideoMAE(
        encoder=encoder,
        decoder_dim=cfg.get("decoder_dim", 192),
        decoder_depth=cfg.get("decoder_depth", 2),
        decoder_heads=cfg.get("decoder_heads", 3),
        mask_ratio=cfg.get("mask_ratio", 0.75),
    )
    model = model.to(device)

    param_count = count_parameters(model)
    print(f"Total parameters: {param_count:,} ({param_count/1e6:.2f}M)")
    assert param_count < 100_000_000, f"Model exceeds 100M params: {param_count:,}"

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1.5e-4),
        weight_decay=cfg.get("weight_decay", 0.05),
        betas=tuple(cfg.get("betas", [0.9, 0.95])),
    )

    scaler = GradScaler()

    # Resume from checkpoint
    start_epoch = 0
    checkpoint_dir = cfg.get("checkpoint_dir", "checkpoints")
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)
        print(f"Resumed from epoch {start_epoch}")
    else:
        # Check for auto-resume (spot preemption)
        auto_ckpt = os.path.join(checkpoint_dir, "latest.pt")
        if os.path.exists(auto_ckpt):
            start_epoch = load_checkpoint(auto_ckpt, model, optimizer, scaler)
            print(f"Auto-resumed from epoch {start_epoch}")

    # W&B (optional — set WANDB_API_KEY or run `wandb login` to enable)
    use_wandb = wandb is not None
    if use_wandb:
        try:
            wandb.init(
                project=cfg.get("wandb_project", "dl-active-matter"),
                config=cfg,
                resume="allow",
            )
            wandb.log({"param_count": param_count})
        except Exception as e:
            print(f"W&B init failed ({e}), continuing without logging")
            use_wandb = False

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, cfg.get("epochs", 100)):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch)
        val_loss = validate(model, val_loader, device)

        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if use_wandb and wandb.run:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "epoch": epoch,
            })

        # Save latest (for preemption recovery)
        save_checkpoint({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": cfg,
        }, checkpoint_dir, "latest.pt")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": cfg,
                "val_loss": val_loss,
            }, checkpoint_dir, "best.pt")
            print(f"  -> New best val loss: {val_loss:.4f}")

    if use_wandb and wandb.run:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    main()
