"""Supervised end-to-end baseline: ViT encoder + linear head → predict (α, ζ).

This serves as the UPPER BOUND on representation quality — a self-supervised
model cannot outperform a supervised model trained directly on the labels.

Architecture: Same SpatioTemporalViT encoder as JEPA/VideoMAE, with a
single linear layer (mean-pooled features → 2 outputs).
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
from utils.training import set_seed, save_checkpoint, load_checkpoint


class SupervisedViT(nn.Module):
    """ViT encoder + linear head for direct regression of physical parameters."""

    def __init__(self, encoder: SpatioTemporalViT, num_targets=2):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embed_dim, num_targets)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        features = self.encoder(x)           # (B, N, D)
        pooled = self.encoder.mean_pool(features)  # (B, D)
        pred = self.head(pooled)             # (B, 2)
        return pred, pooled


def parse_args():
    parser = argparse.ArgumentParser(description="Train supervised baseline")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scaler, criterion, device,
                    epoch, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        targets = torch.stack([labels["alpha"], labels["zeta"]], dim=1).to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda"):
            pred, _ = model(x)
            loss = criterion(pred, targets)

        scaler.scale(loss).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(dataloader)} | "
                  f"Loss: {loss.item():.4f}", flush=True)

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_pred, all_targets = [], []
    num_batches = 0

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        targets = torch.stack([labels["alpha"], labels["zeta"]], dim=1).to(device)

        with autocast("cuda"):
            pred, _ = model(x)
            loss = criterion(pred, targets)

        total_loss += loss.item()
        all_pred.append(pred.cpu())
        all_targets.append(targets.cpu())
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    # Per-target MSE breakdown
    preds = torch.cat(all_pred)
    tgts = torch.cat(all_targets)
    mse_alpha = ((preds[:, 0] - tgts[:, 0]) ** 2).mean().item()
    mse_zeta = ((preds[:, 1] - tgts[:, 1]) ** 2).mean().item()

    return avg_loss, mse_alpha, mse_zeta


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for key in ["data_dir", "checkpoint_dir"]:
        if key in cfg and isinstance(cfg[key], str):
            cfg[key] = os.path.expandvars(cfg[key])

    experiment_name = cfg.get("experiment_name", "supervised")
    checkpoint_dir = os.path.join(cfg.get("checkpoint_dir", "checkpoints"), experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_every = cfg.get("checkpoint_every", 5)

    print("=" * 60)
    print(f"EXPERIMENT: {experiment_name} (supervised baseline)")
    print(f"Checkpoints → {checkpoint_dir}")
    print("=" * 60)

    set_seed(cfg.get("seed", 42))
    cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    from torch.utils.data import DataLoader, random_split

    n_frames = cfg.get("n_frames", 16)
    train_ds = ActiveMatterDataset(cfg["data_dir"], split="train", n_frames=n_frames)

    try:
        val_ds = ActiveMatterDataset(cfg["data_dir"], split="val", n_frames=n_frames)
    except FileNotFoundError:
        print("Validation split not found — holding out 10%")
        n_val = max(1, int(0.1 * len(train_ds)))
        train_ds, val_ds = random_split(
            train_ds, [len(train_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(cfg.get("seed", 42)),
        )

    num_workers = cfg.get("num_workers", 4)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, pin_memory=True,
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

    model = SupervisedViT(encoder, num_targets=2)
    model = model.to(device)

    total_params = count_parameters(model)
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    assert total_params < 100_000_000, f"Model exceeds 100M params: {total_params:,}"

    # torch.compile
    if cfg.get("compile", True):
        try:
            model = torch.compile(model)
            print("torch.compile() enabled")
        except Exception as e:
            print(f"torch.compile() failed ({e}), continuing without")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 0.05)),
        betas=tuple(float(b) for b in cfg.get("betas", [0.9, 0.95])),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.get("epochs", 100), eta_min=1e-6
    )

    criterion = nn.MSELoss()
    scaler = GradScaler("cuda")
    grad_clip = cfg.get("grad_clip", 1.0)

    # Resume
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
            wandb.init(project=cfg.get("wandb_project", "dl-active-matter"),
                       config=cfg, name=experiment_name, resume="allow")
        except Exception as e:
            print(f"W&B init failed ({e})")
            use_wandb = False

    # Training loop
    best_val_loss = float("inf")

    for epoch in range(start_epoch, cfg.get("epochs", 100)):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion, device,
            epoch, grad_clip
        )
        val_loss, val_alpha, val_zeta = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} | Train: {train_loss:.4f} | Val: {val_loss:.4f} "
              f"(α={val_alpha:.4f}, ζ={val_zeta:.4f}) | LR: {lr:.2e}", flush=True)

        if use_wandb and wandb.run:
            wandb.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "val/mse_alpha": val_alpha,
                "val/mse_zeta": val_zeta,
                "lr": lr,
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

        if (epoch + 1) % checkpoint_every == 0:
            save_checkpoint(ckpt_state, checkpoint_dir, f"epoch_{epoch+1}.pt")
            print(f"  -> Saved checkpoint: epoch_{epoch+1}.pt")

    if use_wandb and wandb.run:
        wandb.finish()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
