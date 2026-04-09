"""Linear probe + kNN evaluation on frozen encoder."""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from data.dataset import ActiveMatterDataset
from models.encoder import SpatioTemporalViT, count_parameters


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate frozen encoder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to encoder checkpoint")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--data_dir", type=str, default=None, help="Override data dir from config")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--probe_epochs", type=int, default=50)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=10, help="k for kNN")
    return parser.parse_args()


@torch.no_grad()
def extract_features(encoder, dataloader, device):
    """Extract mean-pooled features from frozen encoder."""
    encoder.eval()
    all_features = []
    all_alpha = []
    all_zeta = []

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        with autocast(device_type="cuda"):
            features = encoder(x)              # (B, N, D)
            features = encoder.mean_pool(features)  # (B, D)
        all_features.append(features.cpu())
        all_alpha.append(labels["alpha"])
        all_zeta.append(labels["zeta"])

    features = torch.cat(all_features, dim=0).numpy()
    alpha = torch.cat(all_alpha, dim=0).numpy()
    zeta = torch.cat(all_zeta, dim=0).numpy()
    return features, alpha, zeta


def linear_probe(train_features, train_targets, val_features, val_targets,
                 embed_dim, epochs=50, lr=1e-3, device="cuda"):
    """Train a single linear layer on frozen features."""
    probe = nn.Linear(embed_dim, 2).to(device)  # Predict [alpha, zeta]
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_targets, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_targets, dtype=torch.float32).to(device)

    best_mse = float("inf")
    for epoch in range(epochs):
        probe.train()
        pred = probe(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val)
            val_mse = criterion(val_pred, y_val).item()

        if val_mse < best_mse:
            best_mse = val_mse

        if epoch % 10 == 0:
            print(f"  Probe epoch {epoch} | Train MSE: {loss.item():.4f} | Val MSE: {val_mse:.4f}")

    # Per-target MSE
    probe.eval()
    with torch.no_grad():
        val_pred = probe(X_val)
        mse_alpha = ((val_pred[:, 0] - y_val[:, 0]) ** 2).mean().item()
        mse_zeta = ((val_pred[:, 1] - y_val[:, 1]) ** 2).mean().item()

    return best_mse, mse_alpha, mse_zeta


def knn_evaluate(train_features, train_targets, val_features, val_targets, k=10):
    """kNN regression on frozen features."""
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance", n_jobs=-1)
    knn.fit(train_features, train_targets)
    pred = knn.predict(val_features)

    mse_total = np.mean((pred - val_targets) ** 2)
    mse_alpha = np.mean((pred[:, 0] - val_targets[:, 0]) ** 2)
    mse_zeta = np.mean((pred[:, 1] - val_targets[:, 1]) ** 2)
    return mse_total, mse_alpha, mse_zeta


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_dir = args.data_dir or cfg["data_dir"]

    # Rebuild encoder
    encoder = SpatioTemporalViT(
        in_channels=cfg.get("in_channels", 11),
        patch_size=tuple(cfg.get("patch_size", [2, 16, 16])),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("encoder_depth", 6),
        num_heads=cfg.get("num_heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    )

    # Load encoder weights (handle full MAE checkpoint)
    state_dict = ckpt["model_state_dict"]
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("encoder.")
    }
    if not encoder_state:
        encoder_state = state_dict
    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)
    encoder.eval()

    print(f"Encoder parameters: {count_parameters(encoder):,}")

    # Data
    train_ds = ActiveMatterDataset(data_dir, split="train")
    eval_ds = ActiveMatterDataset(data_dir, split=args.split)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    print("Extracting train features...")
    train_feat, train_alpha, train_zeta = extract_features(encoder, train_loader, device)
    print("Extracting eval features...")
    eval_feat, eval_alpha, eval_zeta = extract_features(encoder, eval_loader, device)

    # Z-score normalize targets
    scaler = StandardScaler()
    train_targets = scaler.fit_transform(np.stack([train_alpha, train_zeta], axis=1))
    eval_targets = scaler.transform(np.stack([eval_alpha, eval_zeta], axis=1))

    embed_dim = cfg.get("embed_dim", 384)

    # Linear probe
    print("\n=== Linear Probe ===")
    lp_mse, lp_alpha, lp_zeta = linear_probe(
        train_feat, train_targets, eval_feat, eval_targets,
        embed_dim, epochs=args.probe_epochs, lr=args.probe_lr, device=device,
    )
    print(f"Linear Probe — Total MSE: {lp_mse:.4f} | α MSE: {lp_alpha:.4f} | ζ MSE: {lp_zeta:.4f}")

    # kNN
    print(f"\n=== kNN (k={args.k}) ===")
    knn_mse, knn_alpha, knn_zeta = knn_evaluate(
        train_feat, train_targets, eval_feat, eval_targets, k=args.k,
    )
    print(f"kNN — Total MSE: {knn_mse:.4f} | α MSE: {knn_alpha:.4f} | ζ MSE: {knn_zeta:.4f}")

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Method':<15} {'Total MSE':<12} {'α MSE':<12} {'ζ MSE':<12}")
    print(f"{'Linear Probe':<15} {lp_mse:<12.4f} {lp_alpha:<12.4f} {lp_zeta:<12.4f}")
    print(f"{'kNN (k={args.k})':<15} {knn_mse:<12.4f} {knn_alpha:<12.4f} {knn_zeta:<12.4f}")


if __name__ == "__main__":
    main()
