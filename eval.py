"""Linear probe + kNN evaluation on frozen encoder."""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
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
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=10, help="k for kNN")
    parser.add_argument("--use_target_encoder", action="store_true",
                        help="Use EMA target encoder instead of online encoder. "
                             "In JEPA/BYOL, the target (EMA) encoder is the recommended "
                             "representation network for downstream evaluation.")
    return parser.parse_args()


@torch.no_grad()
def extract_features(encoder, dataloader, device, context_frames=16):
    """Extract mean-pooled features from frozen encoder."""
    encoder.eval()
    all_features = []
    all_alpha = []
    all_zeta = []

    for batch_idx, (x, labels) in enumerate(dataloader):
        x = x.to(device, non_blocking=True)
        
        # If the sample has more frames than the encoder expects (e.g. JEPA uses 32, wants 16)
        if x.shape[1] > context_frames:
            x = x[:, :context_frames]

        with autocast("cuda"):
            features = encoder(x)              # (B, N, D)
            features = encoder.mean_pool(features)  # (B, D)
        all_features.append(features.cpu())
        all_alpha.append(labels["alpha"])
        all_zeta.append(labels["zeta"])

        if batch_idx % 50 == 0:
            print(f"    Processed batch {batch_idx}/{len(dataloader)}", flush=True)

    features = torch.cat(all_features, dim=0).numpy()
    alpha = torch.cat(all_alpha, dim=0).numpy()
    zeta = torch.cat(all_zeta, dim=0).numpy()
    return features, alpha, zeta


def linear_probe(train_features, train_targets, val_features, val_targets,
                 embed_dim, epochs=200, lr=1e-3, device="cuda"):
    """Train a single linear layer on frozen features."""
    probe = nn.Linear(embed_dim, 2).to(device)  # Predict [alpha, zeta]
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    # Convert to tensors
    X_train = torch.tensor(train_features, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_targets, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_targets, dtype=torch.float32).to(device)

    best_mse = float("inf")
    best_state = None
    for epoch in range(epochs):
        probe.train()
        pred = probe(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        probe.eval()
        with torch.no_grad():
            val_pred = probe(X_val)
            val_mse = criterion(val_pred, y_val).item()

        if val_mse < best_mse:
            best_mse = val_mse
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if epoch % 10 == 0:
            print(f"  Probe epoch {epoch} | Train MSE: {loss.item():.4f} | Val MSE: {val_mse:.4f}")

    # Reload best weights for per-target MSE breakdown
    probe.load_state_dict(best_state)
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

    # Reproducibility
    from utils.training import set_seed
    set_seed(42)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_dir = os.path.expandvars(args.data_dir or cfg["data_dir"])

    # Rebuild encoder
    encoder = SpatioTemporalViT(
        in_channels=cfg.get("in_channels", 11),
        patch_size=tuple(cfg.get("patch_size", [2, 16, 16])),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("encoder_depth", 6),
        num_heads=cfg.get("num_heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    )

    # Load encoder weights — handle 3 checkpoint formats:
    #   1. Plain keys:             "pos_embed", "blocks.0.*"  — VideoMAE / raw encoder
    #   2. JEPA (no compile):      "encoder.*", "target_encoder.*", "predictor.*"
    #   3. JEPA + torch.compile:   "_orig_mod.encoder.*" — torch.compile wraps the whole
    #                              model and prepends "_orig_mod." to every key.
    #
    # JEPA has two encoders:
    #   - online encoder ("encoder.*")        — trained to predict, not ideal for eval
    #   - target encoder ("target_encoder.*") — EMA of online, recommended for downstream eval
    state_dict = ckpt["model_state_dict"]
    encoder_key = "target_encoder" if args.use_target_encoder else "encoder"

    def extract_encoder_state(sd, key="encoder"):
        """Strip _orig_mod. prefix and extract named sub-module weights."""
        # Remove _orig_mod. prefix added by torch.compile
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        # Extract the requested sub-module (encoder or target_encoder)
        sub_sd = {
            k[len(key) + 1:]: v
            for k, v in sd.items()
            if k.startswith(key + ".")
        }
        if sub_sd:
            return sub_sd
        # Fallback: state dict IS the encoder (VideoMAE / raw encoder ckpt)
        return sd

    encoder_state = extract_encoder_state(state_dict, key=encoder_key)
    enc_label = "target (EMA)" if args.use_target_encoder else "online"
    print(f"Loading {enc_label} encoder weights ({len(encoder_state)} keys)")

    encoder.load_state_dict(encoder_state)
    encoder = encoder.to(device)
    encoder.eval()

    print(f"Encoder parameters: {count_parameters(encoder):,}")

    # Data
    n_frames_total = cfg.get("n_frames", 16)
    context_frames_eval = cfg.get("context_frames", 16)

    train_ds = ActiveMatterDataset(data_dir, split="train", n_frames=n_frames_total)
    eval_ds = ActiveMatterDataset(data_dir, split=args.split, n_frames=n_frames_total)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Extract features
    print("Extracting train features...")
    train_feat, train_alpha, train_zeta = extract_features(encoder, train_loader, device, context_frames_eval)
    print("Extracting eval features...")
    eval_feat, eval_alpha, eval_zeta = extract_features(encoder, eval_loader, device, context_frames_eval)

    # Feature statistics (for debugging scale drift)
    print(f"\nFeature stats (raw):")
    print(f"  Train — mean: {train_feat.mean():.4f}, std: {train_feat.std():.4f}, "
          f"min: {train_feat.min():.4f}, max: {train_feat.max():.4f}")
    print(f"  Eval  — mean: {eval_feat.mean():.4f}, std: {eval_feat.std():.4f}, "
          f"min: {eval_feat.min():.4f}, max: {eval_feat.max():.4f}")

    # Z-score normalize features (critical for scale-invariant probing)
    feat_scaler = StandardScaler()
    train_feat = feat_scaler.fit_transform(train_feat)
    eval_feat = feat_scaler.transform(eval_feat)

    # Z-score normalize targets
    target_scaler = StandardScaler()
    train_targets = target_scaler.fit_transform(np.stack([train_alpha, train_zeta], axis=1))
    eval_targets = target_scaler.transform(np.stack([eval_alpha, eval_zeta], axis=1))

    embed_dim = cfg.get("embed_dim", 384)

    # === Primary Linear Probe (default LR) ===
    print("\n=== Linear Probe ===")
    lp_mse, lp_alpha, lp_zeta = linear_probe(
        train_feat, train_targets, eval_feat, eval_targets,
        embed_dim, epochs=args.probe_epochs, lr=args.probe_lr, device=device,
    )
    print(f"Linear Probe — Total MSE: {lp_mse:.4f} | α MSE: {lp_alpha:.4f} | ζ MSE: {lp_zeta:.4f}")

    # === Primary kNN (default k) ===
    print(f"\n=== kNN (k={args.k}) ===")
    knn_mse, knn_alpha, knn_zeta = knn_evaluate(
        train_feat, train_targets, eval_feat, eval_targets, k=args.k,
    )
    print(f"kNN — Total MSE: {knn_mse:.4f} | α MSE: {knn_alpha:.4f} | ζ MSE: {knn_zeta:.4f}")

    # === kNN k-sweep (for ablation) ===
    k_values = [1, 3, 5, 10, 20, 50]
    print(f"\n=== kNN k-sweep ===")
    print(f"{'k':<6} {'Total MSE':<12} {'α MSE':<12} {'ζ MSE':<12}")
    best_knn_mse, best_knn_alpha, best_knn_zeta, best_k = knn_mse, knn_alpha, knn_zeta, args.k
    for k_val in k_values:
        k_mse, k_alpha, k_zeta = knn_evaluate(
            train_feat, train_targets, eval_feat, eval_targets, k=k_val,
        )
        print(f"{k_val:<6} {k_mse:<12.4f} {k_alpha:<12.4f} {k_zeta:<12.4f}")
        if k_mse < best_knn_mse:
            best_knn_mse, best_knn_alpha, best_knn_zeta, best_k = k_mse, k_alpha, k_zeta, k_val

    # === Linear Probe LR sweep (for ablation) ===
    lr_values = [1e-2, 1e-3, 1e-4]
    print(f"\n=== Linear Probe LR sweep (epochs={args.probe_epochs}) ===")
    print(f"{'LR':<12} {'Total MSE':<12} {'α MSE':<12} {'ζ MSE':<12}")
    best_lp_mse, best_lp_alpha, best_lp_zeta, best_lr = lp_mse, lp_alpha, lp_zeta, args.probe_lr
    for lr_val in lr_values:
        lr_mse, lr_alpha, lr_zeta = linear_probe(
            train_feat, train_targets, eval_feat, eval_targets,
            embed_dim, epochs=args.probe_epochs, lr=lr_val, device=device,
        )
        print(f"{lr_val:<12.0e} {lr_mse:<12.4f} {lr_alpha:<12.4f} {lr_zeta:<12.4f}")
        if lr_mse < best_lp_mse:
            best_lp_mse, best_lp_alpha, best_lp_zeta, best_lr = lr_mse, lr_alpha, lr_zeta, lr_val

    # Summary
    print("\n" + "=" * 50)
    print("RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Method':<20} {'Total MSE':<12} {'α MSE':<12} {'ζ MSE':<12}")
    print(f"{'LP (lr=1e-3)':<20} {lp_mse:<12.4f} {lp_alpha:<12.4f} {lp_zeta:<12.4f}")
    knn_label = f"kNN (k={args.k})"
    print(f"{knn_label:<20} {knn_mse:<12.4f} {knn_alpha:<12.4f} {knn_zeta:<12.4f}")
    print("")
    print("★ BEST RESULTS (from sweeps)")
    print(f"{'Best LP (lr='+f'{best_lr:.0e})':<20} {best_lp_mse:<12.4f} {best_lp_alpha:<12.4f} {best_lp_zeta:<12.4f}")
    print(f"{'Best kNN (k='+f'{best_k})':<20} {best_knn_mse:<12.4f} {best_knn_alpha:<12.4f} {best_knn_zeta:<12.4f}")


if __name__ == "__main__":
    main()
