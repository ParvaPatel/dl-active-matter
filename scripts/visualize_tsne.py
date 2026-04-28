"""Generate t-SNE / UMAP visualization of frozen encoder features.

Runs on a checkpoint, extracts features for all test samples, and
generates a 2D scatter plot colored by α and ζ.

Usage:
  python scripts/visualize_tsne.py \
    --checkpoint /scratch/$USER/checkpoints/jepa_small/epoch_80.pt \
    --output figures/tsne_jepa_v1_ep80.png \
    --title "JEPA v1 (epoch 80)"
"""

import os
import argparse
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
except ImportError:
    print("matplotlib not installed.")
    raise

from data.dataset import ActiveMatterDataset
from models.encoder import SpatioTemporalViT, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="figures/tsne.png")
    parser.add_argument("--title", type=str, default="t-SNE of Learned Features")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--use_target_encoder", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def extract_features(encoder, dataloader, device, context_frames=16):
    encoder.eval()
    all_features, all_alpha, all_zeta = [], [], []

    for x, labels in dataloader:
        x = x.to(device, non_blocking=True)
        if x.shape[1] > context_frames:
            x = x[:, :context_frames]
        with autocast("cuda"):
            features = encoder(x)
            features = encoder.mean_pool(features)
        all_features.append(features.cpu())
        all_alpha.append(labels["alpha"])
        all_zeta.append(labels["zeta"])

    return (torch.cat(all_features).numpy(),
            torch.cat(all_alpha).numpy(),
            torch.cat(all_zeta).numpy())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    data_dir = os.path.expandvars(cfg.get("data_dir", ""))

    encoder = SpatioTemporalViT(
        in_channels=cfg.get("in_channels", 11),
        patch_size=tuple(cfg.get("patch_size", [2, 16, 16])),
        embed_dim=cfg.get("embed_dim", 384),
        depth=cfg.get("encoder_depth", 6),
        num_heads=cfg.get("num_heads", 6),
        mlp_ratio=cfg.get("mlp_ratio", 4.0),
    )

    state_dict = ckpt["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    key = "target_encoder" if args.use_target_encoder else "encoder"
    sub_sd = {k[len(key)+1:]: v for k, v in sd.items() if k.startswith(key + ".")}
    encoder.load_state_dict(sub_sd if sub_sd else sd)
    encoder = encoder.to(device).eval()

    n_frames = cfg.get("n_frames", 16)
    context_frames = cfg.get("context_frames", 16)
    ds = ActiveMatterDataset(data_dir, split=args.split, n_frames=n_frames)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    print("Extracting features...")
    features, alpha, zeta = extract_features(encoder, loader, device, context_frames)
    print(f"Features shape: {features.shape}")

    # Standardize features before t-SNE
    features = StandardScaler().fit_transform(features)

    # Run t-SNE
    print(f"Running t-SNE (perplexity={args.perplexity})...")
    tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42,
                n_iter=1000, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(features)

    # Plot: 2 panels — colored by α, colored by ζ
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    sc1 = ax1.scatter(embedded[:, 0], embedded[:, 1], c=alpha, cmap="viridis",
                      s=8, alpha=0.6, rasterized=True)
    ax1.set_title("Colored by α (alignment)")
    ax1.set_xlabel("t-SNE 1")
    ax1.set_ylabel("t-SNE 2")
    plt.colorbar(sc1, ax=ax1, label="α")

    sc2 = ax2.scatter(embedded[:, 0], embedded[:, 1], c=zeta, cmap="plasma",
                      s=8, alpha=0.6, rasterized=True)
    ax2.set_title("Colored by ζ (activity)")
    ax2.set_xlabel("t-SNE 1")
    ax2.set_ylabel("t-SNE 2")
    plt.colorbar(sc2, ax=ax2, label="ζ")

    fig.suptitle(args.title, fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
