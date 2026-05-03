"""Visualize sample frames from the active matter dataset.

Generates a figure showing what the raw simulation data looks like
at different physical parameters (alpha, zeta).

Usage (requires GPU node with dataset access):
    python scripts/visualize_dataset.py --output report/fig_dataset.png
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.dataset import ActiveMatterDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="report/fig_dataset.png")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.expandvars("/scratch/$USER/active_matter"))
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    ds = ActiveMatterDataset(args.data_dir, split=args.split, n_frames=16)
    print(f"Dataset: {len(ds)} samples")

    # Collect samples at diverse (alpha, zeta) values
    # Sample indices spread across the dataset
    n_show = 4
    indices = np.linspace(0, len(ds) - 1, n_show * 3, dtype=int)
    
    # Find samples with diverse parameters
    samples = []
    seen_params = set()
    for idx in indices:
        x, labels = ds[int(idx)]
        alpha = labels["alpha"].item()
        zeta = labels["zeta"].item()
        key = (round(alpha, 1), round(zeta, 0))
        if key not in seen_params and len(samples) < n_show:
            seen_params.add(key)
            samples.append((x, alpha, zeta))

    # If we didn't get enough diverse samples, just take first n_show
    if len(samples) < n_show:
        for idx in range(min(n_show, len(ds))):
            x, labels = ds[idx]
            samples.append((x, labels["alpha"].item(), labels["zeta"].item()))
            if len(samples) >= n_show:
                break

    # Channel names for the 11-channel input
    channel_names = [
        "Concentration",
        "Velocity (x)", "Velocity (y)",
        "Orientation Q11", "Orientation Q12",
        "Orientation Q21", "Orientation Q22",
        "Strain S11", "Strain S12",
        "Strain S21", "Strain S22",
    ]

    # ---- Figure 1: Different physical regimes (concentration channel) ----
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))

    for i, (x, alpha, zeta) in enumerate(samples[:n_show]):
        # Show first frame, concentration channel (channel 0)
        frame_t0 = x[0, 0].numpy()  # shape: (H, W)
        frame_t8 = x[min(8, x.shape[0]-1), 0].numpy()

        axes[0, i].imshow(frame_t0, cmap="inferno", aspect="equal")
        axes[0, i].set_title(f"t=0\nalpha={alpha:.1f}, zeta={zeta:.0f}", fontsize=10)
        axes[0, i].axis("off")

        axes[1, i].imshow(frame_t8, cmap="inferno", aspect="equal")
        axes[1, i].set_title(f"t=8", fontsize=10)
        axes[1, i].axis("off")

    fig.suptitle("Active Matter Simulations: Concentration Field at Different Parameters",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    out_path = args.output
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

    # ---- Figure 2: All 11 channels for one sample ----
    x, alpha, zeta = samples[0]
    n_channels = min(x.shape[1], 11)
    cols = 4
    rows = (n_channels + cols - 1) // cols

    fig2, axes2 = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes2 = axes2.flatten()

    for c in range(n_channels):
        frame = x[0, c].numpy()
        axes2[c].imshow(frame, cmap="viridis", aspect="equal")
        axes2[c].set_title(channel_names[c] if c < len(channel_names) else f"Ch {c}",
                          fontsize=9)
        axes2[c].axis("off")

    # Hide unused axes
    for c in range(n_channels, len(axes2)):
        axes2[c].axis("off")

    fig2.suptitle(f"All Input Channels (alpha={alpha:.1f}, zeta={zeta:.0f})",
                  fontsize=14, fontweight="bold")
    fig2.tight_layout()

    out_path2 = out_path.replace(".png", "_channels.png")
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
