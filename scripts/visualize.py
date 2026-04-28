"""Generate all figures for the report from existing evaluation data.

Usage:
  python scripts/visualize.py --output_dir figures/

Generates:
  1. feature_collapse.png    — Feature σ vs epoch for all models
  2. lp_mse_trajectory.png   — LP MSE vs epoch for all models
  3. knn_mse_trajectory.png  — kNN MSE vs epoch for all models
  4. tsne_features.png       — t-SNE of frozen features colored by α and ζ
  5. comparison_table.png    — Final comparison bar chart
"""

import os
import re
import sys
import argparse
import glob
import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


# ──────────────────────────── Parsing Utilities ────────────────────────────

def parse_eval_log(filepath):
    """Extract metrics from a single eval log file."""
    results = {}
    with open(filepath) as f:
        text = f.read()

    # Extract checkpoint epoch
    epoch_match = re.search(r"epoch_(\d+)\.pt", text)
    best_match = re.search(r"best\.pt", text)
    if epoch_match:
        results["epoch"] = int(epoch_match.group(1))
    elif best_match:
        results["epoch"] = "best"
    else:
        return None

    # Extract LP results
    lp = re.search(r"Linear Probe.*?Total MSE:\s*([\d.]+).*?α MSE:\s*([\d.]+).*?ζ MSE:\s*([\d.]+)", text, re.DOTALL)
    if lp:
        results["lp_mse"] = float(lp.group(1))
        results["lp_alpha"] = float(lp.group(2))
        results["lp_zeta"] = float(lp.group(3))

    # Extract kNN results
    knn = re.search(r"kNN.*?Total MSE:\s*([\d.]+).*?α MSE:\s*([\d.]+).*?ζ MSE:\s*([\d.]+)", text, re.DOTALL)
    if knn:
        results["knn_mse"] = float(knn.group(1))
        results["knn_alpha"] = float(knn.group(2))
        results["knn_zeta"] = float(knn.group(3))

    # Extract feature stats
    feat = re.search(r"Eval\s+.*?mean:\s*([-\d.]+).*?std:\s*([\d.]+)", text)
    if feat:
        results["feat_mean"] = float(feat.group(1))
        results["feat_std"] = float(feat.group(2))

    return results


def load_experiment(log_dir):
    """Load all eval results from a log directory."""
    logs = sorted(glob.glob(os.path.join(log_dir, "*.out")))
    results = []
    for log in logs:
        r = parse_eval_log(log)
        if r and r["epoch"] != "best" and "lp_mse" in r:
            results.append(r)
    results.sort(key=lambda x: x["epoch"])
    return results


# ──────────────────────────── Plotting ────────────────────────────

# Shared style
STYLE = {
    "figure.figsize": (10, 6),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

COLORS = {
    "VideoMAE Small": "#2196F3",
    "JEPA v1 Small": "#FF5722",
    "JEPA v3 tuned": "#4CAF50",
    "JEPA v3 strongvar": "#9C27B0",
    "Supervised Small": "#FF9800",
    "VideoMAE Base": "#0D47A1",
    "JEPA Base": "#B71C1C",
}


def plot_metric_trajectory(experiments, metric_key, ylabel, title, output_path,
                           best_marker=True):
    """Plot a metric vs epoch for multiple experiments."""
    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots()

    for name, data in experiments.items():
        if not data:
            continue
        epochs = [d["epoch"] for d in data if metric_key in d]
        values = [d[metric_key] for d in data if metric_key in d]
        if not epochs:
            continue

        color = COLORS.get(name, "#666666")
        ax.plot(epochs, values, "o-", label=name, color=color, markersize=4)

        if best_marker:
            best_idx = np.argmin(values) if "mse" in metric_key.lower() else np.argmax(values)
            ax.plot(epochs[best_idx], values[best_idx], "*", color=color,
                    markersize=14, zorder=5)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison_bar(experiments, output_path):
    """Bar chart comparing best LP and kNN across all models."""
    plt.rcParams.update(STYLE)

    names = []
    lp_vals = []
    knn_vals = []

    for name, data in experiments.items():
        if not data:
            continue
        lp_scores = [d["lp_mse"] for d in data if "lp_mse" in d]
        knn_scores = [d["knn_mse"] for d in data if "knn_mse" in d]
        if lp_scores and knn_scores:
            names.append(name)
            lp_vals.append(min(lp_scores))
            knn_vals.append(min(knn_scores))

    if not names:
        print("  No data for comparison bar chart, skipping.")
        return

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 2), 6))
    bars1 = ax.bar(x - width / 2, lp_vals, width, label="Linear Probe MSE",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, knn_vals, width, label="kNN MSE",
                   color="#FF5722", alpha=0.85)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("MSE (lower is better)")
    ax.set_title("Best Downstream Performance — All Models")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--output_dir", type=str, default="figures",
                        help="Directory to save figures")
    parser.add_argument("--log_root", type=str, default="logs",
                        help="Root directory containing eval log subdirs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover experiments from log directories
    experiment_dirs = {
        "VideoMAE Small": os.path.join(args.log_root, "videamae_v2_feat_std"),
        "JEPA v1 Small": os.path.join(args.log_root, "jepa_v2_feat_std"),
        "JEPA v3 tuned": os.path.join(args.log_root, "jepa_v3_tuned_eval"),
        "JEPA v3 strongvar": os.path.join(args.log_root, "jepa_v3_strongvar_eval"),
        "Supervised Small": os.path.join(args.log_root, "supervised_small_eval"),
        "VideoMAE Base": os.path.join(args.log_root, "videomae_base_eval"),
        "JEPA Base": os.path.join(args.log_root, "jepa_base_eval"),
    }

    experiments = {}
    for name, log_dir in experiment_dirs.items():
        if os.path.isdir(log_dir):
            data = load_experiment(log_dir)
            if data:
                experiments[name] = data
                print(f"Loaded {name}: {len(data)} checkpoints")
            else:
                print(f"Warning: {name} — no valid eval logs in {log_dir}")
        else:
            print(f"Skip: {name} — {log_dir} not found")

    if not experiments:
        print("No experiment data found. Run eval sweeps first.")
        return

    print(f"\nGenerating figures in {args.output_dir}/...")

    # 1. Feature collapse trajectory
    plot_metric_trajectory(
        experiments, "feat_std",
        ylabel="Feature Std (σ)",
        title="Feature Collapse — Standard Deviation Over Training",
        output_path=os.path.join(args.output_dir, "feature_collapse.png"),
        best_marker=False,
    )

    # 2. LP MSE trajectory
    plot_metric_trajectory(
        experiments, "lp_mse",
        ylabel="Linear Probe MSE",
        title="Linear Probe MSE Over Training Epochs",
        output_path=os.path.join(args.output_dir, "lp_mse_trajectory.png"),
    )

    # 3. kNN MSE trajectory
    plot_metric_trajectory(
        experiments, "knn_mse",
        ylabel="kNN MSE",
        title="kNN MSE Over Training Epochs",
        output_path=os.path.join(args.output_dir, "knn_mse_trajectory.png"),
    )

    # 4. LP alpha and zeta separately
    plot_metric_trajectory(
        experiments, "lp_alpha",
        ylabel="LP α MSE",
        title="Linear Probe — α Prediction Error",
        output_path=os.path.join(args.output_dir, "lp_alpha_trajectory.png"),
    )
    plot_metric_trajectory(
        experiments, "lp_zeta",
        ylabel="LP ζ MSE",
        title="Linear Probe — ζ Prediction Error",
        output_path=os.path.join(args.output_dir, "lp_zeta_trajectory.png"),
    )

    # 5. Comparison bar chart
    plot_comparison_bar(
        experiments,
        output_path=os.path.join(args.output_dir, "comparison_bar.png"),
    )

    # 6. Print summary table (for copy-paste into report)
    print("\n" + "=" * 80)
    print("SUMMARY TABLE (for report)")
    print("=" * 80)
    print(f"{'Model':<22} {'Best LP MSE':>12} {'LP Epoch':>10} {'Best kNN MSE':>13} {'kNN Epoch':>10} {'Final σ':>10}")
    print("-" * 80)
    for name, data in experiments.items():
        lp_scores = [(d["lp_mse"], d["epoch"]) for d in data if "lp_mse" in d]
        knn_scores = [(d["knn_mse"], d["epoch"]) for d in data if "knn_mse" in d]
        sigma_vals = [(d["feat_std"], d["epoch"]) for d in data if "feat_std" in d]

        if lp_scores and knn_scores:
            best_lp = min(lp_scores, key=lambda x: x[0])
            best_knn = min(knn_scores, key=lambda x: x[0])
            final_sigma = sigma_vals[-1][0] if sigma_vals else float("nan")
            print(f"{name:<22} {best_lp[0]:>12.4f} {best_lp[1]:>10} "
                  f"{best_knn[0]:>13.4f} {best_knn[1]:>10} {final_sigma:>10.4f}")
    print("=" * 80)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
