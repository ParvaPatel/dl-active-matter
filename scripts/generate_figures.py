"""Generate publication-quality figures by parsing test_results.txt.

Usage:
    python scripts/generate_figures.py                         # default
    python scripts/generate_figures.py --results test_results.txt  # specify file
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import argparse

# Publication-ready style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
OUT_DIR = os.path.join(PROJECT_DIR, 'report')
os.makedirs(OUT_DIR, exist_ok=True)


# ====================================================================
# PARSER — reads test_results.txt produced by parse_logs.py
# ====================================================================
def parse_results_file(filepath):
    """Parse test_results.txt into a dict of {experiment_name: data_dict}."""
    experiments = {}
    current_name = None
    current_data = {'epochs': [], 'lp_mse': [], 'lp_alpha': [], 'lp_zeta': [],
                    'knn_mse': [], 'knn_alpha': [], 'knn_zeta': [],
                    'feat_mean': [], 'feat_std': []}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            # Detect experiment header
            m = re.match(r'EVALUATION RESULTS\s*[—-]+\s*(\S+)', line)
            if m:
                if current_name and current_data['epochs']:
                    experiments[current_name] = current_data
                current_name = m.group(1)
                current_data = {'epochs': [], 'lp_mse': [], 'lp_alpha': [], 'lp_zeta': [],
                                'knn_mse': [], 'knn_alpha': [], 'knn_zeta': [],
                                'feat_mean': [], 'feat_std': []}
                continue

            # Parse data rows: | Epoch | LP MSE | LP a | LP z | kNN MSE | kNN a | kNN z | Feat m | Feat s |
            m = re.match(
                r'\|\s*(\d+|Best)\s*\|'
                r'\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
                r'\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|'
                r'\s*([-\d.]+)\s*\|\s*([-\d.]+)\s*\|',
                line
            )
            if m:
                epoch_str = m.group(1)
                if epoch_str == 'Best':
                    continue  # Skip the 'Best' summary row
                current_data['epochs'].append(int(epoch_str))
                current_data['lp_mse'].append(float(m.group(2)))
                current_data['lp_alpha'].append(float(m.group(3)))
                current_data['lp_zeta'].append(float(m.group(4)))
                current_data['knn_mse'].append(float(m.group(5)))
                current_data['knn_alpha'].append(float(m.group(6)))
                current_data['knn_zeta'].append(float(m.group(7)))
                current_data['feat_mean'].append(float(m.group(8)))
                current_data['feat_std'].append(float(m.group(9)))

    # Don't forget the last experiment
    if current_name and current_data['epochs']:
        experiments[current_name] = current_data

    return experiments


# ====================================================================
# COLORS & DISPLAY NAMES
# ====================================================================
STYLE = {
    'videomae_small':           {'color': '#2196F3', 'marker': 's', 'ls': '-',  'lw': 1.5, 'ms': 3, 'label': 'VideoMAE Small'},
    'jepa_small':               {'color': '#FF5722', 'marker': '^', 'ls': '-',  'lw': 1.5, 'ms': 3, 'label': 'JEPA v1 Small'},
    'videomae_base':            {'color': '#90CAF9', 'marker': 's', 'ls': '--', 'lw': 1.0, 'ms': 2, 'label': 'VideoMAE Base'},
    'jepa_base':                {'color': '#FFAB91', 'marker': '^', 'ls': '--', 'lw': 1.0, 'ms': 2, 'label': 'JEPA v1 Base'},
    'jepa_v3_tuned':            {'color': '#4CAF50', 'marker': 'D', 'ls': '-',  'lw': 1.5, 'ms': 3, 'label': 'JEPA v3 Tuned'},
    'jepa_v3_tuned_target':     {'color': '#66BB6A', 'marker': 'D', 'ls': '--', 'lw': 1.0, 'ms': 2, 'label': 'JEPA v3 Tuned (tgt)'},
    'jepa_v3_strongvar':        {'color': '#FFC107', 'marker': 'v', 'ls': '-',  'lw': 1.5, 'ms': 3, 'label': 'JEPA v3 Strong'},
    'jepa_v3_strongvar_target': {'color': '#FFD54F', 'marker': 'v', 'ls': '--', 'lw': 1.0, 'ms': 2, 'label': 'JEPA v3 Strong (tgt)'},
    'supervised_small':         {'color': '#9C27B0', 'marker': 'o', 'ls': '-',  'lw': 1.5, 'ms': 3, 'label': 'Supervised'},
    'jepa_v2':                  {'color': '#795548', 'marker': 'x', 'ls': '-',  'lw': 1.0, 'ms': 3, 'label': 'JEPA v2 (failed)'},
}

def get_style(name):
    return STYLE.get(name, {'color': 'gray', 'marker': '.', 'ls': '-', 'lw': 1, 'ms': 2, 'label': name})

def plot_series(ax, data, key, style):
    ax.plot(data['epochs'], data[key], marker=style['marker'], linestyle=style['ls'],
            color=style['color'], label=style['label'], markersize=style['ms'], linewidth=style['lw'])


# ====================================================================
# FIGURE GENERATION
# ====================================================================
def fig_lp_trajectory(experiments):
    """Fig 1: LP MSE vs Epoch for all models."""
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ['supervised_small', 'videomae_small', 'jepa_small', 'jepa_v3_tuned', 'videomae_base', 'jepa_base']
    for name in order:
        if name in experiments:
            plot_series(ax, experiments[name], 'lp_mse', get_style(name))
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Linear Probe MSE (test set)')
    ax.set_title('Representation Quality During Training')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(0, 0.30)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_lp_trajectory.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


def fig_collapse(experiments):
    """Fig 2: Feature std trajectory (collapse analysis)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ['supervised_small', 'videomae_small', 'jepa_small', 'jepa_v3_tuned', 'videomae_base', 'jepa_base']
    for name in order:
        if name in experiments:
            plot_series(ax, experiments[name], 'feat_std', get_style(name))
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Feature Standard Deviation')
    ax.set_title('Representation Collapse: Feature Scale During Training')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0, 105)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_collapse.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


def fig_knn_collapse(experiments):
    """Fig 3: kNN MSE showing JEPA Base catastrophic collapse."""
    fig, ax = plt.subplots(figsize=(6, 4))
    order = ['supervised_small', 'videomae_small', 'jepa_small', 'jepa_v3_tuned', 'jepa_base']
    for name in order:
        if name in experiments:
            plot_series(ax, experiments[name], 'knn_mse', get_style(name))
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('kNN MSE (test set)')
    ax.set_title('kNN Performance: JEPA Base Catastrophic Collapse')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 0.80)
    # Highlight collapse region
    if 'jepa_base' in experiments:
        ax.axvspan(45, 70, alpha=0.1, color='red')
        ax.annotate('Catastrophic\ncollapse',
                    xy=(57, 0.75), xytext=(75, 0.65),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=9, color='red', fontweight='bold')
    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_knn_collapse.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


def fig_results_bar(experiments):
    """Fig 4: Main results bar chart."""
    # Define models to show and their best-LP-epoch results
    entries = []
    model_configs = [
        ('supervised_small', 'Supervised'),
        ('videomae_small', 'VMae-S'),
        ('jepa_small', 'JEPA-S'),
        ('jepa_v3_tuned', 'JEPA v3\n(tuned)'),
        ('jepa_v3_strongvar_target', 'JEPA v3\n(strong)'),
        ('videomae_base', 'VMae-B'),
        ('jepa_base', 'JEPA-B'),
    ]
    # Fallback: try non-target version
    fallback = {'jepa_v3_strongvar_target': 'jepa_v3_strongvar'}

    for key, label in model_configs:
        actual_key = key
        if key not in experiments and key in fallback:
            actual_key = fallback[key]
        if actual_key in experiments:
            d = experiments[actual_key]
            best_idx = int(np.argmin(d['lp_mse']))
            entries.append((label, d['lp_mse'][best_idx], d['knn_mse'][best_idx]))

    if not entries:
        print("[SKIP] fig_results.png - no data")
        return

    labels, lp_vals, knn_vals = zip(*entries)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    bars1 = ax.bar(x - width/2, lp_vals, width, label='Linear Probe', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x + width/2, knn_vals, width, label='kNN (k=10)', color='#FF9800', alpha=0.85)

    ax.set_ylabel('MSE (lower is better)')
    ax.set_title('Test-Set Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    max_val = max(max(lp_vals), max(v for v in knn_vals if v < 0.5))
    ax.set_ylim(0, min(max_val * 1.3, 0.25))

    for bar in bars1:
        h = bar.get_height()
        if h < 0.25:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        if h < 0.25:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.003,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_results.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


def fig_lp_knn_divergence(experiments):
    """Fig 5: LP vs kNN divergence for JEPA v3 Tuned."""
    key = 'jepa_v3_tuned'
    if key not in experiments:
        print("[SKIP] fig_lp_knn_divergence.png - no jepa_v3_tuned data")
        return

    d = experiments[key]
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(d['epochs'], d['lp_mse'], 's-', color='#2196F3',
             label='LP MSE', markersize=4, linewidth=1.5)
    ax1.plot(d['epochs'], d['knn_mse'], '^-', color='#FF5722',
             label='kNN MSE', markersize=4, linewidth=1.5)
    ax1.set_xlabel('Training Epoch')
    ax1.set_ylabel('MSE (test set)')
    ax1.set_title('JEPA v3 Tuned: LP Improves While kNN Collapses')
    ax1.set_ylim(0, 0.22)

    ax2 = ax1.twinx()
    ax2.plot(d['epochs'], d['feat_std'], 'D--', color='gray',
             label='Feature std', markersize=3, linewidth=1, alpha=0.6)
    ax2.set_ylabel('Feature std', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    ax2.set_ylim(0, 1.0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', framealpha=0.9)

    ax1.annotate('LP improves but kNN degrades\n(directional collapse)',
                 xy=(70, 0.15), xytext=(25, 0.19),
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=8, color='red')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_lp_knn_divergence.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


def fig_alpha_vs_zeta(experiments):
    """Fig 6: Per-parameter comparison (alpha much easier than zeta)."""
    entries = []
    model_configs = [
        ('supervised_small', 'Supervised'),
        ('videomae_small', 'VMae-S'),
        ('jepa_small', 'JEPA-S'),
        ('jepa_v3_tuned', 'JEPA v3 T'),
        ('videomae_base', 'VMae-B'),
        ('jepa_base', 'JEPA-B'),
    ]
    for key, label in model_configs:
        if key in experiments:
            d = experiments[key]
            best_idx = int(np.argmin(d['lp_mse']))
            entries.append((label, d['lp_alpha'][best_idx], d['lp_zeta'][best_idx]))

    if not entries:
        print("[SKIP] fig_alpha_vs_zeta.png - no data")
        return

    labels, alphas, zetas = zip(*entries)
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width/2, alphas, width, label='alpha (dipole)', color='#42A5F5', alpha=0.85)
    ax.bar(x + width/2, zetas, width, label='zeta (alignment)', color='#EF5350', alpha=0.85)
    ax.set_ylabel('LP MSE (per parameter)')
    ax.set_title('Per-Parameter Difficulty: zeta >> alpha Across All Models')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'fig_alpha_vs_zeta.png')
    plt.savefig(path)
    plt.close()
    print(f"[OK] {path}")


# ====================================================================
# MAIN
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default=os.path.join(PROJECT_DIR, 'test_results.txt'),
                        help='Path to test_results.txt')
    args = parser.parse_args()

    print(f"Parsing {args.results}...")
    experiments = parse_results_file(args.results)
    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")
    print()

    fig_lp_trajectory(experiments)
    fig_collapse(experiments)
    fig_knn_collapse(experiments)
    fig_results_bar(experiments)
    fig_lp_knn_divergence(experiments)
    fig_alpha_vs_zeta(experiments)

    print(f"\nAll figures saved to {OUT_DIR}/")
