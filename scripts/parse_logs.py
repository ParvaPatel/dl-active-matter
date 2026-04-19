"""Parse all SLURM eval logs and display + optionally log results.

Usage:
    python scripts/parse_logs.py                          # Just display
    python scripts/parse_logs.py --log --experiment jepa_v1 --eval_version v2_feat_std  # Display + log to CSV
"""

import os
import re
import glob
import argparse


def parse_log(filepath):
    """Parse a single eval log file and return a dict of results."""
    result = {"filepath": filepath}

    # Extract epoch/checkpoint name from filename
    m_epoch = re.search(r'e(\d+)', os.path.basename(filepath))
    if m_epoch:
        result["epoch"] = int(m_epoch.group(1))
        result["checkpoint"] = f"epoch_{m_epoch.group(1)}"
    elif 'best' in os.path.basename(filepath).lower():
        result["epoch"] = 999
        result["checkpoint"] = "best"
    else:
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Linear Probe
        m_lp = re.search(
            r'Linear Probe — Total MSE:\s*([\d.]+)\s*\|\s*α MSE:\s*([\d.]+)\s*\|\s*ζ MSE:\s*([\d.]+)',
            content
        )
        if m_lp:
            result["lp_mse"] = float(m_lp.group(1))
            result["lp_alpha"] = float(m_lp.group(2))
            result["lp_zeta"] = float(m_lp.group(3))

        # kNN
        m_knn = re.search(
            r'kNN — Total MSE:\s*([\d.]+)\s*\|\s*α MSE:\s*([\d.]+)\s*\|\s*ζ MSE:\s*([\d.]+)',
            content
        )
        if m_knn:
            result["knn_mse"] = float(m_knn.group(1))
            result["knn_alpha"] = float(m_knn.group(2))
            result["knn_zeta"] = float(m_knn.group(3))

        # Feature stats (v2+ eval)
        m_feat = re.search(r'Train — mean:\s*([-\d.]+),\s*std:\s*([-\d.]+)', content)
        if m_feat:
            result["feat_mean"] = float(m_feat.group(1))
            result["feat_std"] = float(m_feat.group(2))

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    if "lp_mse" not in result or "knn_mse" not in result:
        return None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="store_true", help="Also log results to experiments/results.csv")
    parser.add_argument("--experiment", type=str, default="jepa_v1", help="Experiment name for logging")
    parser.add_argument("--eval_version", type=str, default="v1", help="Eval protocol version")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory containing eval log files")
    args = parser.parse_args()

    log_files = glob.glob(os.path.join(args.log_dir, "eval-*-*.out"))
    if not log_files:
        print(f"No eval log files found in {args.log_dir}/")
        return

    results = []
    for filepath in log_files:
        r = parse_log(filepath)
        if r:
            results.append(r)

    results.sort(key=lambda x: x["epoch"])

    # Display table
    has_feat_stats = any("feat_mean" in r for r in results)

    print(f"\n{'='*90}")
    print(f"EVALUATION RESULTS — {args.experiment} (eval: {args.eval_version})")
    print(f"{'='*90}")

    header = f"| {'Epoch':<6} | {'LP MSE':<8} | {'LP α':<8} | {'LP ζ':<8} | {'kNN MSE':<8} | {'kNN α':<8} | {'kNN ζ':<8} |"
    if has_feat_stats:
        header += f" {'Feat μ':<8} | {'Feat σ':<8} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    for r in results:
        name = "Best" if r["epoch"] == 999 else str(r["epoch"])
        line = (f"| {name:<6} | {r['lp_mse']:<8.4f} | {r['lp_alpha']:<8.4f} | {r['lp_zeta']:<8.4f} "
                f"| {r['knn_mse']:<8.4f} | {r['knn_alpha']:<8.4f} | {r['knn_zeta']:<8.4f} |")
        if has_feat_stats:
            line += f" {r.get('feat_mean', 0):<8.4f} | {r.get('feat_std', 0):<8.4f} |"
        print(line)

    print(f"{'='*90}")

    # Find best
    best_lp = min(results, key=lambda x: x["lp_mse"])
    best_knn = min(results, key=lambda x: x["knn_mse"])
    best_lp_name = "Best" if best_lp["epoch"] == 999 else f"epoch_{best_lp['epoch']}"
    best_knn_name = "Best" if best_knn["epoch"] == 999 else f"epoch_{best_knn['epoch']}"
    print(f"\n★ Best Linear Probe: {best_lp_name} (MSE={best_lp['lp_mse']:.4f})")
    print(f"★ Best kNN:          {best_knn_name} (MSE={best_knn['knn_mse']:.4f})")

    # Optionally log to CSV
    if args.log:
        from scripts.log_result import append_result
        for r in results:
            row = {
                "timestamp": "",
                "experiment": args.experiment,
                "checkpoint": r["checkpoint"],
                "eval_version": args.eval_version,
                "split": "test",
                "lp_mse": r["lp_mse"],
                "lp_alpha_mse": r["lp_alpha"],
                "lp_zeta_mse": r["lp_zeta"],
                "knn_mse": r["knn_mse"],
                "knn_alpha_mse": r["knn_alpha"],
                "knn_zeta_mse": r["knn_zeta"],
                "notes": f"feat_mean={r.get('feat_mean','')}, feat_std={r.get('feat_std','')}",
            }
            from datetime import datetime
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            append_result(row)
        print(f"\n✓ All {len(results)} results logged to experiments/results.csv")


if __name__ == "__main__":
    main()
