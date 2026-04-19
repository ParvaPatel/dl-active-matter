"""Structured experiment result logging for ablation studies.

Usage:
    # After an eval run, log the result:
    python scripts/log_result.py \
        --experiment "jepa_v1" \
        --checkpoint "epoch_40" \
        --eval_version "v2_feat_std" \
        --lp_mse 0.1168 --lp_alpha 0.0350 --lp_zeta 0.2420 \
        --knn_mse 0.0687 --knn_alpha 0.0028 --knn_zeta 0.1239 \
        --notes "Feature standardization + 200 probe epochs + cosine LR"

    # Or parse from a log file automatically:
    python scripts/log_result.py --from_log logs/eval-v2-e40-12345.out \
        --experiment "jepa_v1" --eval_version "v2_feat_std"
"""

import argparse
import csv
import os
import re
from datetime import datetime

RESULTS_FILE = "experiments/results.csv"
COLUMNS = [
    "timestamp", "experiment", "checkpoint", "eval_version", "split",
    "lp_mse", "lp_alpha_mse", "lp_zeta_mse",
    "knn_mse", "knn_alpha_mse", "knn_zeta_mse",
    "notes"
]


def parse_log_file(filepath):
    """Extract results from a SLURM eval log file."""
    results = {}
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Extract checkpoint path
    m = re.search(r'Checkpoint:\s*(\S+)', content)
    if m:
        ckpt_path = m.group(1)
        # Derive checkpoint name from path (e.g. "epoch_40" from ".../epoch_40.pt")
        results["checkpoint"] = os.path.splitext(os.path.basename(ckpt_path))[0]

    # Extract split
    m = re.search(r'Split:\s*(\w+)', content)
    if m:
        results["split"] = m.group(1)

    # Extract Linear Probe scores
    m = re.search(
        r'Linear Probe — Total MSE:\s*([\d.]+)\s*\|\s*α MSE:\s*([\d.]+)\s*\|\s*ζ MSE:\s*([\d.]+)',
        content
    )
    if m:
        results["lp_mse"] = float(m.group(1))
        results["lp_alpha_mse"] = float(m.group(2))
        results["lp_zeta_mse"] = float(m.group(3))

    # Extract kNN scores
    m = re.search(
        r'kNN — Total MSE:\s*([\d.]+)\s*\|\s*α MSE:\s*([\d.]+)\s*\|\s*ζ MSE:\s*([\d.]+)',
        content
    )
    if m:
        results["knn_mse"] = float(m.group(1))
        results["knn_alpha_mse"] = float(m.group(2))
        results["knn_zeta_mse"] = float(m.group(3))

    # Extract feature stats if present
    m = re.search(r'Train — mean:\s*([-\d.]+),\s*std:\s*([-\d.]+)', content)
    if m:
        results["feat_mean"] = float(m.group(1))
        results["feat_std"] = float(m.group(2))

    return results


def append_result(row):
    """Append a result row to the CSV file."""
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    file_exists = os.path.exists(RESULTS_FILE)

    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"✓ Logged to {RESULTS_FILE}")


def print_results_table():
    """Print all logged results as a formatted table."""
    if not os.path.exists(RESULTS_FILE):
        print("No results logged yet.")
        return

    with open(RESULTS_FILE, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No results logged yet.")
        return

    print(f"\n{'='*100}")
    print(f"{'Experiment':<15} {'Checkpoint':<12} {'EvalVer':<15} {'LP MSE':<10} {'LP α':<10} {'LP ζ':<10} {'kNN MSE':<10} {'kNN α':<10} {'kNN ζ':<10}")
    print(f"{'='*100}")
    for r in rows:
        print(f"{r.get('experiment',''):<15} "
              f"{r.get('checkpoint',''):<12} "
              f"{r.get('eval_version',''):<15} "
              f"{r.get('lp_mse',''):<10} "
              f"{r.get('lp_alpha_mse',''):<10} "
              f"{r.get('lp_zeta_mse',''):<10} "
              f"{r.get('knn_mse',''):<10} "
              f"{r.get('knn_alpha_mse',''):<10} "
              f"{r.get('knn_zeta_mse',''):<10}")
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Log evaluation results")
    parser.add_argument("--from_log", type=str, help="Parse results from SLURM log file")
    parser.add_argument("--experiment", type=str, required=True, help="Experiment name (e.g. jepa_v1, videomae_v1)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint name (e.g. epoch_40, best)")
    parser.add_argument("--eval_version", type=str, default="v1", help="Eval protocol version (e.g. v1, v2_feat_std)")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--lp_mse", type=float, default=None)
    parser.add_argument("--lp_alpha", type=float, default=None)
    parser.add_argument("--lp_zeta", type=float, default=None)
    parser.add_argument("--knn_mse", type=float, default=None)
    parser.add_argument("--knn_alpha", type=float, default=None)
    parser.add_argument("--knn_zeta", type=float, default=None)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--show", action="store_true", help="Print all logged results")
    args = parser.parse_args()

    if args.show:
        print_results_table()
        return

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.experiment,
        "eval_version": args.eval_version,
        "split": args.split,
        "notes": args.notes,
    }

    if args.from_log:
        parsed = parse_log_file(args.from_log)
        row["checkpoint"] = parsed.get("checkpoint", args.checkpoint or "unknown")
        row["split"] = parsed.get("split", args.split)
        row["lp_mse"] = parsed.get("lp_mse", "")
        row["lp_alpha_mse"] = parsed.get("lp_alpha_mse", "")
        row["lp_zeta_mse"] = parsed.get("lp_zeta_mse", "")
        row["knn_mse"] = parsed.get("knn_mse", "")
        row["knn_alpha_mse"] = parsed.get("knn_alpha_mse", "")
        row["knn_zeta_mse"] = parsed.get("knn_zeta_mse", "")
    else:
        row["checkpoint"] = args.checkpoint or "unknown"
        row["lp_mse"] = args.lp_mse or ""
        row["lp_alpha_mse"] = args.lp_alpha or ""
        row["lp_zeta_mse"] = args.lp_zeta or ""
        row["knn_mse"] = args.knn_mse or ""
        row["knn_alpha_mse"] = args.knn_alpha or ""
        row["knn_zeta_mse"] = args.knn_zeta or ""

    append_result(row)
    print_results_table()


if __name__ == "__main__":
    main()
