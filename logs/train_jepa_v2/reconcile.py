import re
import csv
from pathlib import Path

ROOT_DIR = "."   # or your logs directory
OUTPUT_FILE = "merged_experiment_detailed.csv"

epoch_pattern = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+"
    r"Train:\s+([0-9.]+)\s+"
    r"\(pred=([0-9.]+),\s*var=([0-9.]+),\s*cov=([0-9.]+)\)\s+\|\s+"
    r"Val:\s+([0-9.]+)\s+\|\s+"
    r"LR:\s+([0-9.eE+-]+)\s+\|\s+"
    r"EMA:\s+([0-9.]+)"
)

best_pattern = re.compile(
    r"New best val loss:\s+([0-9.]+)"
)

epoch_data = {}

log_files = sorted(
    list(Path(ROOT_DIR).rglob("*.out")) +
    list(Path(ROOT_DIR).rglob("*.log")),
    key=lambda p: p.stat().st_mtime
)

print(f"Found {len(log_files)} log files")

for logfile in log_files:
    try:
        with open(logfile, "r", errors="ignore") as f:
            lines = list(f)

        for idx, line in enumerate(lines):
            line = line.strip()

            m = epoch_pattern.search(line)
            if not m:
                continue

            epoch = int(m.group(1))

            train_total = float(m.group(2))
            pred_loss = float(m.group(3))
            var_loss = float(m.group(4))
            cov_loss = float(m.group(5))
            val_loss = float(m.group(6))
            lr = float(m.group(7))
            ema = float(m.group(8))

            is_best = False
            best_val = None

            # look at next 3 lines for "new best"
            for j in range(idx + 1, min(idx + 4, len(lines))):
                b = best_pattern.search(lines[j])
                if b:
                    is_best = True
                    best_val = float(b.group(1))
                    break

            epoch_data[epoch] = {
                "epoch": epoch,
                "train_total": train_total,
                "pred_loss": pred_loss,
                "var_loss": var_loss,
                "cov_loss": cov_loss,
                "val_loss": val_loss,
                "lr": lr,
                "ema": ema,
                "is_new_best": is_best,
                "best_val": best_val,
                "source_file": logfile.name,
            }

    except Exception as e:
        print(f"Could not read {logfile}: {e}")

epochs = sorted(epoch_data.keys())

if not epochs:
    print("No epoch summaries found.")
    exit()

rows = [epoch_data[e] for e in epochs]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_total",
            "pred_loss",
            "var_loss",
            "cov_loss",
            "val_loss",
            "lr",
            "ema",
            "is_new_best",
            "best_val",
            "source_file",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} epochs to {OUTPUT_FILE}")

missing = [e for e in range(epochs[0], epochs[-1] + 1) if e not in epoch_data]

if missing:
    print("Missing epochs:", missing)
else:
    print("No missing epochs")