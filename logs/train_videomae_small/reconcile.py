import re
import csv
from pathlib import Path

ROOT_DIR = "/scratch/psp8502/dl-active-matter/logs/VideoMAE"          # or "/scratch/psp8502/dl-active-matter/logs/VideoMAE"
OUTPUT_FILE = "videomae_merged.csv"

epoch_pattern = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Val Loss:\s+([0-9.]+)"
)

best_pattern = re.compile(
    r"New best val loss:\s+([0-9.]+)"
)

epoch_data = {}

# later files overwrite older cancelled runs
log_files = sorted(
    list(Path(ROOT_DIR).rglob("*.out")) +
    list(Path(ROOT_DIR).rglob("*.log")),
    key=lambda p: p.stat().st_mtime
)

print(f"Found {len(log_files)} log files")

for logfile in log_files:
    pending_best = None

    try:
        with open(logfile, "r", errors="ignore") as f:
            lines = list(f)

        for idx, line in enumerate(lines):
            line = line.strip()

            m = epoch_pattern.search(line)
            if m:
                epoch = int(m.group(1))
                train_loss = float(m.group(2))
                val_loss = float(m.group(3))

                # check next few lines for "new best"
                is_best = False
                best_val = None

                for j in range(idx + 1, min(idx + 4, len(lines))):
                    b = best_pattern.search(lines[j])
                    if b:
                        is_best = True
                        best_val = float(b.group(1))
                        break

                epoch_data[epoch] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "is_new_best": is_best,
                    "best_val": best_val,
                    "source_file": logfile.name,
                }

    except Exception as e:
        print(f"Could not read {logfile}: {e}")

epochs = sorted(epoch_data.keys())

if not epochs:
    print("No epoch summaries found")
    exit()

rows = [epoch_data[e] for e in epochs]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "is_new_best",
            "best_val",
            "source_file",
        ]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved {len(rows)} epochs to {OUTPUT_FILE}")

missing = [e for e in range(epochs[0], epochs[-1] + 1) if e not in epoch_data]

if missing:
    print("Missing epochs:", missing)
else:
    print("No missing epochs")