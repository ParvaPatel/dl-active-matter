import re
import csv
from pathlib import Path

ROOT_DIR = "/scratch/psp8502/dl-active-matter/logs/JEPA"   # top-level directory containing all log files/subfolders
OUTPUT_FILE = "merged_experiment.csv"

# Match:
# Epoch 22 | Train Loss: 0.6345 | Val Loss: 1.0263
epoch_pattern = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Val Loss:\s+([0-9.]+)"
)

# Optional:
# -> New best val loss: 0.9897
best_pattern = re.compile(
    r"New best val loss:\s+([0-9.]+)"
)

# Store only one entry per epoch.
# If the same epoch appears in multiple files, later modified file wins.
epoch_data = {}

# Sort files by modification time so resumed runs overwrite older cancelled runs
log_files = sorted(
    [p for p in Path(ROOT_DIR).rglob("*") if p.is_file()],
    key=lambda p: p.stat().st_mtime
)
print(log_files)
for logfile in log_files:
    current_best = None

    try:
        with open(logfile, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()

                b = best_pattern.search(line)
                if b:
                    current_best = float(b.group(1))
                    continue

                m = epoch_pattern.search(line)
                if m:
                    epoch = int(m.group(1))
                    train_loss = float(m.group(2))
                    val_loss = float(m.group(3))

                    epoch_data[epoch] = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "best_val_so_far": current_best,
                        "source_file": logfile.name,
                        "source_path": str(logfile),
                    }

    except Exception as e:
        print(f"Could not read {logfile}: {e}")

# Sort by epoch into one continuous experiment
merged_rows = [epoch_data[e] for e in sorted(epoch_data.keys())]

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "epoch",
            "train_loss",
            "val_loss",
            "best_val_so_far",
            "source_file",
            "source_path",
        ],
    )
    writer.writeheader()
    writer.writerows(merged_rows)

print(f"Saved merged experiment with {len(merged_rows)} epochs to {OUTPUT_FILE}")

# Optional: print missing epochs
epochs = sorted(epoch_data.keys())
missing = []

for i in range(epochs[0], epochs[-1] + 1):
    if i not in epoch_data:
        missing.append(i)

if missing:
    print("Missing epochs:", missing)
else:
    print("No missing epochs.")