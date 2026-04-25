#!/bin/bash
# Eval sweep for all jepa_v3_tuned checkpoints — full 100-epoch run.
#
# Checkpoints (every 5 epochs):
#   epoch_5.pt through epoch_100.pt  (20 checkpoints)
#   best.pt  — lowest val_loss (epoch ~99, val=0.2512)
#   latest.pt — same as epoch_100.pt
#
# Usage: bash scripts/eval_jepa_v3_tuned_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v3_tuned
LOG_DIR=logs/jepa_v3_tuned_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting eval jobs for jepa_v3_tuned (full 100-epoch run) ==="
echo "Checkpoint dir: $CKPT_DIR"
echo "Log dir:        $LOG_DIR"
echo "Split:          $SPLIT"
echo ""

# --- All epoch checkpoints (every 5 epochs, 5 → 100) ---
for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-jv3t-e${e} \
           --output=${LOG_DIR}/eval-jv3t-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jv3t-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

# --- Best checkpoint (lowest val_loss = ~epoch 99) ---
CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-jv3t-best \
         --output=${LOG_DIR}/eval-jv3t-best-%j.out \
         --error=${LOG_DIR}/eval-jv3t-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
else
  echo "SKIP: $CKPT not found"
fi

echo ""
echo "All jobs submitted. Logs will be in: $LOG_DIR/"
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v3_tuned --log_dir $LOG_DIR/"

