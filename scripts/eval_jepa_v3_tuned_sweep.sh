#!/bin/bash
# Eval sweep for all jepa_v3_tuned checkpoints from the first training run.
#
# Checkpoints available (epochs 0-15, saved every 5):
#   best.pt    — epoch 0, val_loss=0.3507 (best during warmup-only run)
#   epoch_5.pt — epoch 5
#   epoch_10.pt — epoch 10
#   epoch_15.pt — epoch 15 (last epoch, val_loss=0.5048)
#   latest.pt  — same as epoch_15.pt
#
# Note: "best.pt" here is epoch 0 which had the lowest val_loss only because
# the warmup LR was very conservative. The epoch_15.pt is architecturally the
# most trained and may still give the best downstream eval despite higher val_loss.
#
# Usage: bash scripts/eval_jepa_v3_tuned_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v3_tuned
LOG_DIR=logs/jepa_v3_tuned_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting eval jobs for jepa_v3_tuned ==="
echo "Checkpoint dir: $CKPT_DIR"
echo "Log dir:        $LOG_DIR"
echo "Split:          $SPLIT"
echo ""

# --- Epoch checkpoints ---
for e in 5 10 15; do
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

# --- Best checkpoint (epoch 0 — warmup model) ---
CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt (epoch 0 warmup model)..."
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
