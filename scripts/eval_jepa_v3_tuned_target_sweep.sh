#!/bin/bash
# Eval sweep using the TARGET (EMA) encoder from jepa_v3_tuned checkpoints.
#
# In JEPA/BYOL, the target encoder is the EMA-smoothed network and is the
# recommended encoder for downstream evaluation — NOT the online encoder.
# This script re-runs the same checkpoints with --use_target_encoder.
#
# Usage: bash scripts/eval_jepa_v3_tuned_target_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v3_tuned
LOG_DIR=logs/jepa_v3_tuned_target_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting TARGET ENCODER eval jobs for jepa_v3_tuned ==="
echo "Checkpoint dir: $CKPT_DIR"
echo "Log dir:        $LOG_DIR"
echo "Split:          $SPLIT"
echo ""

# All epoch checkpoints (every 5 epochs, 5 → 100)
for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting target-encoder eval for epoch ${e}..."
    sbatch --job-name=eval-jv3t-tgt-e${e} \
           --output=${LOG_DIR}/eval-jv3t-tgt-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jv3t-tgt-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
  else
    echo "SKIP: $CKPT not found"
  fi
done

# Best checkpoint
CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting target-encoder eval for best.pt..."
  sbatch --job-name=eval-jv3t-tgt-best \
         --output=${LOG_DIR}/eval-jv3t-tgt-best-%j.out \
         --error=${LOG_DIR}/eval-jv3t-tgt-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
fi

echo ""
echo "All jobs submitted. Logs will be in: $LOG_DIR/"
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v3_tuned_target --log_dir $LOG_DIR/"
