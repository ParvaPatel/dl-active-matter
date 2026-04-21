#!/bin/bash
# Submit eval jobs for all JEPA v3 checkpoints
# Usage: bash scripts/eval_jepa_v3_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v3
LOG_DIR=logs/jepa_v3_eval
SPLIT=test

mkdir -p $LOG_DIR

# Epoch checkpoints (every 5 epochs up to 100)
for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-jv3-e${e} \
           --output=${LOG_DIR}/eval-jv3-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jv3-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

# Best checkpoint
CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-jv3-best \
         --output=${LOG_DIR}/eval-jv3-best-%j.out \
         --error=${LOG_DIR}/eval-jv3-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
fi

echo ""
echo "All jobs submitted. Logs will be in: $LOG_DIR/"
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v3 --log_dir $LOG_DIR/"
