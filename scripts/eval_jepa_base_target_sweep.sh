#!/bin/bash
# Eval sweep for jepa_base using TARGET (EMA) encoder
# Usage: bash scripts/eval_jepa_base_target_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_base
LOG_DIR=logs/jepa_base_target_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting TARGET encoder eval jobs for jepa_base ==="

for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e} (target encoder)..."
    sbatch --job-name=eval-jbt-e${e} \
           --output=${LOG_DIR}/eval-jbt-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jbt-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt (target encoder)..."
  sbatch --job-name=eval-jbt-best \
         --output=${LOG_DIR}/eval-jbt-best-%j.out \
         --error=${LOG_DIR}/eval-jbt-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_base_target --log_dir $LOG_DIR/"
