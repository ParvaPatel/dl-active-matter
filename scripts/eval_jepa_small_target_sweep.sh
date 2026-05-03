#!/bin/bash
# Eval sweep for jepa_small using TARGET (EMA) encoder
# Usage: bash scripts/eval_jepa_small_target_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_small
LOG_DIR=logs/jepa_small_target_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting TARGET encoder eval jobs for jepa_small ==="

for e in 10 20 30 40 50 60 70 80 90 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e} (target encoder)..."
    sbatch --job-name=eval-jst-e${e} \
           --output=${LOG_DIR}/eval-jst-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jst-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt (target encoder)..."
  sbatch --job-name=eval-jst-best \
         --output=${LOG_DIR}/eval-jst-best-%j.out \
         --error=${LOG_DIR}/eval-jst-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_small_target --log_dir $LOG_DIR/"
