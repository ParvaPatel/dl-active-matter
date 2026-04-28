#!/bin/bash
# Eval sweep for supervised_small baseline
# Usage: bash scripts/eval_supervised_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/supervised_small
LOG_DIR=logs/supervised_small_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting eval jobs for supervised_small ==="

for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-sup-e${e} \
           --output=${LOG_DIR}/eval-sup-e${e}-%j.out \
           --error=${LOG_DIR}/eval-sup-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-sup-best \
         --output=${LOG_DIR}/eval-sup-best-%j.out \
         --error=${LOG_DIR}/eval-sup-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment supervised_small --log_dir $LOG_DIR/"
