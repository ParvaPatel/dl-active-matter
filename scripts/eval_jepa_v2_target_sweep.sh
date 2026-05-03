#!/bin/bash
# Eval sweep for jepa_v2 using TARGET (EMA) encoder
# Usage: bash scripts/eval_jepa_v2_target_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v2
LOG_DIR=logs/jepa_v2_target_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting TARGET encoder eval jobs for jepa_v2 ==="

for e in 5 10 15 20 25 30 35 40 45 50; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e} (target encoder)..."
    sbatch --job-name=eval-j2t-e${e} \
           --output=${LOG_DIR}/eval-j2t-e${e}-%j.out \
           --error=${LOG_DIR}/eval-j2t-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt (target encoder)..."
  sbatch --job-name=eval-j2t-best \
         --output=${LOG_DIR}/eval-j2t-best-%j.out \
         --error=${LOG_DIR}/eval-j2t-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v2_target --log_dir $LOG_DIR/"
