#!/bin/bash
# Submit eval jobs for all JEPA v2 checkpoints
# Usage: bash scripts/eval_jepa_v2_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v2
LOG_DIR=logs/jepa_v2_model_eval
SPLIT=test

mkdir -p $LOG_DIR

# Epoch checkpoints
for e in 5 10 15 20 25 30 35 40 45 50; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-jv2-e${e} \
           --output=${LOG_DIR}/eval-jv2-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jv2-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

# Best checkpoint
CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-jv2-best \
         --output=${LOG_DIR}/eval-jv2-best-%j.out \
         --error=${LOG_DIR}/eval-jv2-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
fi

echo ""
echo "All jobs submitted. Logs will be in: $LOG_DIR/"
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v2_model --log_dir $LOG_DIR/"
