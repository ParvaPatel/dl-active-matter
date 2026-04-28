#!/bin/bash
# Eval sweep for jepa_base (ViT-Base encoder, JEPA v1 architecture)
# Usage: bash scripts/eval_jepa_base_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/jepa_base
LOG_DIR=logs/jepa_base_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting eval jobs for jepa_base (ViT-Base, JEPA v1) ==="
echo "Checkpoint dir: $CKPT_DIR"
echo ""

for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-jb-e${e} \
           --output=${LOG_DIR}/eval-jb-e${e}-%j.out \
           --error=${LOG_DIR}/eval-jb-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-jb-best \
         --output=${LOG_DIR}/eval-jb-best-%j.out \
         --error=${LOG_DIR}/eval-jb-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_base --log_dir $LOG_DIR/"
