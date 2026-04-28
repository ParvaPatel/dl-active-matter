#!/bin/bash
# Eval sweep for videomae_base (ViT-Base encoder)
# Usage: bash scripts/eval_videomae_base_sweep.sh

CKPT_DIR=/scratch/$USER/checkpoints/videomae_base
LOG_DIR=logs/videomae_base_eval
SPLIT=test

mkdir -p $LOG_DIR

echo "=== Submitting eval jobs for videomae_base (ViT-Base) ==="
echo "Checkpoint dir: $CKPT_DIR"
echo ""

for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ -f "$CKPT" ]; then
    echo "Submitting eval for epoch ${e}..."
    sbatch --job-name=eval-vmb-e${e} \
           --output=${LOG_DIR}/eval-vmb-e${e}-%j.out \
           --error=${LOG_DIR}/eval-vmb-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  else
    echo "SKIP: $CKPT not found"
  fi
done

CKPT="${CKPT_DIR}/best.pt"
if [ -f "$CKPT" ]; then
  echo "Submitting eval for best.pt..."
  sbatch --job-name=eval-vmb-best \
         --output=${LOG_DIR}/eval-vmb-best-%j.out \
         --error=${LOG_DIR}/eval-vmb-best-%j.out \
         scripts/eval.sh "$CKPT" "$SPLIT"
fi

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment videomae_base --log_dir $LOG_DIR/"
