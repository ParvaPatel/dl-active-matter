#!/bin/bash
# Reproduce reported results for any model.
# Usage: bash scripts/reproduce.sh <model_name>
#
# Examples:
#   bash scripts/reproduce.sh videomae_small
#   bash scripts/reproduce.sh jepa_small
#   bash scripts/reproduce.sh supervised_small
#
# Available models:
#   videomae_small, videomae_base, jepa_small, jepa_base,
#   jepa_v3_tuned, jepa_v3_strongvar, supervised_small

MODEL=${1:?Usage: bash scripts/reproduce.sh <model_name>}
CKPT_DIR=/scratch/$USER/checkpoints/$MODEL
CKPT="${CKPT_DIR}/best_eval.pt"
SPLIT=test

# Determine if model needs --use_target_encoder
case "$MODEL" in
  jepa_v3_strongvar)
    EXTRA_ARGS="--use_target_encoder"
    ;;
  *)
    EXTRA_ARGS=""
    ;;
esac

if [ ! -f "$CKPT" ]; then
  echo "ERROR: Checkpoint not found: $CKPT"
  echo "Available checkpoints:"
  ls "$CKPT_DIR"/*.pt 2>/dev/null || echo "  (none)"
  exit 1
fi

echo "=== Reproducing results for: $MODEL ==="
echo "Checkpoint: $CKPT"
echo "Split: $SPLIT"
echo "Extra args: $EXTRA_ARGS"
echo ""

sbatch --job-name=repro-${MODEL} \
  --output=logs/reproduce-${MODEL}-%j.out \
  --error=logs/reproduce-${MODEL}-%j.out \
  scripts/eval.sh "$CKPT" "$SPLIT" $EXTRA_ARGS

echo "Submitted. Check logs/reproduce-${MODEL}-*.out for results."
