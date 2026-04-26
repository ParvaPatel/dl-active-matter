#!/bin/bash
# Eval sweep for jepa_v3_strongvar (var_weight=0.8 ablation)
# Evaluates both online and target encoders at every 5-epoch checkpoint.
#
# Usage: bash scripts/eval_jepa_v3_strongvar_sweep.sh [online|target|both]

CKPT_DIR=/scratch/$USER/checkpoints/jepa_v3_strongvar
LOG_DIR_ONLINE=logs/jepa_v3_strongvar_eval
LOG_DIR_TARGET=logs/jepa_v3_strongvar_target_eval
SPLIT=test
MODE=${1:-both}  # online | target | both

mkdir -p $LOG_DIR_ONLINE $LOG_DIR_TARGET

echo "=== Submitting eval jobs for jepa_v3_strongvar (var_weight=0.8) ==="
echo "Checkpoint dir: $CKPT_DIR"
echo "Mode: $MODE"
echo ""

for e in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100; do
  CKPT="${CKPT_DIR}/epoch_${e}.pt"
  if [ ! -f "$CKPT" ]; then
    echo "SKIP: $CKPT not found"
    continue
  fi

  if [[ "$MODE" == "online" || "$MODE" == "both" ]]; then
    echo "  [online] epoch ${e}"
    sbatch --job-name=eval-jv3sv-e${e} \
           --output=${LOG_DIR_ONLINE}/eval-jv3sv-e${e}-%j.out \
           --error=${LOG_DIR_ONLINE}/eval-jv3sv-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT"
  fi

  if [[ "$MODE" == "target" || "$MODE" == "both" ]]; then
    echo "  [target] epoch ${e}"
    sbatch --job-name=eval-jv3sv-tgt-e${e} \
           --output=${LOG_DIR_TARGET}/eval-jv3sv-tgt-e${e}-%j.out \
           --error=${LOG_DIR_TARGET}/eval-jv3sv-tgt-e${e}-%j.out \
           scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
  fi
done

# Best checkpoint
for CKPT in "${CKPT_DIR}/best.pt"; do
  if [ -f "$CKPT" ]; then
    if [[ "$MODE" == "online" || "$MODE" == "both" ]]; then
      sbatch --job-name=eval-jv3sv-best \
             --output=${LOG_DIR_ONLINE}/eval-jv3sv-best-%j.out \
             --error=${LOG_DIR_ONLINE}/eval-jv3sv-best-%j.out \
             scripts/eval.sh "$CKPT" "$SPLIT"
    fi
    if [[ "$MODE" == "target" || "$MODE" == "both" ]]; then
      sbatch --job-name=eval-jv3sv-tgt-best \
             --output=${LOG_DIR_TARGET}/eval-jv3sv-tgt-best-%j.out \
             --error=${LOG_DIR_TARGET}/eval-jv3sv-tgt-best-%j.out \
             scripts/eval.sh "$CKPT" "$SPLIT" --use_target_encoder
    fi
  fi
done

echo ""
echo "Once complete, parse with:"
echo "  python scripts/parse_logs.py --experiment jepa_v3_strongvar        --log_dir $LOG_DIR_ONLINE/"
echo "  python scripts/parse_logs.py --experiment jepa_v3_strongvar_target --log_dir $LOG_DIR_TARGET/"
