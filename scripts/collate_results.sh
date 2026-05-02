#!/bin/bash
# Run this on HPC from the dl-active-matter directory

echo "========================================"
echo "COLLATING ALL EVALUATION RESULTS"
echo "========================================"

for dir in \
  logs/videamae_small_eval \
  logs/jepa_small_eval \
  logs/jepa_v2_model_eval \
  logs/jepa_v3_tuned_eval \
  logs/jepa_v3_tuned_target_eval \
  logs/jepa_v3_strongvar_eval \
  logs/jepa_v3_strongvar_target_eval \
  logs/videomae_base_eval \
  logs/jepa_base_eval \
  logs/supervised_small_eval \
  logs/JEPA-eval-sweeps \
  logs/JEPA-test
do
  name=$(basename $dir)
  if [ -d "$dir" ]; then
    echo ""
    python scripts/parse_logs.py --experiment "$name" --log_dir "$dir/"
  fi
done

# Also check standalone eval files in logs/
echo ""
echo "========================================"
echo "STANDALONE EVAL FILES"
echo "========================================"
for f in logs/eval-eval-*.out; do
  if [ -f "$f" ]; then
    echo "--- $(basename $f) ---"
    grep -E "LP MSE|kNN MSE|Feat stats" "$f" | head -5
  fi
done
