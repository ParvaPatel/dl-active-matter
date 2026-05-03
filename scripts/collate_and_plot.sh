#!/bin/bash
# Collate all eval results into test_results.txt, then generate figures.
# Usage: bash scripts/collate_and_plot.sh
#
# This is the ONE script that does everything:
# 1. Parses all eval log directories
# 2. Writes test_results.txt
# 3. Generates publication figures in report/

set -e
cd "$(dirname "$0")/.."

RESULTS_FILE="test_results.txt"

echo "=== Collating all eval results into $RESULTS_FILE ==="
> "$RESULTS_FILE"  # Clear file

# Define all eval directories and their experiment names
declare -A EVAL_DIRS
EVAL_DIRS=(
    ["videomae_small"]="logs/videomae_small_eval"
    ["videomae_base"]="logs/videomae_base_eval"
    ["jepa_small"]="logs/jepa_small_eval"
    ["jepa_small_target"]="logs/jepa_small_target_eval"
    ["jepa_base"]="logs/jepa_base_eval"
    ["jepa_base_target"]="logs/jepa_base_target_eval"
    ["jepa_v2"]="logs/jepa_v2_model_eval"
    ["jepa_v2_target"]="logs/jepa_v2_target_eval"
    ["jepa_v3_tuned"]="logs/jepa_v3_tuned_eval"
    ["jepa_v3_tuned_target"]="logs/jepa_v3_tuned_target_eval"
    ["jepa_v3_strongvar"]="logs/jepa_v3_strongvar_eval"
    ["jepa_v3_strongvar_target"]="logs/jepa_v3_strongvar_target_eval"
    ["supervised_small"]="logs/supervised_small_eval"
)

for name in videomae_small videomae_base jepa_small jepa_small_target jepa_base jepa_base_target jepa_v2 jepa_v2_target \
            jepa_v3_tuned jepa_v3_tuned_target \
            jepa_v3_strongvar jepa_v3_strongvar_target \
            supervised_small; do
    dir="${EVAL_DIRS[$name]}"
    if [ -d "$dir" ]; then
        echo "  Parsing $name ($dir/)..."
        python scripts/parse_logs.py --experiment "$name" --log_dir "$dir/" >> "$RESULTS_FILE" 2>&1
        echo "" >> "$RESULTS_FILE"
    else
        echo "  SKIP: $dir not found"
    fi
done

echo ""
echo "=== Results written to $RESULTS_FILE ==="
echo ""

# Generate figures
echo "=== Generating publication figures ==="
python scripts/generate_figures.py --results "$RESULTS_FILE"

echo ""
echo "=== Done! ==="
echo "Results: $RESULTS_FILE"
echo "Figures: report/fig_*.png"
ls -la report/fig_*.png 2>/dev/null || echo "(no figures found)"
