#!/bin/bash
#SBATCH --job-name=jepa-hpo
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hpo-sweep-%j.out
#SBATCH --error=logs/hpo-sweep-%j.out
#SBATCH --requeue

# ---- Hyperparameter Optimization for JEPA v3 ----
# Usage:
#   sbatch scripts/hpo_sweep.sh                          # default: 20 trials, 30 epochs each
#   sbatch scripts/hpo_sweep.sh 15 20                    # 15 trials, 20 epochs each
#
# Time estimate: ~20 trials × 30 epochs × ~3 min/epoch ≈ 30 hours
# With pruning: ~50% of trials get pruned → ~15 hours effective

N_TRIALS=${1:-20}
HPO_EPOCHS=${2:-30}

echo "=== HPO Sweep started at $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Trials: $N_TRIALS | Epochs/trial: $HPO_EPOCHS"
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    # Install optuna if not present
    pip install -q optuna 2>/dev/null

    cd /scratch/\$USER/dl-active-matter

    # Source .env if it exists
    if [ -f \".env\" ]; then
        export \$(cat .env | grep -v '^#' | xargs)
    fi

    python scripts/hpo_sweep.py \
        --config configs/jepa_v3.yaml \
        --n_trials $N_TRIALS \
        --hpo_epochs $HPO_EPOCHS \
        --study_name jepa_v3_hpo

    echo '=== HPO Sweep finished at \$(date) ==='
  "
