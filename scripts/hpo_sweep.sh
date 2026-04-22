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
# Speed breakdown with new defaults:
#   20% subsample × 8750 = 1750 samples ÷ bs=16 = ~109 batches/epoch
#   ~1 min/epoch × 15 epochs × 20 trials = ~5 GPU-hours total
#   With MedianPruner (~50% pruned) → ~3 GPU-hours effective
#
# Usage:
#   sbatch scripts/hpo_sweep.sh                          # default: 20 trials, 15 epochs
#   sbatch scripts/hpo_sweep.sh 30 15                    # 30 trials, 15 epochs (more thorough)

N_TRIALS=${1:-20}
HPO_EPOCHS=${2:-15}
HPO_SUBSAMPLE=${3:-0.20}
HPO_BATCH_SIZE=${4:-16}

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
        --hpo_subsample $HPO_SUBSAMPLE \
        --hpo_batch_size $HPO_BATCH_SIZE \
        --study_name jepa_v3_hpo

    echo '=== HPO Sweep finished at \$(date) ==='
  "
