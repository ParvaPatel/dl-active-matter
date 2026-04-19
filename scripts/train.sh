#!/bin/bash
#SBATCH --job-name=videomae-train
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train-%x-%j.out
#SBATCH --error=logs/train-%x-%j.out
#SBATCH --requeue

# ---- Usage ----
# sbatch scripts/train.sh                              # default: videomae_small
# sbatch scripts/train.sh configs/jepa_small.yaml      # custom config
# sbatch --job-name=jepa-train scripts/train.sh configs/jepa_small.yaml

CONFIG=${1:-configs/videomae_small.yaml}

echo "=== Job started at $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG"
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    cd /scratch/\$USER/dl-active-matter

    # Source .env if it exists (great for WANDB_API_KEY)
    if [ -f ".env" ]; then
        export \$(cat .env | grep -v '^#' | xargs)
    fi

    # Route to correct entrypoint based on config name
    if [[ "$CONFIG" == *"jepa_v2"* ]]; then
        python train_jepa_v2.py --config $CONFIG
    elif [[ "$CONFIG" == *"jepa"* ]]; then
        python train_jepa.py --config $CONFIG
    else
        python train.py --config $CONFIG
    fi

    echo '=== Job finished at \$(date) ==='
  "
