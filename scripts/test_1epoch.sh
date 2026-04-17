#!/bin/bash
#SBATCH --job-name=test-1epoch
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=logs/train-%x-%j.out
#SBATCH --error=logs/train-%x-%j.out
#SBATCH --requeue

echo "=== Job started at $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter
    export WANDB_MODE=disabled

    cd /scratch/\$USER/dl-active-matter

    python train.py --config configs/test_1epoch.yaml

    echo '=== Job finished at \$(date) ==='
  "
