#!/bin/bash
#SBATCH --job-name=videomae-train
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --requeue

# ---- Spot instance resilience ----
# Job auto-requeues on preemption; checkpoint/resume handled in train.py

echo "=== Job started at $(date) ==="
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ---- Environment ----
singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter
    export WANDB_MODE=disabled

    cd /scratch/$USER/dl-active-matter

    python train.py --config configs/videomae_small.yaml

    echo '=== Job finished at $(date) ==='
  "
