#!/bin/bash
#SBATCH --job-name=eval-probe
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --requeue

echo "=== Eval started at $(date) ==="
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl

    cd /scratch/$USER/dl-active-matter

    python eval.py \
      --checkpoint /scratch/$USER/checkpoints/best.pt \
      --split val \
      --batch_size 16 \
      --probe_epochs 50 \
      --k 10

    echo '=== Eval finished at $(date) ==='
  "
