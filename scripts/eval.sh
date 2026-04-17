#!/bin/bash
#SBATCH --job-name=eval-probe
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval-%x-%j.out
#SBATCH --error=logs/eval-%x-%j.out
#SBATCH --requeue

# ---- Usage ----
# sbatch scripts/eval.sh                                                    # default: videomae_small/best.pt
# sbatch scripts/eval.sh /scratch/$USER/checkpoints/jepa_small/best.pt      # custom checkpoint
# sbatch scripts/eval.sh /scratch/$USER/checkpoints/videomae_small/best.pt test  # eval on test split

CHECKPOINT=${1:-/scratch/$USER/checkpoints/videomae_small/best.pt}
SPLIT=${2:-val}

echo "=== Eval started at $(date) ==="
echo "Checkpoint: $CHECKPOINT | Split: $SPLIT"
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    cd /scratch/\$USER/dl-active-matter

    python eval.py \\
      --checkpoint $CHECKPOINT \\
      --split $SPLIT \\
      --batch_size 16 \\
      --probe_epochs 50 \\
      --k 10

    echo '=== Eval finished at \$(date) ==='
  "
