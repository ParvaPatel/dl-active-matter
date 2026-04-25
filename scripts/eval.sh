#!/bin/bash
#SBATCH --job-name=eval-probe
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=logs/eval-%x-%j.out
#SBATCH --error=logs/eval-%x-%j.out
#SBATCH --requeue

# ---- Usage ----
# sbatch scripts/eval.sh <checkpoint> <split> [extra eval.py flags]
# sbatch scripts/eval.sh /scratch/$USER/checkpoints/jepa_v3_tuned/epoch_90.pt test
# sbatch scripts/eval.sh /scratch/$USER/checkpoints/jepa_v3_tuned/best.pt test --use_target_encoder

CHECKPOINT=${1:-/scratch/$USER/checkpoints/videomae_small/best.pt}
SPLIT=${2:-val}
shift 2  # remaining args forwarded to eval.py (e.g. --use_target_encoder)
EXTRA_ARGS="$@"

echo "=== Eval started at $(date) ==="
echo "Checkpoint: $CHECKPOINT | Split: $SPLIT | Extra: $EXTRA_ARGS"
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    export PYTHONUNBUFFERED=1

    cd /scratch/\$USER/dl-active-matter

    python eval.py \\
      --checkpoint $CHECKPOINT \\
      --split $SPLIT \\
      --batch_size 16 \\
      --probe_epochs 200 \\
      --k 10 \\
      $EXTRA_ARGS

    echo '=== Eval finished at \$(date) ==='
  "
