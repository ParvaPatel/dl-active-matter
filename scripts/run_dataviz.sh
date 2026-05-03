#!/bin/bash
#SBATCH --job-name=dataviz
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=logs/dataviz-%j.out
#SBATCH --error=logs/dataviz-%j.out

echo "=== Dataset visualization started at $(date) ==="

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    pip install --user --quiet matplotlib 2>/dev/null || true

    export PYTHONUNBUFFERED=1
    cd /scratch/\$USER/dl-active-matter

    python scripts/visualize_dataset.py --output report/fig_dataset.png

    echo '=== Done at \$(date) ==='
    ls -la report/fig_dataset*.png
  "
