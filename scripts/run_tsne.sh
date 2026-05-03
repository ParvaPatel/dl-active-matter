#!/bin/bash
#SBATCH --job-name=tsne-viz
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=logs/tsne-%j.out
#SBATCH --error=logs/tsne-%j.out

echo "=== t-SNE / PCA visualization started at $(date) ==="
nvidia-smi

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:ro \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash -c "
    source /ext3/miniconda3/etc/profile.d/conda.sh
    conda activate dl-active-matter

    # Install viz dependencies if missing (overlay is read-only, use --user)
    pip install --user --quiet matplotlib scikit-learn 2>/dev/null || true

    export PYTHONUNBUFFERED=1
    cd /scratch/\$USER/dl-active-matter

    echo '--- VideoMAE Small (epoch 90) ---'
    python scripts/visualize_tsne.py \
      --checkpoint /scratch/\$USER/checkpoints/videomae_small/epoch_90.pt \
      --split test --output report/fig_tsne_vmae.png \
      --title 'VideoMAE Small (epoch 90)'

    echo '--- JEPA v1 Small (epoch 80, target encoder) ---'
    python scripts/visualize_tsne.py \
      --checkpoint /scratch/\$USER/checkpoints/jepa_small/epoch_80.pt \
      --split test --output report/fig_tsne_jepa.png \
      --use_target_encoder \
      --title 'JEPA v1 Small (epoch 80, target enc.)'

    echo '--- Supervised (epoch 65) ---'
    python scripts/visualize_tsne.py \
      --checkpoint /scratch/\$USER/checkpoints/supervised_small/epoch_65.pt \
      --split test --output report/fig_tsne_supervised.png \
      --title 'Supervised (epoch 65)'

    echo '=== Done at \$(date) ==='
    ls -la report/fig_tsne_*.png
  "
