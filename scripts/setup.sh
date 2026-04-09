#!/bin/bash
# =============================================================================
# One-command HPC setup: environment + dataset
# Usage: bash scripts/setup.sh
# =============================================================================
set -euo pipefail

SCRATCH="/scratch/$USER"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OVERLAY_PATH="$SCRATCH/overlay-dl.ext3"
SIF_IMAGE="/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif"
DATA_DIR="$SCRATCH/data/active_matter"

echo "============================================"
echo "  DL Active Matter — HPC Setup"
echo "============================================"
echo "Repo:    $REPO_DIR"
echo "Scratch: $SCRATCH"
echo ""

# --- Step 1: Singularity overlay ---
if [ ! -f "$OVERLAY_PATH" ]; then
    echo "[1/3] Creating Singularity overlay (15 GB)..."
    cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz "$OVERLAY_PATH.gz"
    gunzip "$OVERLAY_PATH.gz"
    echo "  ✓ Overlay created at $OVERLAY_PATH"
else
    echo "[1/3] Overlay already exists at $OVERLAY_PATH — skipping."
fi

# --- Step 2: Install Miniconda + Conda environment ---
echo "[2/3] Setting up Miniconda + conda environment inside Singularity..."
singularity exec --nv \
    --overlay "$OVERLAY_PATH:rw" \
    "$SIF_IMAGE" \
    /bin/bash -c '
        set -euo pipefail

        # ---- Install Miniconda if not present ----
        CONDA_DIR=/ext3/miniconda3
        if [ ! -d "$CONDA_DIR" ]; then
            echo "  Installing Miniconda..."
            wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
            bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
            rm /tmp/miniconda.sh
            echo "  ✓ Miniconda installed at $CONDA_DIR"
        else
            echo "  Miniconda already installed — skipping."
        fi

        # ---- Activate conda ----
        source "$CONDA_DIR/etc/profile.d/conda.sh"
        conda init bash 2>/dev/null || true

        # ---- Create or update conda env ----
        if conda env list | grep -q "dl-active-matter"; then
            echo "  Conda env dl-active-matter already exists — updating..."
            conda activate dl-active-matter
            pip install -r '"$REPO_DIR"'/requirements.txt --quiet
        else
            echo "  Creating conda env dl-active-matter..."
            conda env create -f '"$REPO_DIR"'/environment.yml
            conda activate dl-active-matter
        fi

        echo "  ✓ Conda environment ready"
        python -c "import torch; print(f\"  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}\")"
    '

# --- Step 3: Dataset ---
if [ -d "$DATA_DIR" ] && [ "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
    echo "[3/3] Dataset already exists at $DATA_DIR — skipping download."
else
    echo "[3/3] Downloading active_matter dataset (~52 GB)..."
    echo "  This will take a while. Run in a screen/tmux session if needed."
    mkdir -p "$DATA_DIR"
    singularity exec --nv \
        --overlay "$OVERLAY_PATH:ro" \
        "$SIF_IMAGE" \
        /bin/bash -c '
            source /ext3/miniconda3/etc/profile.d/conda.sh
            conda activate dl-active-matter
            huggingface-cli download polymathic-ai/active_matter \
                --repo-type dataset \
                --local-dir '"$DATA_DIR"'
        '
    echo "  ✓ Dataset downloaded to $DATA_DIR"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Submit a training job:"
echo "     sbatch scripts/train.sh"
echo ""
echo "  2. Monitor on W&B:"
echo "     https://wandb.ai/"
echo ""
echo "  3. Evaluate:"
echo "     sbatch scripts/eval.sh"
echo ""
