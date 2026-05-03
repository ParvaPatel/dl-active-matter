#!/bin/bash
# Download pre-trained checkpoints from Google Drive.
# Usage: bash scripts/download_checkpoints.sh
#
# The grader should run this ONCE before using reproduce.sh.

# ======================================================================
# IMPORTANT: Replace this with your actual Google Drive folder ID
# after uploading the checkpoints.
# ======================================================================
GDRIVE_FOLDER_ID="REPLACE_WITH_GDRIVE_FOLDER_ID"

CKPT_DIR=/scratch/$USER/checkpoints

echo "=== Downloading pre-trained checkpoints ==="
echo "Target: $CKPT_DIR"
echo ""

# Install gdown if not available
pip install --user --quiet gdown 2>/dev/null || true

# Download entire folder from Google Drive
python -c "
import gdown
import os

folder_id = '${GDRIVE_FOLDER_ID}'
output = '${CKPT_DIR}'

# Download all files from the shared folder
url = f'https://drive.google.com/drive/folders/{folder_id}'
gdown.download_folder(url, output=output, quiet=False)
print('Download complete!')
"

echo ""
echo "=== Verifying checkpoints ==="
for model in videomae_small videomae_base jepa_small jepa_base jepa_v2 jepa_v3_tuned jepa_v3_strongvar supervised_small; do
  if [ -f "$CKPT_DIR/$model/best_eval.pt" ]; then
    echo "  [OK] $model"
  else
    echo "  [MISSING] $model"
  fi
done

echo ""
echo "Done. Run: bash scripts/reproduce.sh <model_name>"
