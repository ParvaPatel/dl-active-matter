# Environment Setup — NYU HPC (Greene / OOD Burst)

## Prerequisites

- NYU HPC account with access to `csci_ga_2572-2026sp` Slurm account
- NYU VPN (for off-campus access)
- ~70 GB free space on `/scratch/$USER` (52 GB dataset + checkpoints)

## 1. Access HPC

**Web Portal (recommended)**: https://ood-burst-001.hpc.nyu.edu/  
**SSH**: `ssh <NetID>@greene.hpc.nyu.edu`

## 2. Automated Setup (Recommended)

```bash
cd /scratch/$USER
git clone https://github.com/ParvaPatel/dl-active-matter.git
cd dl-active-matter
bash scripts/setup.sh
```

This script:
1. Creates a Singularity overlay filesystem (15 GB)
2. Installs Miniconda inside the overlay
3. Creates the `dl-active-matter` conda environment
4. Downloads the active_matter dataset (~52 GB)

## 3. Manual Setup (Alternative)

### 3a. Singularity Overlay

```bash
# Copy and extract overlay template
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/overlay-dl.ext3.gz
gunzip /scratch/$USER/overlay-dl.ext3.gz

# Launch Singularity shell with GPU support
singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

### 3b. Conda Environment

```bash
# Inside Singularity shell
conda env create -f environment.yml
conda activate dl-active-matter

# Or install via pip
pip install -r requirements.txt
```

### 3c. Download Dataset

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='polymathic-ai/active_matter',
    repo_type='dataset',
    local_dir='/scratch/$USER/data/active_matter'
)
"
```

## 4. Software Versions

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | ≥2.1.0 | Training, `torch.compile()` |
| CUDA | 12.1 | GPU acceleration |
| h5py | ≥3.9.0 | HDF5 dataset loading |
| scikit-learn | ≥1.3.0 | kNN evaluation |
| wandb | ≥0.15.0 | Experiment tracking (optional) |
| optuna | ≥3.4.0 | Hyperparameter optimization |
| PyYAML | ≥6.0 | Config parsing |
| numpy | ≥1.24.0 | Array operations |

## 5. Key Paths on HPC

| What | Path |
|------|------|
| Dataset | `/scratch/$USER/data/active_matter` |
| Checkpoints | `/scratch/$USER/checkpoints/<experiment_name>/` |
| Training logs | `logs/train-<job_name>-<job_id>.out` |
| Eval logs | `logs/<experiment>_eval/` |
| Code | `/scratch/$USER/dl-active-matter/` |
| Overlay | `/scratch/$USER/overlay-dl.ext3` |
| Singularity image | `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
| Conda env name | `dl-active-matter` |

## 6. SLURM Configuration

All jobs use these SLURM settings (defined in `scripts/train.sh`):

| Setting | Value |
|---------|-------|
| Account | `csci_ga_2572-2026sp` |
| Partition | `c12m85-a100-1` |
| GPU | 1× NVIDIA A100 40GB |
| CPUs | 4 |
| Memory | 40 GB |
| Time limit | 24 hours |
| Requeue | Yes (auto-resume on preemption) |

## 7. Weights & Biases (Optional)

```bash
# Create .env file in project root
echo "WANDB_API_KEY=your_key_here" > .env

# Or disable W&B entirely
echo "WANDB_MODE=disabled" > .env
```

W&B is optional — all training scripts handle missing API keys gracefully with local-only logging.

## 8. Development Workflow

```
Local (VS Code) ──git push──> GitHub ──git pull──> HPC (/scratch)
                                                     │
                                                     ├─ sbatch scripts/train.sh configs/<config>.yaml
                                                     ├─ sbatch scripts/eval.sh <checkpoint> <split>
                                                     └─ Results in logs/ directory
```
