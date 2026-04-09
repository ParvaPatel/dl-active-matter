# Environment Setup — NYU HPC (OOD Burst)

## 1. Access HPC

Portal: https://ood-burst-001.hpc.nyu.edu/ (NYU VPN required off-campus)

## 2. Create Conda Environment via Singularity Overlay

```bash
# Copy an overlay template (15 GB recommended)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/overlay-dl.ext3.gz
gunzip /scratch/$USER/overlay-dl.ext3.gz

# Launch a Singularity shell with GPU access
singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash

# Inside the container, create conda env
conda create -n dl python=3.10 -y
conda activate dl
pip install -r requirements.txt
```

## 3. Download Dataset (Run Once)

```bash
# From an HPC terminal (or Jupyter terminal)
huggingface-cli download polymathic-ai/active_matter \
  --repo-type dataset \
  --local-dir /scratch/$USER/data/active_matter
```

Dataset is ~52 GB. Check your `/scratch` quota first.

## 4. Clone Repository on HPC

```bash
cd /scratch/$USER
git clone https://github.com/<YOUR-USERNAME>/dl-active-matter.git
cd dl-active-matter
```

## 5. Workflow

```
Local (VS Code) ──git push──> GitHub ──git pull──> HPC
                                                     │
                                                     ├─ sbatch scripts/train.sh
                                                     ├─ python eval.py --checkpoint ...
                                                     └─ results → W&B dashboard
```

## 6. Data Transfer (Alternative)

If you need to transfer files directly:

```bash
# From HPC to local (or vice versa)
scp -r /scratch/$USER/checkpoints/best.pt $USER@dtn.torch.hpc.nyu.edu:~/
```

## 7. Key Paths on HPC

| What | Path |
|------|------|
| Dataset | `/scratch/$USER/data/active_matter` |
| Checkpoints | `/scratch/$USER/checkpoints` |
| Code | `/scratch/$USER/dl-active-matter` |
| Overlay | `/scratch/$USER/overlay-dl.ext3` |
| OS image | `/share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif` |
