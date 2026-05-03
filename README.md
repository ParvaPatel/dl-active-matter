# Representation Learning for Physical Simulations

**NYU CSCI-GA 2572 — Deep Learning, Spring 2026 Final Project**

Self-supervised representation learning on the [`active_matter`](https://huggingface.co/datasets/polymathic-ai/active_matter) dataset from [The Well](https://github.com/PolymathicAI/the_well), comparing **VideoMAE** and **Video-JEPA** architectures with a shared Vision Transformer backbone.

**Authors**: Parva Patel (psp8502), Mohith Niranjen R K (mr7841), Surbhi (xs2682)

---

## Results

Test-set performance (z-score normalized MSE, lower is better):

| Model | Encoder | LP MSE | LP α | LP ζ | kNN MSE | kNN α | kNN ζ |
|-------|---------|--------|------|------|---------|-------|-------|
| Supervised (upper bound) | ViT-S | **0.018** | **0.009** | **0.027** | **0.015** | **0.005** | **0.025** |
| **VideoMAE** | ViT-S | **0.044** | 0.023 | 0.066 | **0.050** | 0.021 | 0.078 |
| JEPA v1 | ViT-S | 0.061 | 0.027 | 0.096 | 0.089 | 0.038 | 0.140 |
| JEPA v3 (λ_v=0.26, tuned) | ViT-S | 0.092 | 0.036 | 0.149 | 0.189 | 0.196 | 0.182 |
| JEPA v3 (λ_v=0.80, strong) | ViT-S | 0.091 | 0.054 | 0.129 | 0.075 | 0.006 | 0.144 |
| VideoMAE | ViT-B | 0.083 | 0.037 | 0.130 | 0.149 | 0.078 | 0.220 |
| JEPA v1 | ViT-B | 0.085 | 0.046 | 0.125 | 0.149 | 0.065 | 0.232 |

**Key findings**:
- VideoMAE consistently outperforms JEPA with ViT encoders — the opposite of prior work using CNN encoders
- VICReg regularization monotonically degrades linear probing performance
- Scaling from 13M → 91M parameters worsens representations by 1.4–3× with limited data

### 📊 Experiment Tracking

All training runs are logged to Weights & Biases with full loss curves, gradient norms, and feature statistics:

**[W&B Dashboard →](https://wandb.ai/psp8502-new-york-university/dl-active-matter)**

---

## For Graders: Quick Reproduce

> **Prerequisites**: Access to NYU HPC with `csci_ga_2572-2026sp` Slurm account and dataset already downloaded to `/scratch/$USER/data/active_matter`.

### Step 1: Clone & Setup

```bash
cd /scratch/$USER
git clone https://github.com/ParvaPatel/dl-active-matter.git
cd dl-active-matter

# Full automated setup (overlay + conda + dataset download)
bash scripts/setup.sh
```

### Step 2: Download Pre-trained Checkpoints

```bash
bash scripts/download_checkpoints.sh
```

This downloads all 8 model checkpoints to `/scratch/$USER/checkpoints/` from [**Google Drive**](https://drive.google.com/drive/folders/1o2nbCbxdOdfLvs7bTWUwU_iuOsf4RqGK).

### Step 3: Reproduce Any Model's Results

```bash
# Reproduce reported test-set numbers for any model:
bash scripts/reproduce.sh videomae_small       # Best SSL model
bash scripts/reproduce.sh jepa_small           # JEPA v1
bash scripts/reproduce.sh jepa_v3_tuned        # JEPA v3 (HPO-tuned)
bash scripts/reproduce.sh jepa_v3_strongvar    # JEPA v3 (strong VICReg)
bash scripts/reproduce.sh supervised_small     # Supervised upper bound
bash scripts/reproduce.sh videomae_base        # Scaling study
bash scripts/reproduce.sh jepa_base            # Scaling study

# Check results:
cat logs/reproduce-videomae_small-*.out
```

Each `reproduce.sh` run submits a SLURM job that evaluates the frozen encoder with both linear probing and kNN regression on the test set.

---

## Environment Setup

See [ENV.md](ENV.md) for full detailed instructions. Summary:

### Automated (Recommended)

```bash
cd /scratch/$USER
git clone https://github.com/ParvaPatel/dl-active-matter.git
cd dl-active-matter
bash scripts/setup.sh
```

`setup.sh` handles:
1. Creates a 15 GB Singularity overlay filesystem
2. Installs Miniconda inside the overlay
3. Creates the `dl-active-matter` conda environment with all dependencies
4. Downloads the `active_matter` dataset (~52 GB) from HuggingFace

### Manual Setup

#### 1. Singularity Overlay

```bash
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/overlay-dl.ext3.gz
gunzip /scratch/$USER/overlay-dl.ext3.gz

singularity exec --nv \
  --overlay /scratch/$USER/overlay-dl.ext3:rw \
  /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
  /bin/bash
```

#### 2. Conda Environment

```bash
# Inside Singularity shell:
conda env create -f environment.yml
conda activate dl-active-matter
pip install -r requirements.txt
```

#### 3. Download Dataset

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='polymathic-ai/active_matter',
    repo_type='dataset',
    local_dir='/scratch/\$USER/data/active_matter'
)
"
```

### Software Versions

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10+ | Runtime |
| PyTorch | ≥2.1.0 | Training, `torch.compile()` |
| CUDA | 12.1 | GPU acceleration |
| h5py | ≥3.9.0 | HDF5 dataset loading |
| scikit-learn | ≥1.3.0 | kNN evaluation |
| wandb | ≥0.15.0 | Experiment tracking |
| optuna | ≥3.4.0 | Hyperparameter optimization |

---

## Scripts Reference

### Training Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `scripts/train.sh` | Main SLURM training job. Routes config to the correct training entrypoint based on the `method` field in the YAML config. | `sbatch --job-name=<name> scripts/train.sh configs/<config>.yaml` |
| `scripts/test_1epoch.sh` | Quick 1-epoch sanity check on a small subset. | `sbatch scripts/test_1epoch.sh` |
| `scripts/hpo_sweep.sh` | Submits a 20-trial Optuna hyperparameter optimization sweep. | `sbatch scripts/hpo_sweep.sh` |

### Evaluation Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `scripts/eval.sh` | Evaluates a single checkpoint (LP + kNN) on a given split. | `sbatch scripts/eval.sh <checkpoint_path> <split> [--use_target_encoder]` |
| `scripts/reproduce.sh` | One-command reproduction of reported results for any model. | `bash scripts/reproduce.sh <model_name>` |
| `scripts/eval_*_sweep.sh` | Per-experiment epoch sweeps (evaluates every checkpoint). | `bash scripts/eval_videomae_base_sweep.sh` |

### Analysis & Visualization Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `scripts/collate_and_plot.sh` | Parses all eval logs, generates CSV tables and figures. | `bash scripts/collate_and_plot.sh` |
| `scripts/collate_results.sh` | Lightweight version: collates logs into `test_results.txt`. | `bash scripts/collate_results.sh` |
| `scripts/parse_logs.py` | Parses raw eval log files into structured tables. | `python scripts/parse_logs.py` |
| `scripts/log_result.py` | Appends a single eval result to the results log. | `python scripts/log_result.py <log_file>` |
| `scripts/generate_figures.py` | Generates all report figures from collated data. | `python scripts/generate_figures.py` |
| `scripts/visualize.py` | Plots training loss curves from W&B or log files. | `python scripts/visualize.py` |
| `scripts/visualize_tsne.py` | Generates t-SNE/PCA embedding visualizations. | `python scripts/visualize_tsne.py --checkpoint <path>` |
| `scripts/visualize_dataset.py` | Visualizes raw dataset samples and channel fields. | `python scripts/visualize_dataset.py` |
| `scripts/run_tsne.sh` | SLURM job for generating t-SNE plots for all models. | `sbatch scripts/run_tsne.sh` |
| `scripts/run_dataviz.sh` | SLURM job for dataset visualization. | `sbatch scripts/run_dataviz.sh` |

### Setup & Utility Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `scripts/setup.sh` | Full automated HPC environment bootstrap. | `bash scripts/setup.sh` |
| `scripts/download_checkpoints.sh` | Downloads pre-trained checkpoints from Google Drive. | `bash scripts/download_checkpoints.sh` |
| `scripts/inspect_data.py` | Inspects HDF5 dataset structure and field shapes. | `python scripts/inspect_data.py` |

### Available Models for `reproduce.sh`

```
videomae_small       # Best SSL model (LP MSE: 0.036)
videomae_base        # ViT-Base scaling study
jepa_small           # JEPA v1 baseline
jepa_base            # JEPA v1 ViT-Base scaling study
jepa_v2              # JEPA v2 (failed experiment — documented)
jepa_v3_tuned        # JEPA v3 with HPO-tuned VICReg (λ_v=0.26)
jepa_v3_strongvar    # JEPA v3 with strong VICReg (λ_v=0.80)
supervised_small     # End-to-end supervised upper bound
```

---

## Project Structure

```
.
├── configs/                         # YAML experiment configs (all hyperparams)
│   ├── videomae_small.yaml          # VideoMAE ViT-Small (best model)
│   ├── videomae_base.yaml           # VideoMAE ViT-Base (scaling study)
│   ├── jepa_small.yaml              # JEPA v1 ViT-Small
│   ├── jepa_base.yaml               # JEPA v1 ViT-Base (scaling study)
│   ├── jepa_v2.yaml                 # JEPA v2 (failed — documented)
│   ├── jepa_v3.yaml                 # JEPA v3 default
│   ├── jepa_v3_tuned.yaml           # JEPA v3 HPO-optimized
│   ├── jepa_v3_strongvar.yaml       # JEPA v3 high VICReg
│   └── supervised_small.yaml        # End-to-end supervised baseline
│
├── models/                          # Architecture implementations
│   ├── encoder.py                   # SpatioTemporalViT (shared backbone)
│   ├── mae.py                       # VideoMAE: tube masking + pixel decoder
│   ├── jepa.py                      # Video-JEPA v1: temporal prediction + EMA
│   ├── jepa_v2.py                   # Video-JEPA v2: L2-norm targets (failed)
│   └── jepa_v3.py                   # Video-JEPA v3: raw MSE + tuned VICReg
│
├── data/
│   └── dataset.py                   # HDF5 dataset loader (sliding window, center crop)
│
├── utils/
│   └── training.py                  # Seed fixing, checkpointing, GPU monitoring
│
├── train.py                         # VideoMAE training loop
├── train_jepa.py                    # JEPA v1 training loop
├── train_jepa_v2.py                 # JEPA v2 training loop
├── train_jepa_v3.py                 # JEPA v3 training loop (early stopping, grad accum)
├── train_supervised.py              # Supervised baseline training loop
├── eval.py                          # Frozen-encoder evaluation (LP + kNN + sweeps)
│
├── scripts/                         # SLURM jobs, analysis, and visualization
├── logs/                            # Training & evaluation output logs
├── report/                          # ICML 2026 format paper (main.tex + references.bib)
│
├── ENV.md                           # Detailed HPC environment setup guide
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # pip dependency list
├── test_results.txt                 # Collated test-set evaluation results
└── README.md                        # This file
```

---

## Method

### Shared Encoder: SpatioTemporalViT

All models use a **SpatioTemporalViT** encoder processing inputs of shape `(T=16, C=11, H=224, W=224)`:
- **3D patch embedding**: patch size `(2, 16, 16)` → 1,568 tokens per sample
- **ViT-Small**: D=384, 6 layers, 6 heads (13.4M encoder params)
- **ViT-Base**: D=768, 12 layers, 12 heads (90.6M encoder params)

### VideoMAE

Masks 75% of spatiotemporal tube patches and reconstructs pixel values via a lightweight decoder (2–3 transformer layers). Dense pixel-level supervision forces fine-grained feature learning.

### Video-JEPA (3 evolutionary variants)

| Variant | Prediction Loss | VICReg | Target Encoder | Key Insight |
|---------|----------------|--------|----------------|-------------|
| **v1** | Raw MSE | None | EMA (τ=0.996) | Best LP — prediction alone suffices |
| **v2** | Smooth-L1 on L2-normed targets | λ_v=5.0, λ_c=0.04 | EMA | ❌ Failed — pred loss ~0.003, VICReg dominated |
| **v3** | Raw MSE | λ_v∈{0.26, 0.80}, λ_c=0.01 | EMA + cosine warmup | VICReg hurts LP, but kNN benefits at λ_v=0.80 |

### Evaluation Protocol

1. **Freeze** encoder after self-supervised pretraining
2. **Extract** mean-pooled features → z-score normalize (using train set statistics)
3. **Linear Probe**: `nn.Linear(D, 2)` → predict [α, ζ], 200 epochs, Adam, cosine LR, MSE loss
4. **kNN Regression**: distance-weighted k=10, scikit-learn `KNeighborsRegressor`

---

## Dataset

| Property | Value |
|----------|-------|
| Name | `active_matter` from [The Well](https://github.com/PolymathicAI/the_well) |
| Source | [polymathic-ai/active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter) |
| Size | ~52 GB (HDF5 files) |
| Splits | 45 train / 16 val / 21 test HDF5 files |
| Trajectories | 175 train / 64 val / 84 test |
| Channels | concentration (1) + velocity (2) + orientation tensor (4) + strain-rate tensor (4) = **11** |
| Resolution | 81 time steps × 256×256 → center-cropped to 224×224 |
| Labels | α ∈ {-1, -2, -3, -4, -5} × ζ ∈ {1, 3, 5, 7, 9, 11, 13, 15, 17} = **45 combos** |
| Samples (n=16, VideoMAE) | 11,550 train / 1,584 val / 1,716 test |
| Samples (n=32, JEPA) | 8,750 train / 1,200 val / 1,300 test |

---

## Compute

| Resource | Detail |
|----------|--------|
| Cluster | NYU HPC Greene (OOD Burst, Google Cloud) |
| GPU | NVIDIA A100-SXM4-40GB |
| Total GPU hours | **255h** across all experiments |
| Mixed precision | FP16 via `torch.amp` + `torch.compile()` |
| Seeds | All experiments: seed 42 |
| SLURM account | `csci_ga_2572-2026sp` |

---

## Reproducibility Checklist

- [x] All hyperparameters in version-controlled YAML configs
- [x] Fixed random seeds (torch, numpy, cuda) — seed 42
- [x] Model size < 100M params enforced (max: 98.28M for JEPA-Base)
- [x] Auto-resume from checkpoints on SLURM preemption (`--requeue`)
- [x] Feature and target z-score normalization in evaluation
- [x] All results reproducible via `scripts/reproduce.sh`
- [x] Environment specified in `requirements.txt` + `environment.yml` + `ENV.md`
- [x] W&B experiment tracking for all training runs
- [x] No pretrained weights — all models trained from scratch

---

## References

1. Qu et al. *Representation Learning for Spatiotemporal Physical Systems*. [arXiv:2603.13227](https://arxiv.org/abs/2603.13227), 2025.
2. Tong et al. *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*. NeurIPS 2022.
3. Assran et al. *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*. CVPR 2023.
4. Bardes et al. *VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning*. ICLR 2022.
5. Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*. arXiv 2024.
6. Ohana et al. *The Well: A Large-Scale Collection of Diverse Physics Simulations for Machine Learning*. NeurIPS D&B 2024.

---

## License

Academic use only — NYU CSCI-GA 2572 Final Project, Spring 2026.
