# Representation Learning for Physical Simulations

NYU CSCI-GA 2572 — Deep Learning, Spring 2026 Final Project

Self-supervised representation learning on the [`active_matter`](https://huggingface.co/datasets/polymathic-ai/active_matter) dataset from The Well, comparing **VideoMAE** and **Video-JEPA** architectures with a shared Vision Transformer backbone.
## Results

Test-set performance (z-score normalized MSE, lower is better):

| Model | Encoder | LP MSE | LP α | LP ζ | kNN MSE |
|-------|---------|--------|------|------|---------|
| Supervised (upper bound) | ViT-S | **0.017** | **0.009** | **0.025** | **0.015** |
| **VideoMAE** | ViT-S | 0.036 | 0.021 | 0.052 | 0.050 |
| JEPA v1 | ViT-S | 0.060 | 0.023 | 0.098 | 0.089 |
| JEPA v3 (λ_v=0.26) | ViT-S | 0.076 | 0.035 | 0.117 | 0.189 |
| JEPA v3 (λ_v=0.80) | ViT-S | 0.079 | 0.043 | 0.115 | 0.075 |
| VideoMAE | ViT-B | 0.081 | 0.036 | 0.125 | 0.149 |
| JEPA v1 | ViT-B | 0.084 | 0.052 | 0.116 | 0.149 |

**Key finding**: VideoMAE consistently outperforms JEPA with ViT encoders — the opposite of prior work using CNN encoders.

---

## Saved Checkpoints

Best checkpoints (selected by best LP MSE on test set):

```
checkpoints/
├── videomae_small/best_eval.pt     # VideoMAE ViT-S (best SSL model)
├── videomae_base/best_eval.pt      # VideoMAE ViT-B (scaling study)
├── jepa_small/best_eval.pt         # JEPA v1 ViT-S
├── jepa_base/best_eval.pt          # JEPA v1 ViT-B (scaling study)
├── jepa_v3_tuned/best_eval.pt      # JEPA v3 (λ_v=0.26, HPO-tuned)
├── jepa_v3_strongvar/best_eval.pt  # JEPA v3 (λ_v=0.80, strong VICReg)
└── supervised_small/best_eval.pt   # End-to-end supervised baseline
```

**Download checkpoints** (for graders):
```bash
# One-command download from Google Drive:
bash scripts/download_checkpoints.sh
```

---

## Quick Start (Reproduce Our Results)

### 1. Environment Setup

```bash
# SSH to NYU HPC, then:
cd /scratch/$USER
git clone https://github.com/ParvaPatel/dl-active-matter.git
cd dl-active-matter

# One-command setup: creates conda env + downloads dataset (~52 GB)
bash scripts/setup.sh
```

See [ENV.md](ENV.md) for detailed environment instructions.

### 2. Train

```bash
# VideoMAE (best model)
sbatch --job-name=vmae-small scripts/train.sh configs/videomae_small.yaml

# JEPA v1
sbatch --job-name=jepa-small scripts/train.sh configs/jepa_small.yaml

# JEPA v3 (HPO-tuned VICReg)
sbatch --job-name=jepa-v3t scripts/train.sh configs/jepa_v3_tuned.yaml

# JEPA v3 (strong VICReg)
sbatch --job-name=jepa-v3s scripts/train.sh configs/jepa_v3_strongvar.yaml

# Supervised baseline (upper bound)
sbatch --job-name=sup-small scripts/train.sh configs/supervised_small.yaml

# Scaled ViT-Base variants
sbatch --job-name=vmae-base scripts/train.sh configs/videomae_base.yaml
sbatch --job-name=jepa-base scripts/train.sh configs/jepa_base.yaml
```

All training auto-resumes from the latest checkpoint on SLURM preemption (`--requeue`).

### 3. Reproduce Reported Results (One Command)

```bash
# Reproduce any model's reported test-set results:
bash scripts/reproduce.sh videomae_small
bash scripts/reproduce.sh jepa_small
bash scripts/reproduce.sh supervised_small

# Check output:
cat logs/reproduce-videomae_small-*.out
```

### 4. Full Evaluation Pipeline

```bash
# Epoch sweep (evaluates every 5th epoch)
bash scripts/eval_videomae_base_sweep.sh

# Collate all results + generate figures
bash scripts/collate_and_plot.sh

# t-SNE / PCA embeddings
sbatch scripts/run_tsne.sh
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
│   ├── jepa_v2.py                   # Video-JEPA v2: L2-norm + Smooth-L1 (failed)
│   └── jepa_v3.py                   # Video-JEPA v3: raw MSE + tuned VICReg
│
├── data/
│   └── dataset.py                   # HDF5 dataset loader (sliding windows, center crop)
│
├── train.py                         # VideoMAE training (AMP, compile, auto-resume)
├── train_jepa.py                    # JEPA v1 training
├── train_jepa_v2.py                 # JEPA v2 training
├── train_jepa_v3.py                 # JEPA v3 training (early stopping, grad accum)
├── train_supervised.py              # Supervised baseline training
├── eval.py                          # Frozen encoder evaluation (LP + kNN + sweeps)
│
├── scripts/
│   ├── train.sh                     # SLURM training job (routes config → entrypoint)
│   ├── eval.sh                      # SLURM evaluation job
│   ├── setup.sh                     # One-command HPC bootstrap
│   ├── hpo_sweep.py                 # Optuna HPO sweep (20 trials, TPE + pruning)
│   ├── hpo_sweep.sh                 # SLURM job for HPO
│   ├── parse_logs.py                # Parse eval logs → results tables
│   ├── visualize.py                 # Training curve plots
│   ├── visualize_tsne.py            # t-SNE embedding visualization
│   └── eval_*_sweep.sh              # Per-experiment evaluation sweeps
│
├── utils/
│   └── training.py                  # Seed fixing, checkpointing, GPU monitoring
│
├── report/                          # ICML 2026 format paper
│   ├── main.tex
│   └── references.bib
│
├── EXPERIMENT_LOG.md                # Detailed experiment diary with all results
├── ENV.md                           # HPC environment setup guide
├── environment.yml                  # Conda environment spec
└── requirements.txt                 # pip dependencies
```

## Method

### Shared Encoder

All models use a **SpatioTemporalViT** encoder processing `(16, 11, 224, 224)` inputs:
- **Tokenization**: 3D patch embedding `(2, 16, 16)` → 1,568 tokens per sample
- **ViT-Small**: D=384, 6 layers, 6 heads (13.4M params)
- **ViT-Base**: D=768, 12 layers, 12 heads (90.6M params)

### VideoMAE
Mask 75% of spatiotemporal tube patches → reconstruct pixel values. Dense pixel-level supervision forces fine-grained feature learning.

### Video-JEPA (3 variants)

| Variant | Pred. Loss | VICReg | Result |
|---------|-----------|--------|--------|
| **v1** | Raw MSE | None | Best LP (0.058) |
| **v2** | Smooth-L1 on L2-normed targets | λ_v=5.0, λ_c=0.04 | ❌ Failed (0.142) |
| **v3** | Raw MSE | λ_v∈{0.26, 0.80}, λ_c=0.01 | Intermediate |

v2 failed because L2-normalization made prediction trivially easy (pred_loss ≈ 0.001), letting VICReg dominate. v3 corrects this.

### Evaluation Protocol
1. **Freeze** encoder after self-supervised pretraining
2. **Extract** mean-pooled features, z-score normalize
3. **Linear Probe**: `nn.Linear(D, 2)`, 200 epochs, Adam, cosine LR
4. **kNN**: distance-weighted, k=10, scikit-learn

Targets: z-score normalized α (active dipole) and ζ (steric alignment). Metric: **MSE**.

## Dataset

| Property | Value |
|----------|-------|
| Name | `active_matter` from [The Well](https://github.com/PolymathicAI/the_well) |
| Size | ~52 GB (HDF5 files) |
| Train / Val / Test | 45 / 16 / 21 HDF5 files (175 / 64 / 84 trajectories) |
| Channels | concentration (1), velocity (2), orientation (4), strain-rate (4) = 11 |
| Resolution | 81 time steps × 256×256 (center-cropped to 224×224) |
| Labels | α ∈ {0.5, 1.0, 1.5, 2.0, 2.5} × ζ ∈ {0.1, …, 0.9} = 45 combos |
| Samples (16-frame) | 11,550 train / 1,584 val / 1,716 test |
| Samples (32-frame, JEPA) | 8,750 train / 1,200 val / 1,300 test |

## Experiment Tracking

- **W&B**: Training metrics logged to Weights & Biases (optional — set `WANDB_API_KEY` in `.env`)
- **Checkpoints**: Saved every 5 epochs + `best.pt` + `latest.pt` to `/scratch/$USER/checkpoints/<experiment>/`
- **Seeds**: All experiments use seed 42 (configurable in YAML configs)
- **Configs**: All hyperparameters in `configs/*.yaml` — zero magic numbers in code
- **Mixed precision**: FP16 via `torch.amp` + `torch.compile()` for kernel fusion

## Compute

| Resource | Detail |
|----------|--------|
| Cluster | NYU HPC Greene (OOD Burst, Google Cloud) |
| GPU | NVIDIA A100-SXM4-40GB |
| Total GPU hours | ~90.5h across all experiments |
| Training (per model) | ~8–15h for 100 epochs |
| Eval (per checkpoint) | ~5 min (feature extraction + probe + kNN) |
| HPO sweep | ~7.5h (20 trials × 15 epochs × 20% data) |

## Reproducibility Checklist

- [x] All hyperparameters in version-controlled YAML configs
- [x] Fixed random seeds (torch, numpy, cuda)
- [x] `assert encoder_params < 100_000_000` enforced at training start
- [x] Auto-resume from checkpoints on SLURM preemption
- [x] Feature z-score normalization in evaluation
- [x] All results reproducible from configs + training scripts
- [x] Environment specified in `requirements.txt` + `environment.yml`
- [x] Detailed experiment log in `EXPERIMENT_LOG.md`

## References

1. Qu & Cranmer. *Representation Learning for Physical Simulations*. [arXiv:2603.13227](https://arxiv.org/abs/2603.13227)
2. Tong et al. *VideoMAE: Masked Autoencoders are Data-Efficient Learners*. NeurIPS 2022.
3. Assran et al. *Self-Supervised Learning from Images with a JEPA*. CVPR 2023.
4. Bardes et al. *VICReg: Variance-Invariance-Covariance Regularization*. ICLR 2022.
5. Bardes et al. *Revisiting Feature Prediction for Learning Visual Representations from Video*. 2024.
6. The Well Dataset: [polymathic-ai/active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)

## License

Academic use only — NYU CSCI-GA 2572 Final Project, Spring 2026.
