# Representation Learning for Physical Simulations

NYU CSCI-GA 2572 — Deep Learning, Spring 2026 Final Project

Self-supervised representation learning on the [`active_matter`](https://huggingface.co/datasets/polymathic-ai/active_matter) physical simulation dataset using **VideoMAE** (Masked Autoencoder) and **Video-JEPA** architectures.

## Quick Start (Reproduce Our Results)

### 1. HPC Environment Setup

```bash
# SSH into NYU HPC or use OOD: https://ood-burst-001.hpc.nyu.edu/

# Clone the repo
cd /scratch/$USER
git clone https://github.com/<YOUR-USERNAME>/dl-active-matter.git
cd dl-active-matter

# One-command setup: creates conda env + downloads dataset
bash scripts/setup.sh
```

### 2. Train

```bash
# Submit training job (A100, auto-resumes on spot preemption)
sbatch scripts/train.sh
```

### 3. Evaluate

```bash
# Linear probe + kNN on frozen encoder
sbatch scripts/eval.sh

# Or run interactively
python eval.py --checkpoint /scratch/$USER/checkpoints/best.pt --split val
```

## Project Structure

```
.
├── configs/                    # YAML experiment configs (no magic numbers)
│   └── videomae_small.yaml     # Baseline VideoMAE config (~10M params)
├── data/                       # Dataset loading & preprocessing
│   └── dataset.py              # ActiveMatter HuggingFace Arrow loader
├── models/                     # Architecture implementations
│   ├── encoder.py              # SpatioTemporalViT with tube patch embedding
│   └── mae.py                  # VideoMAE: masking, decoder, reconstruction loss
├── scripts/                    # Slurm batch scripts & automation
│   ├── train.sh                # Training job (A100, --requeue for spot instances)
│   ├── eval.sh                 # Evaluation job (L4)
│   └── setup.sh                # One-command HPC environment bootstrap
├── utils/
│   └── training.py             # Seed fixing, checkpoint save/load
├── train.py                    # Training entry point (AMP, W&B, auto-resume)
├── eval.py                     # Linear probe + kNN evaluation entry point
├── environment.yml             # Conda environment (exact reproducibility)
├── requirements.txt            # pip dependencies
└── ENV.md                      # Detailed HPC environment setup guide
```

## Method

### Architecture

We treat each simulation sample as a **video**: 16 time steps × 11 physical channels × 224×224 spatial resolution.

**Tokenization**: 3D tube patches of size `(2, 16, 16)` — 2 frames × 16×16 pixels — producing **784 tokens** per sample.

**Encoder**: Spatiotemporal Vision Transformer (ViT-Small variant):
- Embedding dimension: 384
- Depth: 6 transformer blocks
- Heads: 6
- **~9.5M parameters** (well under 100M limit)

**Self-supervised objective**: Masked reconstruction (VideoMAE) — mask 75% of tube patches, reconstruct pixel values from visible patches only.

### Evaluation

Frozen encoder → mean-pooled features → evaluated with:
1. **Linear probe**: Single `nn.Linear(384, 2)` trained on frozen features
2. **kNN regression**: `k=10`, distance-weighted, scikit-learn

Target: z-score normalized α (active dipole strength) and ζ (steric alignment), metric = **MSE**.

## Dataset

| Property | Value |
|----------|-------|
| Name | `active_matter` from The Well |
| Size | ~52 GB |
| Splits | 8,750 train / 1,200 val / 1,300 test |
| Shape | `(16, 11, 224, 224)` per sample |
| Channels | concentration (1), velocity (2), orientation (4), strain-rate (4) |
| Labels | α (5 values) × ζ (9 values) = 45 combos |

## Results

| Method | Total MSE | α MSE | ζ MSE |
|--------|-----------|-------|-------|
| Linear Probe | — | — | — |
| kNN (k=10) | — | — | — |
| Supervised baseline* | — | — | — |

*End-to-end finetuned; for comparison only, not our main submission.

## Reproducibility

- **Seeds**: All experiments use seed `42` (configurable in YAML). Seeds are logged to W&B.
- **Configs**: All hyperparameters live in `configs/` — no magic numbers in code.
- **Checkpoints**: Saved every epoch to `/scratch/$USER/checkpoints/` for spot-instance resilience.
- **Mixed precision**: `torch.cuda.amp` (FP16) used throughout training.
- **Parameter count**: Verified < 100M at training start (assertion in `train.py`).

## Compute

| Resource | Detail |
|----------|--------|
| Cluster | NYU HPC OOD Burst (Google Cloud) |
| Slurm account | `csci_ga_2572-2026sp` |
| GPU | NVIDIA A100 40GB (training), L4 24GB (eval) |
| Training time | — |
| Total GPU hours | — |
| Peak VRAM | — |

## Team

| Member | Contribution |
|--------|-------------|
| — | — |
| — | — |
| — | — |

## References

1. Baseline paper: [arXiv:2603.13227](https://arxiv.org/abs/2603.13227)
2. EB-JEPA: [GitHub](https://github.com/facebookresearch/ijepa)
3. Dataset: [polymathic-ai/active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)
4. VideoMAE: [He et al., 2022](https://arxiv.org/abs/2203.12602)

## License

Academic use only — NYU CSCI-GA 2572 Final Project.
