# DL Final Project — Representation Learning for Physical Simulations

NYU CSCI-GA 2572, Spring 2026. Self-supervised representation learning on the `active_matter` physical simulation dataset.

## Project Goal

Train a representation learning model (e.g., JEPA, MAE, contrastive) **from scratch** on spatiotemporal physical simulation data. Evaluate with **linear probing + kNN regression** (MSE) to predict physical parameters α and ζ.

## Dataset — `active_matter`

- **Source**: HuggingFace [`polymathic-ai/active_matter`](https://huggingface.co/datasets/polymathic-ai/active_matter) (~52 GB)
- **Splits**: 8,750 train / 1,200 val / 1,300 test
- **Shape per sample**: `(16, 11, 224, 224)` — 16 time steps, 11 channels, 224×224 spatial
- **11 channels**: concentration scalar (1), velocity vector (2), orientation tensor (4), strain-rate tensor (4)
- **Labels**: α (active dipole strength, 5 values) + ζ (steric alignment, 9 values) — **45 unique combos**
- **Local path**: `/scratch/$NETID/data/active_matter`
- Labels must **not** be used during representation learning; only during linear probe / kNN eval

## Architecture

- Treat samples as **video data**: time = temporal axis, 11 channels ≈ multi-channel frames
- Adapt video methods (Video-JEPA, VideoMAE) directly to 11-channel input
- **Strict limit: < 100M parameters total** (baseline JEPA uses ~3.3M but ~100 GB VRAM at batch 8)
- No pretrained weights of any kind (no ImageNet, CLIP, DINO, VideoMAE, etc.)

## Evaluation Rules (Non-negotiable)

- Evaluation = **frozen encoder** + single linear layer OR kNN, both must be reported
- Regression target: normalized (z-score) α and ζ, metric = **MSE**
- MLPs, attention pooling, or any complex head are **prohibited** for main evaluation
- End-to-end backbone finetuning is **prohibited** for main evaluation (allowed only as a supervised baseline comparison)

## Build / Run Conventions

```bash
# Install deps
pip install -r requirements.txt

# Download dataset (run once on HPC)
huggingface-cli download polymathic-ai/active_matter --local-dir /scratch/$NETID/data/active_matter

# Train (submit as Slurm batch job)
sbatch scripts/train.sh

# Linear probe / kNN eval
python eval.py --checkpoint checkpoints/best.pt --split val
```

## HPC — NYU OOD Burst

- **Portal**: https://ood-burst-001.hpc.nyu.edu/ (NYU VPN required off-campus)
- **Slurm account**: `csci_ga_2572-2026sp` | **Quota**: 300 GPU hours/student
- **GPU partitions**: `g2-standard-12` (1× L4), `g2-standard-24` (2× L4), `c12m85-a100-1` (1× A100 40GB), `n1s8-t4-1` (1× T4)
- **Data transfer node**: `dtn.torch.hpc.nyu.edu`
- Checkpoints → `/scratch/$NETID/checkpoints/`

### Spot Instance Resilience (required in every Slurm script)

```bash
#SBATCH --requeue
```

Always load/save checkpoints so jobs resume after preemption.

## Code Conventions

- **Experiment tracking**: Weights & Biases (`wandb`) — log train/val loss, MSE, representation collapse metrics
- **Reproducibility**: Fix all random seeds; record seeds in logs and `configs/`
- **Config files**: YAML in `configs/` — no magic numbers in training scripts
- **Mixed precision**: use `torch.cuda.amp` to manage VRAM (critical given ~100 GB baseline footprint)
- **Conda + Singularity**: use overlay files from `/share/apps/overlay-fs-ext3`; OS images from `/share/apps/images`

## Repository Layout (target)

```
.
├── configs/           # YAML experiment configs
├── data/              # Dataset loading & preprocessing
├── models/            # Encoder, predictor, and probe architectures
├── scripts/           # Slurm batch scripts (train.sh, eval.sh)
├── eval.py            # Linear probe + kNN evaluation entry point
├── train.py           # Representation learning training entry point
├── requirements.txt
└── ENV.md             # Environment setup instructions
```

## Key References

- Baseline paper: [arXiv:2603.13227](https://arxiv.org/abs/2603.13227)
- EB-JEPA: [GitHub](https://github.com/facebookresearch/ijepa) *(use as reference, not weights)*
- Dataset: [polymathic-ai/active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)
- Report format: ICML 2026 Style Template

## Submission Checklist

- [ ] Parameter count < 100M (report exact count)
- [ ] Fixed seeds documented
- [ ] Both linear probe **and** kNN results on val/test
- [ ] `requirements.txt` + `ENV.md`
- [ ] Configs checked in; command to reproduce
- [ ] Compute accounting (GPUs used, hours, peak VRAM)
- [ ] No references to external pretrained weights in code
- [ ] Statement of Contributions in report
