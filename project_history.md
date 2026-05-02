# Complete Project History — Active Matter Representation Learning

## Phase 1: Infrastructure & Data Pipeline (Apr 14–15)
- Set up conda environment (`dl-active-matter`), `requirements.txt`, `environment.yml`
- Downloaded active_matter dataset from HuggingFace (52GB HDF5 files)
- **Dataset loader** (`data/dataset.py`): Reads 11 physical channels from HDF5, extracts sliding temporal windows, center-crops 256→224
  - Train: 45 files → 175 trajectories
  - Val: 16 files → 64 trajectories  
  - Test: 21 files → 84 trajectories
- Fixed HDF5 loading issues (h5py memory leak, split naming conventions)
- Set up SLURM scripts for NYU Greene HPC (A100 40GB GPUs)

## Phase 2: VideoMAE Baseline (Apr 15–17)
- Implemented `models/mae.py` — SpatioTemporalViT encoder + lightweight decoder
  - 75% tube masking, per-patch MSE reconstruction loss
  - Encoder: ViT-Small (D=384, 6 layers, 13.4M params)
  - Decoder: D'=192, 2 layers
- Implemented `train.py` — training loop with mixed precision, W&B logging
- Trained 100 epochs on A100
- Implemented `eval.py` — frozen encoder → Linear Probe (200ep) + kNN (k=10)
  - Initially had bugs: no feature standardization, only 50 probe epochs

### VideoMAE Results (Initial Eval — Before Eval Fix)
- LP MSE ~0.05–0.07 range (underestimated due to eval bugs)

---

## Phase 3: JEPA v1 (Apr 17)
- Implemented `models/jepa.py` — VideoJEPA architecture
  - Context encoder: processes first 16 of 32 frames
  - Target encoder: EMA copy (momentum=0.996, fixed)
  - Predictor: 3-layer transformer (D'=256, 4 heads)
  - Loss: `MSE(pred, target) + 1.0 × variance_loss`
  - No LR schedule, no gradient clipping, no EMA warmup
- Implemented `train_jepa.py` — separate training loop for JEPA
- Trained 100 epochs on A100

### JEPA v1 Results (Initial Eval)
- LP Best: epoch 40, MSE=0.1168 (but degraded to 0.3976 at epoch 100!)
- kNN Best: epoch 30, MSE=0.0587
- **Diagnosis**: Feature scale drift (σ drops 6×) → LP catastrophically degrades

---

## Phase 4: Evaluation Protocol Fix (Apr 17–19)
- **Critical fix to `eval.py`**:
  - Added z-score feature standardization (StandardScaler)
  - Increased probe epochs: 50 → 200
  - Added cosine LR schedule for probe
  - Added feature statistics logging (mean, std per epoch)
- Re-evaluated ALL existing checkpoints with improved eval

### JEPA v1 Results (Improved Eval)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | 80 | **0.0601** |
| kNN | 30 | **0.0643** |

### VideoMAE Results (Improved Eval)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | 90 | **0.0362** |
| kNN | 90 | **0.0496** |

**Finding: VideoMAE >> JEPA v1** (opposite of baseline paper!)

---

## Phase 5: JEPA v2 — ❌ FAILED (Apr 19)
- Implemented `models/jepa_v2.py` — attempted to close gap with VideoMAE
  - L2-normalized prediction targets + Smooth-L1 loss
  - Full VICReg: variance + covariance on both pred AND target features
  - Aggressive weights: λ_v=5.0, λ_c=0.04
  - Added cosine LR schedule, EMA warmup (0.996→0.999), gradient clipping
- Trained 50 epochs — val_loss monotonically increased from epoch 1

### JEPA v2 Results
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | 5 | 0.1420 (2.4× worse than v1) |
| kNN | 5 | 0.2066 (3.2× worse than v1) |

### Root Cause Analysis
- L2-normalization made pred_loss trivially ≈ 0.001 for entire training
- VICReg (weight 5.0) completely dominated the loss
- Encoder optimized for regularization compliance, not physics

---

## Phase 6: JEPA v3 — Design + HPO (Apr 19–23)
- Implemented `models/jepa_v3.py` — best of v1 loss + v2 stability
  - **Raw MSE prediction loss** (from v1 — NOT L2-normalized)
  - VICReg at reduced weights: λ_v=1.0, λ_c=0.01
  - Cosine LR, EMA warmup, gradient clipping (from v2)
- Implemented `train_jepa_v3.py` with:
  - Early stopping (patience=20, starts after warmup+grace)
  - Gradient accumulation (effective BS=8)
  - `torch.compile()` for kernel fusion
  - `cudnn.benchmark=True`
  - `persistent_workers=True`

### Optuna HPO Sweep (20 Trials)
- TPE sampler + MedianPruner, 15 epochs × 20% data subsample
- ~7.5 GPU-hours total, 8/20 trials pruned

**Key Findings:**
| Parameter | Default | HPO Best | Change |
|---|---|---|---|
| var_weight | 1.0 | **0.257** | 4× lower |
| cov_weight | 0.01 | **0.0095** | ~same |
| lr | 1.5e-4 | **3.4e-4** | 2.3× higher |
| weight_decay | 0.05 | **0.087** | 1.7× higher |

All top-5 trials had var_weight ∈ [0.25, 0.30]. All trials with var_weight ≥ 1.1 were pruned.

---

## Phase 7: JEPA v3 Tuned Training (Apr 23–28)

### Run 1 — Misconfigured (epochs 0–15 only)
- Early stopping fired prematurely (counted warmup epochs)
- LR not batch-size corrected (3.4e-4 for BS=16, should be 1.7e-4 for BS=8)
- Still got LP MSE=0.0801 at epoch 10 — competitive despite bugs

### Run 2 — Fixed (100 epochs)
- Fixed early stopping: starts at epoch 20 (warmup=10 + grace=10)
- LR corrected: 3.4e-4 → 1.7e-4
- Patience increased: 15 → 20 epochs

### JEPA v3 Tuned Results (val set)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | — | 0.0840 |
| kNN | — | 0.0885 |

### JEPA v3 StrongVar (λ_v=0.80) Results (val set)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | — | 0.0865 |
| kNN | — | **0.0670** |

**Finding: VICReg monotonically degrades LP but can help kNN at intermediate weights**

---

## Phase 8: Model Scaling — ViT-Base Experiments (Apr 28)
- Created `configs/jepa_base.yaml` — ViT-Base (D=768, 12 layers, 90.6M encoder params)
- Created `configs/videomae_base.yaml` — same ViT-Base for VideoMAE
- Created `configs/supervised_small.yaml` — end-to-end supervised baseline
- Created `train_supervised.py` — supervised training loop
- Applied speed optimizations to all training scripts:
  - `torch.compile()`, `cudnn.benchmark`, gradient accumulation
  - `float()` casts for YAML config values (fixed TypeError crash)
  - Parameter assertion: checks `encoder_params < 100M` (decoder excluded)
- Launched all 3 experiments on HPC

---

## Phase 9: Evaluation & Analysis (Apr 29–30)

### VideoMAE Base Results (test set, 100 epochs)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | 75 | 0.0805 |
| kNN | 100 | 0.1275 |

### JEPA Base Results (test set, 100 epochs)
| Metric | Best Epoch | MSE |
|---|---|---|
| LP | 95 | 0.0839 |
| kNN | 25 | 0.0926 |
- **Catastrophic kNN collapse**: MSE spikes to 0.75 at epoch 60, recovers to 0.14

### Evaluation Enhancements
- Added kNN k-sweep (k=1,3,5,10,20,50) to `eval.py`
- Added LP LR-sweep (lr=1e-2, 1e-3, 1e-4) to `eval.py`
- Ran detailed sweeps on best checkpoints of all models

### Key Discovery: LP LR=1e-2 is Consistently Better
| Model | LP @ lr=1e-3 | LP @ lr=1e-2 | Improvement |
|---|---|---|---|
| VideoMAE Small | 0.0346 | **0.0237** | 31% better |
| JEPA v1 Small | 0.0579 | **0.0531** | 8% better |

---

## Phase 10: Report Writing (May 1)
- Created ICML 2026 format report (`report/main.tex`)
- 7 sections + appendices with 6 tables
- Generated architecture diagram and placeholder figures
- Expanded Method section with v1/v2/v3 variant definitions
- Expanded Dataset section with complete pipeline description

---

## Final Results Summary (Test Set, Standard Eval: LP lr=1e-3, kNN k=10)

| Model | Params | LP MSE | kNN MSE | Notes |
|---|---|---|---|---|
| **VideoMAE Small** | 13.4M | **0.035** | **0.050** | 🏆 Best overall |
| JEPA v1 Small | 13.4M | 0.058 | 0.088 | Best JEPA variant |
| JEPA v3 (λv=0.26) | 13.4M | 0.084 | 0.089 | HPO-tuned |
| JEPA v3 (λv=0.80) | 13.4M | 0.087 | 0.067 | Best JEPA kNN |
| JEPA v2 | 13.4M | 0.142 | 0.207 | ❌ Failed |
| VideoMAE Base | 90.6M | 0.078 | 0.128 | Scaling hurts |
| JEPA Base | 90.6M | 0.081 | 0.159 | Scaling hurts + collapse |
| Supervised | 13.4M | *pending* | *pending* | Upper bound |

## What's Still Pending
1. ⬜ **Supervised baseline** — needs HPC re-submission (float() bug fixed)
2. ⬜ **Data-driven figures** — collapse trajectories, bar charts from real data
3. ⬜ **t-SNE visualizations** — `scripts/visualize_tsne.py` ready but not run
4. ⬜ **Report finalization** — author names, contributions, supervised numbers
5. ⬜ **Code cleanup** — docstrings, README update for submission

## File Reference

| File | Purpose |
|---|---|
| `models/encoder.py` | SpatioTemporalViT (shared backbone) |
| `models/mae.py` | VideoMAE model |
| `models/jepa.py` | JEPA v1 model |
| `models/jepa_v2.py` | JEPA v2 model (failed) |
| `models/jepa_v3.py` | JEPA v3 model |
| `train.py` | VideoMAE training |
| `train_jepa.py` | JEPA v1 training |
| `train_jepa_v2.py` | JEPA v2 training |
| `train_jepa_v3.py` | JEPA v3 training |
| `train_supervised.py` | Supervised baseline training |
| `eval.py` | Frozen encoder evaluation (LP + kNN + sweeps) |
| `scripts/hpo_sweep.py` | Optuna HPO for JEPA v3 |
| `scripts/parse_logs.py` | Parse SLURM eval logs → tables |
| `scripts/visualize_tsne.py` | t-SNE embedding visualization |
| `configs/*.yaml` | All experiment configurations |
| `EXPERIMENT_LOG.md` | Detailed experiment diary |
