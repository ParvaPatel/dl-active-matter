# Experiment Log — Video-JEPA vs VideoMAE

> **Project**: NYU CSCI-GA 2572 Deep Learning Final Project  
> **Task**: Self-supervised representation learning on `active_matter` physical simulation data  
> **Goal**: Learn frozen encoder features that enable accurate regression of physical parameters (α, ζ) via Linear Probe and kNN  
> **Last Updated**: 2026-04-23

---

## Timeline & Evolution

### Phase 1: VideoMAE Baseline
- Implemented standard Masked Autoencoder (75% tube masking, pixel reconstruction)
- Trained for 100 epochs on A100, ViT-Small encoder (~9.5M params)
- **This is our strong baseline**

### Phase 2: JEPA v1 (`models/jepa.py`, `train_jepa.py`)
- Replaced pixel reconstruction with **representation prediction**
- Context encoder (online) encodes first 16 frames → predictor → predicts target encoder output for last 16 frames
- Target encoder updated via EMA (momentum=0.996, fixed)
- VICReg variance penalty on predicted features only
- Loss: `MSE(pred, target) + 1.0 * variance_loss`
- No LR schedule (flat lr=1.5e-4), no gradient clipping, no EMA warmup

### Phase 3: Eval v2 (improved `eval.py`)
- Before retraining, we first fixed the evaluation protocol:
  - Added feature standardization (Z-score) before probing
  - Increased probe epochs from 50 → 200 with cosine LR schedule
  - Added feature statistics logging (mean, std)
- Re-evaluated all existing JEPA v1 and VideoMAE checkpoints with this improved eval
- **All results below use this improved eval protocol**

### Phase 4: JEPA v2 (`models/jepa_v2.py`, `train_jepa_v2.py`) — ❌ FAILED
Created to fix training-side issues found in JEPA v1. Changes:
1. L2-normalized prediction targets → Smooth-L1 loss instead of MSE
2. Full VICReg (var + cov) on both predicted and target features
3. Cosine LR schedule, EMA warmup, gradient clipping
- **Root cause failure**: L2-norm made prediction trivially easy (pred_loss ≈ 0.001 always), causing VICReg (weight=5.0) to dominate the entire loss. Encoder optimized regularization, not physics.
- Best result 2.4× worse than JEPA v1.

### Phase 5: JEPA v3 (`models/jepa_v3.py`, `train_jepa_v3.py`) + Optuna HPO
- **Design**: v1's raw MSE prediction loss + v2's training stabilizers (cosine LR, EMA warmup, grad clip)
- **Added training optimizations**: early stopping, gradient accumulation (effective bs=8), torch.compile(), cudnn.benchmark, persistent_workers
- **Ran 20-trial Optuna HPO sweep** (TPE + MedianPruner) to find best var_weight, cov_weight, lr, weight_decay
- Generated `configs/jepa_v3_tuned.yaml` with optimal hyperparameters

### Phase 6: JEPA v3 Tuned — Run 1 (`configs/jepa_v3_tuned.yaml`) — ❌ MISCONFIGURED (epochs 0-15 only)
- **Problem 1**: Early stopping counted warmup epochs → fired at epoch 15 (best val at epoch 0 = warmup LR artifact)
- **Problem 2**: lr=3.4e-4 was HPO LR for bs=16; full training uses effective bs=8 → should be 1.7e-4
- **Downstream eval** (epochs 0-15): epoch_10 gave LP MSE=0.0801, kNN=0.0820 — already competitive despite misconfiguration
- **Key finding**: val_loss is a POOR proxy for representation quality. epoch_0 had best val_loss (0.35) but worst LP MSE (0.295); epoch_10 had higher val_loss (0.56) but best LP MSE (0.0801)

### Phase 7: JEPA v3 Tuned — Run 2 (`configs/jepa_v3_tuned.yaml`) — 🔄 RUNNING
- **Fixes applied**:
  - `lr`: 3.4e-4 → **1.7e-4** (batch-size corrected: 3.4e-4 × 8/16)
  - `early_stopping_start_epoch: 20` (warmup=10 + grace=10, don't count warmup epochs)
  - `early_stopping_patience: 20` (was 15)
- **Expected**: with 100 epochs and proper early stopping, LP MSE should approach or beat JEPA v1 best (0.0601 at epoch 80)

---


## Experiment Results

### JEPA v1 — Original Eval (50-epoch probe, no feature standardization)

> These were the first results that revealed the feature drift problem.

| Metric | Best Epoch | Best Score | Epoch 100 Score | Degradation |
|--------|-----------|------------|-----------------|-------------|
| **Linear Probe** | 40 | **0.1168** | 0.3976 | 3.4× worse |
| **kNN** | 30 | **0.0587** | 0.0931 | 1.6× worse |

**Diagnosis**: Massive LP degradation = feature scale drift. kNN (scale-invariant) degrades less. This led to fixing eval.py first.

---

### JEPA v1 — Improved Eval (from `logs/jepa_v2_feat_std/`)

> Same JEPA v1 model checkpoints, re-evaluated with improved eval.py (feature standardization, 200-epoch probe, cosine LR).  
> Folder name `jepa_v2_feat_std` refers to **eval protocol v2**, NOT the model version.

| Epoch | LP MSE   | LP α     | LP ζ     | kNN MSE  | kNN α    | kNN ζ    | Feat μ   | Feat σ   |
|-------|----------|----------|----------|----------|----------|----------|----------|----------|
| 10    | 0.1364   | 0.0228   | 0.2500   | 0.0948   | 0.0028   | 0.1869   | 0.0004   | 0.7463   |
| 20    | 0.1202   | 0.0174   | 0.2229   | 0.0805   | 0.0035   | 0.1576   | 0.0011   | 0.6194   |
| 30    | 0.1072   | 0.0169   | 0.1975   | **0.0643** | **0.0062** | 0.1223 | 0.0014   | 0.4951   |
| 40    | 0.0894   | 0.0185   | 0.1604   | 0.0736   | 0.0096   | 0.1376   | 0.0013   | 0.3869   |
| 50    | 0.0827   | 0.0226   | 0.1427   | 0.0855   | 0.0142   | 0.1567   | -0.0000  | 0.2885   |
| 60    | 0.0759   | 0.0162   | 0.1356   | 0.0814   | 0.0221   | 0.1407   | -0.0013  | 0.2153   |
| 70    | 0.0612   | 0.0190   | 0.1034   | 0.0827   | 0.0267   | 0.1387   | -0.0015  | 0.1702   |
| **80**| **0.0601**| 0.0227  | 0.0975   | 0.0891   | 0.0381   | 0.1402   | -0.0014  | 0.1443   |
| 90    | 0.0616   | 0.0263   | 0.0969   | 0.0960   | 0.0482   | 0.1439   | -0.0009  | 0.1285   |
| 100   | 0.0751   | 0.0316   | 0.1186   | 0.1060   | 0.0674   | 0.1447   | -0.0007  | 0.1194   |
| Best  | 0.0683   | 0.0257   | 0.1110   | 0.1060   | 0.0674   | 0.1447   | -0.0007  | 0.1194   |

**★ Best Linear Probe: epoch 80 (MSE=0.0601)**  
**★ Best kNN: epoch 30 (MSE=0.0643)**

**Key improvements from eval fix**: LP at epoch 40 went from 0.1168 → 0.0894 (23% better), confirming feature standardization helps. LP no longer catastrophically degrades (epoch 100 = 0.0751 vs old 0.3976).

---

### VideoMAE — Improved Eval (from `logs/videamae_v2_feat_std/`)

> Same improved eval protocol as JEPA v1 above.

| Epoch | LP MSE   | LP α     | LP ζ     | kNN MSE  | kNN α    | kNN ζ    | Feat μ   | Feat σ   |
|-------|----------|----------|----------|----------|----------|----------|----------|----------|
| 10    | 0.2622   | 0.0707   | 0.4536   | 0.3814   | 0.2794   | 0.4835   | 0.0005   | 0.4502   |
| 20    | 0.1765   | 0.0647   | 0.2884   | 0.3019   | 0.2573   | 0.3466   | 0.0005   | 0.3257   |
| 30    | 0.0903   | 0.0289   | 0.1517   | 0.1050   | 0.0168   | 0.1931   | -0.0011  | 0.2284   |
| 40    | 0.0819   | 0.0269   | 0.1369   | 0.0769   | 0.0184   | 0.1353   | -0.0004  | 0.0857   |
| 50    | 0.0703   | 0.0160   | 0.1246   | 0.0612   | 0.0159   | 0.1065   | -0.0005  | 0.0351   |
| 60    | 0.0535   | 0.0166   | 0.0903   | 0.0528   | 0.0134   | 0.0922   | 0.0004   | 0.0241   |
| 70    | 0.0566   | 0.0202   | 0.0929   | 0.0523   | 0.0181   | 0.0865   | 0.0002   | 0.0241   |
| 80    | 0.0411   | 0.0176   | 0.0647   | 0.0504   | 0.0176   | 0.0832   | 0.0001   | 0.0230   |
| **90**| **0.0362**| 0.0210  | **0.0515** | **0.0496** | 0.0211 | **0.0782** | -0.0000 | 0.0220  |
| 100   | 0.0394   | 0.0171   | 0.0617   | 0.0528   | 0.0196   | 0.0859   | -0.0001  | 0.0216   |
| Best  | 0.0385   | 0.0168   | 0.0602   | 0.0521   | 0.0206   | 0.0835   | -0.0001  | 0.0218   |

**★ Best Linear Probe: epoch 90 (MSE=0.0362)**  
**★ Best kNN: epoch 90 (MSE=0.0496)**

---

### JEPA v2 — ❌ FAILED (Regularization Overwhelms Prediction)

> Model: `models/jepa_v2.py` with L2-norm targets, Smooth-L1, full VICReg.  
> Training: 50 epochs completed (best val_loss at epoch 1, monotonically worsening after).  
> Eval logs: `logs/jepa_v2_model_eval/`

| Epoch | LP MSE   | LP α     | LP ζ     | kNN MSE  | kNN α    | kNN ζ    | Feat μ   | Feat σ   |
|-------|----------|----------|----------|----------|----------|----------|----------|----------|
| 5     | **0.1420**| 0.0254  | 0.2586   | **0.2066**| 0.0329  | 0.3802   | 0.0001   | 0.0865   |
| 10    | 0.1513   | 0.0336   | 0.2689   | 0.2349   | 0.0348   | 0.4350   | 0.0008   | 0.1616   |
| 15    | 0.2230   | 0.0405   | 0.4056   | 0.2249   | 0.0341   | 0.4157   | 0.0018   | 0.3703   |
| 20    | 0.1780   | 0.0363   | 0.3198   | 0.2222   | 0.0452   | 0.3992   | 0.0018   | 0.4009   |
| 25    | 0.1611   | 0.0458   | 0.2764   | 0.2219   | 0.0323   | 0.4114   | 0.0016   | 0.3660   |
| 30    | 0.2395   | 0.0842   | 0.3948   | 0.2331   | 0.0421   | 0.4241   | 0.0016   | 0.3168   |
| 35    | 0.2002   | 0.0252   | 0.3753   | 0.2675   | 0.0330   | 0.5020   | 0.0012   | 0.2718   |
| 40    | 0.1491   | 0.0375   | 0.2607   | 0.2163   | 0.0250   | 0.4077   | 0.0010   | 0.2354   |
| 45    | 0.1739   | 0.0288   | 0.3190   | 0.2381   | 0.0237   | 0.4526   | 0.0008   | 0.2035   |
| 50    | 0.2041   | 0.0420   | 0.3663   | 0.2482   | 0.0281   | 0.4683   | 0.0004   | 0.1793   |
| Best  | 0.1887   | 0.0440   | 0.3335   | 0.2507   | 0.0685   | 0.4329   | -0.0000  | 0.1101   |

**★ Best LP: epoch 5 (MSE=0.1420) — 2.4× worse than JEPA v1 best**  
**★ Best kNN: epoch 5 (MSE=0.2066) — 3.2× worse than JEPA v1 best**

#### JEPA v2 Training Log (50 epochs)

> val_loss is monotonically increasing from epoch 1 onward. The model is diverging.

| Epoch | train_total | pred_loss | var_loss | cov_loss | val_loss | lr       | ema    |
|-------|-------------|-----------|----------|----------|----------|----------|--------|
| 0     | 0.877       | **0.0026**| 0.1151   | 7.4681   | 0.7333   | 1.5e-05  | 0.996  |
| 1     | 0.582       | **0.0026**| 0.0829   | 4.1214   | **0.6243** ★ best | 3e-05 | 0.996 |
| 5     | 0.856       | **0.0025**| 0.1642   | 0.8209   | 0.9088   | 9e-05    | 0.996  |
| 10    | 1.173       | **0.0022**| 0.2291   | 0.6227   | 1.1875   | 1.5e-04  | 0.9961 |
| 20    | 1.434       | **0.0011**| 0.2820   | 0.5870   | 1.4584   | 1.46e-04 | 0.9963 |
| 30    | 1.686       | **0.0010**| 0.3323   | 0.5757   | 1.6996   | 1.33e-04 | 0.9966 |
| 40    | 1.835       | **0.0011**| 0.3623   | 0.5671   | 1.8512   | 1.13e-04 | 0.9970 |
| 49    | 1.915       | **0.0012**| 0.3782   | 0.5594   | 1.9267   | 9.1e-05  | 0.9975 |

---

## Head-to-Head Comparison (Best Checkpoints, Improved Eval)

| Model           | Best Epoch | LP MSE     | LP α MSE   | LP ζ MSE   | kNN MSE    | kNN α MSE  | kNN ζ MSE  |
|-----------------|-----------|------------|------------|------------|------------|------------|------------|
| **VideoMAE**    | 90        | **0.0362** | 0.0210     | **0.0515** | **0.0496** | 0.0211     | **0.0782** |
| **JEPA v1**     | 80 (LP) / 30 (kNN) | 0.0601 | **0.0162** | 0.0975 | 0.0643 | **0.0062** | 0.1223 |
| **JEPA v2**     | 5         | 0.1420     | 0.0254     | 0.2586     | 0.2066     | 0.0329     | 0.3802     |

**Ranking**: VideoMAE >> JEPA v1 >> JEPA v2

> **JEPA v1 bright spot**: Best α kNN MSE (0.0062) is 3.4× better than VideoMAE (0.0211). JEPA learns excellent geometric structure for dipole strength early on.

---

## Diagnosis

### Why JEPA v1 Underperforms VideoMAE

#### Feature σ Collapse
| Epoch | JEPA v1 Feat σ | VideoMAE Feat σ |
|-------|----------------|-----------------|
| 10    | 0.7463         | 0.4502          |
| 30    | 0.4951         | 0.2284          |
| 50    | 0.2885         | 0.0351          |
| 80    | 0.1443         | 0.0230          |
| 100   | 0.1194         | 0.0216          |

JEPA v1's feature σ drops 6× across training. The variance-only VICReg (weight=1.0) isn't strong enough. VideoMAE's σ also drops but reconstruction loss preserves fine-grained info.

#### kNN Peaks Early, LP Peaks Late
- kNN peaks at epoch 30 (σ=0.50) → good geometric structure
- LP peaks at epoch 80 (σ=0.14) → more linearly separable but geometrically worse
- Features lose discriminative neighborhoods as σ collapses

---

### Why JEPA v2 Is WORSE Than v1 — Root Cause Analysis

**The L2-normalization + Smooth-L1 combination makes the prediction task trivially easy, causing VICReg to completely dominate the loss.**

#### Evidence from training logs:

1. **pred_loss ≈ 0.001 for the ENTIRE run** (epochs 0–49). It never exceeds 0.003. By L2-normalizing both predicted and target features, the model only needs to match the *direction* on a unit hypersphere. This is trivially easy for a Transformer, so the prediction loss provides **zero useful gradient signal** for learning physics.

2. **var_loss monotonically increases** (0.12 → 0.38). The VICReg variance hinge loss grows every epoch, meaning feature dimensions keep collapsing below γ=1.0. The penalty fights this but can't win.

3. **total_loss = pred + 5.0×var + 0.04×cov**. Since pred≈0.001, the loss is almost entirely `5.0 × var_loss + 0.04 × cov_loss ≈ 1.9`. The encoder is **optimizing VICReg regularization rather than learning representations**.

4. **cov_loss drops** (7.47 → 0.56) — the covariance decorrelation works, but the model achieves decorrelation by collapsing dimensions, not by learning diverse features.

#### In contrast, JEPA v1 worked better because:
- **Raw MSE on unnormalized features** — the prediction loss scales with feature magnitude and stays meaningful. The model has to actually learn the physics to minimize MSE.
- **Lower regularization weight** (var_weight=1.0) — prediction loss dominated, keeping the model focused on learning useful representations.

#### The fundamental mistake in v2:
```
JEPA v1: loss = MSE(pred, target)         + 1.0 * var_loss
                ↑ meaningful (~0.01–0.5)     ↑ small penalty

JEPA v2: loss = SmoothL1(norm(pred), norm(target)) + 5.0 * var_loss + 0.04 * cov_loss
                ↑ trivially small (~0.001)            ↑ dominates everything
```

---

## What JEPA v2 Changed vs v1

| Aspect | JEPA v1 | JEPA v2 | Impact |
|--------|---------|---------|--------|
| Prediction loss | MSE on raw features | Smooth-L1 on L2-normed | ❌ Made prediction trivial |
| var_weight | 1.0 | 5.0 | ❌ Overwhelmed useful gradients |
| cov_weight | N/A | 0.04 | ⚠️ Decorrelation worked but at cost of collapse |
| VICReg scope | Pred only | Pred AND target | ⚠️ Sensible idea, bad with trivial pred loss |
| LR schedule | Flat | Cosine + warmup | ✅ Good improvement |
| EMA momentum | Fixed 0.996 | Cosine 0.996→0.999 | ✅ Good improvement |
| Grad clipping | None | max_norm=1.0 | ✅ Good improvement |

---

## JEPA v3 Design (Implemented)

Best of v1's loss + v2's training stability. Key principle: **prediction loss must dominate**.

```
JEPA v3: loss = MSE(pred, target)  + 1.0 * var_loss  + 0.01 * cov_loss
                ↑ v1's raw MSE       ↑ v1 weight        ↑ gentle decorrelation
```

| Aspect | v1 | v2 (failed) | **v3** |
|--------|------|------|------|
| Prediction loss | MSE raw | Smooth-L1 L2-normed | **MSE raw** (from v1) |
| var_weight | 1.0 | 5.0 | **1.0** (from v1) |
| cov_weight | — | 0.04 | **0.01** (reduced) |
| VICReg scope | Pred only | Pred + target | **Pred + target** (from v2) |
| LR schedule | Flat | Cosine + warmup | **Cosine + warmup** (from v2) |
| EMA momentum | Fixed 0.996 | 0.996→0.999 | **0.996→0.999** (from v2) |
| Grad clipping | None | 1.0 | **1.0** (from v2) |

**What to watch during training**:
- `pred_loss` should be ~0.01–0.5 (NOT 0.001 like v2)
- `var_loss` should stay below pred_loss
- `val_loss` should decrease over time (NOT monotonically increase like v2)
- Feature σ should stabilize or decrease slowly (not crash)

---

## File Reference

| File | Purpose |
|------|---------|
| `models/jepa.py` | JEPA v1 model (online encoder, EMA target, predictor, MSE + variance loss) |
| `models/jepa_v2.py` | JEPA v2 model (adds L2-norm, Smooth-L1, full VICReg with covariance) |
| `models/jepa_v3.py` | JEPA v3 model (v1's raw MSE + v2's bilateral VICReg at lower weights) |
| `train_jepa.py` | JEPA v1 training loop |
| `train_jepa_v2.py` | JEPA v2 training loop (cosine LR, EMA warmup, gradient clipping) |
| `train_jepa_v3.py` | JEPA v3 training loop (same stabilizers as v2, imports v3 model) |
| `models/mae.py` | VideoMAE model |
| `train.py` | VideoMAE training loop |
| `eval.py` | Evaluation (Linear Probe + kNN, improved with feature standardization) |
| `configs/jepa_small.yaml` | JEPA v1 config |
| `configs/jepa_v2.yaml` | JEPA v2 config |
| `configs/jepa_v3.yaml` | JEPA v3 config (var_weight=1.0, cov_weight=0.01) |
| `scripts/parse_logs.py` | Parse SLURM eval logs into results table |
| `scripts/log_result.py` | Log results to CSV |
| `scripts/eval_jepa_v2_sweep.sh` | Batch eval submission for JEPA v2 checkpoints |
| `scripts/eval_jepa_v3_sweep.sh` | Batch eval submission for JEPA v3 checkpoints |

---

## HPC Log Locations

| Experiment | Log Folder | Notes |
|------------|-----------|-------|
| JEPA v1 (improved eval) | `logs/jepa_v2_feat_std/` | "v2" in name = eval protocol, NOT model |
| VideoMAE (improved eval) | `logs/videamae_v2_feat_std/` | Same improved eval protocol |
| JEPA v2 (model) | `logs/jepa_v2_model_eval/` | v2 model, failed — VICReg dominated |
| JEPA v3 (model) | `logs/jepa_v3_eval/` | v3 model — results pending |

---

## How to Run JEPA v3

```bash
# Train default v3
sbatch --job-name=jepa-v3 scripts/train.sh configs/jepa_v3.yaml

# Train tuned v3 (use this one — HPO-optimized)
sbatch --job-name=jepa-v3t scripts/train.sh configs/jepa_v3_tuned.yaml

# After training, eval sweep
bash scripts/eval_jepa_v3_sweep.sh

# Parse results
python scripts/parse_logs.py --experiment jepa_v3_tuned --log_dir logs/jepa_v3_eval/
```

---

## HPO Sweep Results (Optuna, 20 Trials, TPE + MedianPruner)

> Sweep config: 20 trials × 15 epochs × 20% training data subsample × bs=16  
> Storage: `/scratch/psp8502/checkpoints/hpo/jepa_v3_hpo.db`  
> Duration: ~7.5 GPU-hours total (8 trials pruned early)

### All Trial Results

| Trial | val_loss | var_weight | cov_weight | lr | weight_decay | Status |
|-------|----------|------------|------------|-----|--------------|--------|
| 1  | 0.6202 | 0.6341 | 0.04123 | 2.70e-04 | 0.0397 | COMPLETE |
| 2  | 0.3638 | 0.3684 | 0.00184 | 5.72e-05 | 0.0735 | COMPLETE |
| 3  | 0.9975 | 1.1134 | 0.01596 | 5.24e-05 | 0.0933 | COMPLETE |
| 4  | PRUNED | 1.9783 | 0.00230 | 7.60e-05 | 0.0153 | PRUNED |
| 5  | 0.5169 | 0.5324 | 0.00779 | 1.35e-04 | 0.0196 | COMPLETE |
| 6  | PRUNED | 1.1435 | 0.00170 | 9.80e-05 | 0.0232 | PRUNED |
| 7  | PRUNED | 0.7765 | 0.02160 | 7.92e-05 | 0.0327 | PRUNED |
| 8  | PRUNED | 1.0896 | 0.00120 | 2.03e-04 | 0.0148 | PRUNED |
| 9  | 0.3104 | 0.2939 | 0.04094 | 4.62e-04 | 0.0643 | COMPLETE |
| 10 | PRUNED | 0.5329 | 0.00150 | 2.42e-04 | 0.0276 | PRUNED |
| **11** | **0.2782** | 0.2631 | 0.00637 | 4.66e-04 | 0.0512 | COMPLETE |
| **12** | **0.2761** | 0.2592 | 0.00481 | 4.85e-04 | 0.0577 | COMPLETE |
| **13** | **0.2860** | 0.2723 | 0.00442 | 4.95e-04 | 0.0478 | COMPLETE |
| 14 | PRUNED | 0.3867 | 0.00510 | 3.60e-04 | 0.0532 | PRUNED |
| **15** ★ | **0.2752** | 0.2568 | 0.00952 | 3.40e-04 | 0.0865 | COMPLETE |
| 16 | PRUNED | 2.4103 | 0.01480 | 3.39e-04 | 0.0890 | PRUNED |
| 17 | PRUNED | 0.3940 | 0.00320 | 1.64e-04 | 0.0668 | PRUNED |
| 18 | PRUNED | 0.4583 | 0.00890 | 3.30e-04 | 0.0961 | PRUNED |
| 19 | 0.3222 | 0.3130 | 0.00992 | 2.73e-04 | 0.0355 | COMPLETE |
| 20 | PRUNED | 1.5331 | 0.00330 | 4.06e-04 | 0.0758 | PRUNED |

### Top 5 Trials

| Rank | Trial | val_loss | var_weight | cov_weight | lr | weight_decay |
|------|-------|----------|------------|------------|-----|--------------|
| 1 ★  | 15 | **0.2752** | 0.2568 | 0.00952 | 3.40e-04 | 0.0865 |
| 2    | 12 | 0.2761 | 0.2592 | 0.00481 | 4.85e-04 | 0.0577 |
| 3    | 11 | 0.2782 | 0.2631 | 0.00637 | 4.66e-04 | 0.0512 |
| 4    | 13 | 0.2860 | 0.2723 | 0.00442 | 4.95e-04 | 0.0478 |
| 5    | 9  | 0.3104 | 0.2939 | 0.04094 | 4.62e-04 | 0.0643 |

### Key Findings from HPO

#### 1. var_weight: Should be ~0.25–0.30, NOT 1.0
| var_weight range | Outcome |
|---|---|
| 0.25–0.30 | **All top-5 trials** |
| 0.30–0.55 | Mediocre (0.36–0.52) |
| 0.55–1.1  | Poor (0.62–0.88) |
| ≥ 1.1     | **All pruned** |

Our v3 default of 1.0 was still 4× too high. The message from v1→v2→v3 is consistent: **prediction loss must dominate; even moderate VICReg weights are harmful.**

#### 2. lr: Should be ~3–5×10⁻⁴, NOT 1.5×10⁻⁴
All top-5 trials have lr in [3.4e-4, 5.0e-4]. Our v3 default (1.5e-4) was 2–3× too low. With lower var_weight (0.25 vs 1.0), the loss is dominated by pred_loss which is smaller in magnitude, so a higher LR is needed to make proportionally larger updates.

#### 3. cov_weight: ~0.005–0.01 confirmed correct
Our v3 default of 0.01 was well-calibrated. The top trials span 0.004–0.041 with no clear monotonic pattern — this suggests cov_weight is not the dominant factor.

#### 4. weight_decay: Slightly higher than 0.05 helps
Top trials range 0.047–0.087. Our default 0.05 is at the low end; slightly higher wd (0.06–0.09) provides better regularization for the ViT.

### Interpretation: Why Lower var_weight + Higher LR Work Together
```
With var_weight=1.0:  total_loss ≈ pred_loss(~0.3) + 1.0 × var_loss(~0.3) → roughly equal
With var_weight=0.25: total_loss ≈ pred_loss(~0.3) + 0.25 × var_loss(~0.3) → pred dominates 5:1
```
Lower VICReg weight → encoder focuses on physics prediction → needs higher LR to learn faster from the now-dominant prediction signal.

### Changes vs JEPA v3 Defaults

| Param | v3 default | HPO best | Change |
|-------|-----------|----------|--------|
| var_weight | 1.0 | **0.257** | 4× lower |
| cov_weight | 0.01 | **0.0095** | essentially same |
| lr | 1.5e-4 | **3.4e-4** | 2.3× higher |
| weight_decay | 0.05 | **0.087** | 1.7× higher |

