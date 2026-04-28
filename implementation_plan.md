# Video-JEPA Implementation Plan

This document outlines the strategy for moving from your VideoMAE baseline to the **Video-JEPA** (Joint-Embedding Predictive Architecture) model.

## Goal Description

Video-JEPA contrasts with VideoMAE by predicting the representations directly rather than reconstructing original pixels. This generally leads to stronger semantic features but is notoriously prone to representation collapse (all tokens mapping to the same uniform vector). 

To solve this we will:
1. Use an **EMA Target Encoder** to provide stable, non-collapsing learning targets.
2. Introduce a **Predictor** network.
3. Enhance the loss with **VICReg** (Variance-Invariance-Covariance) components as a safety fallback against collapse.

> [!WARNING] 
> Representation collapse is the highest risk right now. JEPA is highly unstable out of the box if hyperparameters aren't perfect. We will be rigorously tracking the standard deviation of our representations in W&B.

## Proposed Changes

---

### Phase 1: JEPA Architectural Components

#### [NEW] `models/jepa.py`

- **`JEPAPredictor`**: A 3-layer narrow Transformer (e.g. `embed_dim=256`, 4 heads). It will take the output of the context encoder combined with positional embeddings for the target masks to predict target embeddings.
- **`VideoJEPA` Main Module**:
  - Initializes `encoder` (online) and `target_encoder` (EMA).
  - Stops gradients for `target_encoder`.
  - Implements masking logic (target blocks vs context blocks).
  - Implements the **VICReg Loss Component** (specifically the Variance bound, to ensure `std(z)` is always pushing > 1).
  - Method `update_target_encoder(momentum)` to handle the exponential moving average update at the end of every batch.

#### [NEW] `configs/jepa_small.yaml`

- Mirroring `videomae_small` for the encoder settings (`ViT-Small`).
- **JEPA-specific hyperparameters**: Predictor depth `3`, Predictor dim `256`, Momentum `0.996` (scaling to `1.0` eventually), weight decay.
- `experiment_name: jepa_small`.

---

### Phase 2: Training Pipeline Adaptation

#### [NEW] `train_jepa.py` (or extending `train.py`)

*Decision: I propose creating a separate `train_jepa.py` to keep the codebase cleaner rather than adding complex `if is_jepa:` branching inside the VideoMAE loop. VideoMAE requires pixel reconstruction, JEPA requires EMA loops and hidden-state loss.*

- Includes the standard dataset loading code.
- **Batch Processing**:
  - `target_features = target_encoder(x, target_mask)` *with `torch.no_grad()`*
  - `pred_features = predictor(encoder(x, context_mask), target_mask_tokens)`
  - Loss = `MSE(pred_features, target_features)` + `VICReg_variance_penalty`.
- **Target Network Update**:
  - Add `model.update_target_encoder(momentum=ema_momentum)` after `optimizer.step()`.
- **W&B Monitors**:
  - Log `std(z)` to ensure we are not collapsing.

#### [MODIFY] `scripts/train.sh`
- Allow targeting `train_jepa.py` easily.

## Open Questions

- **Do you approve of creating a separate `train_jepa.py`?** Keeping them separate means you can easily trace bugs for VideoMAE without digging through JEPA EMA loops. 
- **Context/Target Masking Strategy**: Are you okay with starting with standard VideoMAE-style random masking split for context/target, or do you strictly want contiguous "block masking" as proposed in the original I-JEPA paper? Random is faster to implement.

## Verification Plan

### Automated Checks
- The parameter count is logged and strictly kept under 100M. 
- Validation Loss must decrease without `std(z)` shrinking towards zero. 

### Final Verification
- Run `eval.py` using a checkpoint generated from `train_jepa.py` to see if JEPA beats our current Random/MAE MSE scores on the linear probe and kNN.
