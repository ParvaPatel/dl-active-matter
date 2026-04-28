"""Video-JEPA (Joint-Embedding Predictive Architecture).

Temporal prediction: encode first 16 frames (context), predict the
representation of the next 16 frames (target) produced by an EMA encoder.

With 81 timesteps per trajectory and n_frames=32:
  (81 - 32 + 1) * 175 = 8,750 training samples
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.encoder import SpatioTemporalViT, TransformerBlock


class JEPAPredictor(nn.Module):
    """Narrow Transformer predictor: maps context features → predicted target features."""
    def __init__(self, encoder_dim=384, predictor_dim=256, depth=3, num_heads=4):
        super().__init__()
        self.predictor_embed = nn.Linear(encoder_dim, predictor_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(predictor_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.predictor_proj = nn.Linear(predictor_dim, encoder_dim)
        self.norm = nn.LayerNorm(predictor_dim)

    def forward(self, x):
        x = self.predictor_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.predictor_proj(x)
        return x


class VideoJEPA(nn.Module):
    """
    Video-JEPA: Predicts future frame representations from current frames.

    Architecture:
        - Online encoder processes context frames (first 16 of 32)
        - EMA target encoder processes target frames (last 16 of 32)
        - Predictor maps context features → predicted target features
        - VICReg variance penalty prevents representation collapse

    Input: (B, 32, C, H, W) — 32 consecutive frames
    Context: frames[:16] → online encoder → context features (B, 1568, D)
    Target:  frames[16:] → EMA encoder   → target features  (B, 1568, D)
    Loss:    MSE(predictor(context_features), target_features) + VICReg
    """
    def __init__(
        self,
        encoder: SpatioTemporalViT,
        predictor_dim=256,
        predictor_depth=3,
        predictor_heads=4,
        context_frames=16,
    ):
        super().__init__()
        self.context_frames = context_frames

        # Online encoder (processes context)
        self.encoder = encoder

        # Target encoder (EMA of online encoder, processes target)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor network: maps context representation → predicted target representation
        self.predictor = JEPAPredictor(
            encoder_dim=encoder.embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        """Exponential Moving Average update for target encoder."""
        for online_params, target_params in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_params.data = (
                target_params.data * momentum + online_params.data * (1.0 - momentum)
            )

    def variance_loss(self, x, gamma=1.0, eps=1e-4):
        """VICReg Variance hinge loss to prevent representation collapse.
        
        Penalizes when std of any embedding dimension drops below gamma.
        Applied across the (B * N_tokens) sample dimension.
        """
        std = torch.sqrt(x.var(dim=0) + eps)
        loss_var = torch.mean(F.relu(gamma - std))
        return loss_var

    def forward(self, x):
        """
        Args:
            x: (B, 32, C, H, W) — 32 consecutive frames

        Returns:
            total_loss, mse_loss, variance_loss, target_std
        """
        B = x.shape[0]
        cf = self.context_frames

        # --- Temporal split ---
        x_context = x[:, :cf]   # (B, 16, C, H, W) — first 16 frames
        x_target = x[:, cf:]    # (B, 16, C, H, W) — last 16 frames

        # --- Online encoder: context frames ---
        context_features = self.encoder(x_context)  # (B, 1568, D)

        # --- Target encoder: future frames (no gradient) ---
        with torch.no_grad():
            target_features = self.target_encoder(x_target)  # (B, 1568, D)

        # --- Predictor: context → predicted target ---
        pred_features = self.predictor(context_features)  # (B, 1568, D)

        # --- Loss ---
        # L2 prediction loss (detach target for explicit stop-gradient)
        mse_loss = F.mse_loss(pred_features, target_features.detach())

        # VICReg variance loss on predicted features
        # flatten (B, N, D) → (B*N, D) to compute per-dimension variance
        flat_pred = pred_features.reshape(-1, pred_features.shape[-1])
        var_loss = self.variance_loss(flat_pred, gamma=1.0)

        total_loss = mse_loss + 1.0 * var_loss

        # Collapse monitoring metric: mean std of target features
        # NOTE: returned as tensor (no .item()) to avoid torch.compile graph break.
        # Caller should call .item() outside the compiled forward region.
        target_std = torch.sqrt(
            target_features.reshape(-1, target_features.shape[-1]).var(dim=0) + 1e-4
        ).mean()

        return total_loss, mse_loss, var_loss, target_std
