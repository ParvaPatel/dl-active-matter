"""Video-JEPA v2 — Improved with full VICReg regularization and feature normalization.

Key improvements over v1:
1. L2-normalized prediction targets → prevents scale collapse
2. Smooth-L1 prediction loss → robust to outlier patches
3. Full VICReg: variance + covariance on both predicted and target features
4. Covariance loss decorrelates feature dimensions (prevents dimensional collapse)
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


class VideoJEPAv2(nn.Module):
    """
    Video-JEPA v2: Improved temporal prediction with full VICReg regularization.

    Improvements over v1:
        - L2-normalized prediction targets (prevents scale collapse)
        - Smooth-L1 prediction loss (robust to outlier patches)
        - Full VICReg: variance + covariance on both pred and target features
        - Covariance loss decorrelates feature dimensions

    Input: (B, 32, C, H, W) — 32 consecutive frames
    Context: frames[:16] → online encoder
    Target:  frames[16:] → EMA encoder
    """
    def __init__(
        self,
        encoder: SpatioTemporalViT,
        predictor_dim=256,
        predictor_depth=3,
        predictor_heads=4,
        context_frames=16,
        var_weight=5.0,
        cov_weight=0.04,
        var_gamma=1.0,
    ):
        super().__init__()
        self.context_frames = context_frames
        self.var_weight = var_weight
        self.cov_weight = cov_weight
        self.var_gamma = var_gamma

        # Online encoder (processes context)
        self.encoder = encoder

        # Target encoder (EMA of online encoder, processes target)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Predictor network
        self.predictor = JEPAPredictor(
            encoder_dim=encoder.embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        """Exponential Moving Average update for target encoder."""
        for online_p, target_p in zip(
            self.encoder.parameters(), self.target_encoder.parameters()
        ):
            target_p.data = target_p.data * momentum + online_p.data * (1.0 - momentum)

    def variance_loss(self, x, gamma=1.0, eps=1e-4):
        """VICReg variance hinge loss.

        Penalizes when std of any embedding dimension drops below gamma.
        Applied across the sample dimension (B*N_tokens).
        """
        std = torch.sqrt(x.var(dim=0) + eps)
        return torch.mean(F.relu(gamma - std))

    def covariance_loss(self, x):
        """VICReg covariance loss.

        Forces off-diagonal elements of the feature covariance matrix toward zero,
        decorrelating the embedding dimensions so each dimension encodes unique info.
        """
        N, D = x.shape
        x = x - x.mean(dim=0)
        cov = (x.T @ x) / (N - 1)  # (D, D)
        # Zero out diagonal — we only penalize off-diagonal correlations
        cov.fill_diagonal_(0)
        return cov.pow(2).sum() / D

    def forward(self, x):
        """
        Args:
            x: (B, 32, C, H, W) — 32 consecutive frames

        Returns:
            total_loss, pred_loss, var_loss, cov_loss, target_std
        """
        B = x.shape[0]
        cf = self.context_frames

        # --- Temporal split ---
        x_context = x[:, :cf]   # (B, 16, C, H, W)
        x_target = x[:, cf:]    # (B, 16, C, H, W)

        # --- Encode ---
        context_features = self.encoder(x_context)       # (B, 1568, D)
        with torch.no_grad():
            target_features = self.target_encoder(x_target)  # (B, 1568, D)

        # --- Predict ---
        pred_features = self.predictor(context_features)  # (B, 1568, D)

        # --- Prediction loss on L2-normalized features ---
        # L2-norm prevents the model from "cheating" by shrinking all features
        # Smooth-L1 is more robust to outlier patches than MSE
        target_norm = F.normalize(target_features.detach(), dim=-1)
        pred_norm = F.normalize(pred_features, dim=-1)
        pred_loss = F.smooth_l1_loss(pred_norm, target_norm)

        # --- VICReg regularization on raw features ---
        # Flatten (B, N, D) → (B*N, D) for per-dimension statistics
        flat_pred = pred_features.reshape(-1, pred_features.shape[-1])
        flat_target = target_features.reshape(-1, target_features.shape[-1])

        # Variance loss on BOTH predicted and target features
        var_loss_pred = self.variance_loss(flat_pred, gamma=self.var_gamma)
        var_loss_target = self.variance_loss(flat_target, gamma=self.var_gamma)
        var_loss = (var_loss_pred + var_loss_target) / 2.0

        # Covariance loss on predicted features (decorrelate dimensions)
        cov_loss = self.covariance_loss(flat_pred)

        # --- Total loss ---
        total_loss = pred_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        # --- Monitoring metric ---
        target_std = torch.sqrt(
            flat_target.var(dim=0) + 1e-4
        ).mean().item()

        return total_loss, pred_loss, var_loss, cov_loss, target_std
