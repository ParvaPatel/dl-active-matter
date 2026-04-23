"""Video-JEPA v3 — Best of v1 loss + v2 training stabilizers.

Key design decisions (learned from v1 and v2 failures):
1. RAW MSE prediction loss on unnormalized features (from v1)
   - v2's L2-norm + Smooth-L1 made pred_loss trivially ~0.001, killing gradient signal
   - Raw MSE keeps the prediction task meaningful throughout training
2. Variance loss on BOTH predicted and target features (from v2, but lower weight)
   - var_weight=1.0 (v1 level), NOT 5.0 (v2 was way too aggressive)
3. Covariance decorrelation at very low weight (cov_weight=0.01)
   - Gently encourages feature dimensions to encode unique info
   - Low enough to never overwhelm the prediction loss
4. All v2 training improvements are in train_jepa_v3.py (cosine LR, EMA warmup, grad clip)
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


class VideoJEPAv3(nn.Module):
    """
    Video-JEPA v3: v1's meaningful prediction loss + v2's training stability.

    What changed from v2 (and why):
        - Raw MSE instead of Smooth-L1 on L2-normed features (v2's pred_loss was ~0.001 = useless)
        - var_weight=1.0 instead of 5.0 (v2's VICReg overwhelmed the prediction signal)
        - Covariance loss kept but at cov_weight=0.01 (gentle decorrelation)
        - Variance applied to both pred and target (good idea from v2, safe now that pred_loss dominates)

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
        var_weight=1.0,
        cov_weight=0.01,
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

        # --- Prediction loss: RAW MSE on unnormalized features ---
        # This is the critical difference from v2: MSE on raw features keeps the
        # prediction task meaningful (pred_loss ~ 0.01–0.5), ensuring the encoder
        # actually learns physical structure to minimize prediction error.
        pred_loss = F.mse_loss(pred_features, target_features.detach())

        # --- VICReg regularization ---
        # Flatten (B, N, D) → (B*N, D) for per-dimension statistics
        flat_pred = pred_features.reshape(-1, pred_features.shape[-1])
        flat_target = target_features.reshape(-1, target_features.shape[-1])

        # Variance loss on BOTH predicted and target features
        var_loss_pred = self.variance_loss(flat_pred, gamma=self.var_gamma)
        var_loss_target = self.variance_loss(flat_target, gamma=self.var_gamma)
        var_loss = (var_loss_pred + var_loss_target) / 2.0

        # Covariance loss on predicted features (gentle decorrelation)
        cov_loss = self.covariance_loss(flat_pred)

        # --- Total loss ---
        # pred_loss dominates (~0.01–0.5), VICReg is a gentle regularizer
        total_loss = pred_loss + self.var_weight * var_loss + self.cov_weight * cov_loss

        # --- Monitoring metric (returned as tensor — caller calls .item() outside compiled region) ---
        # NOTE: .item() inside a compiled forward() causes a graph break in torch.compile,
        # splitting the graph into two and reducing the speedup. Keep as tensor here.
        target_std = torch.sqrt(flat_target.var(dim=0) + 1e-4).mean()

        return total_loss, pred_loss, var_loss, cov_loss, target_std
