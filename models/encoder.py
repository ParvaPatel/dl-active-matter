"""Spatiotemporal Vision Transformer encoder with tube tokenization."""

import torch
import torch.nn as nn
import math


class PatchEmbed3D(nn.Module):
    """Convert (B, T, C_in, H, W) → (B, N, D) via 3D tube patches."""

    def __init__(self, in_channels=11, patch_size=(2, 16, 16), embed_dim=384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, T, C, H, W) → (B, C, T, H, W) for Conv3d
        x = x.permute(0, 2, 1, 3, 4)
        x = self.proj(x)  # (B, D, T', H', W')
        B, D, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, (T, H, W)


class TransformerBlock(nn.Module):
    """Standard pre-norm transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class SpatioTemporalViT(nn.Module):
    """
    ViT encoder for spatiotemporal data with tube patch embedding.

    Input:  (B, 16, 11, 224, 224)
    Output: (B, N, D) where N = (T/pt) * (H/ph) * (W/pw)
    """

    def __init__(
        self,
        in_channels=11,
        patch_size=(2, 16, 16),
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        drop_rate=0.0,
        input_size=(16, 224, 224),
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed3D(in_channels, patch_size, embed_dim)

        # Compute number of tokens
        T, H, W = input_size
        pt, ph, pw = patch_size
        self.num_patches = (T // pt) * (H // ph) * (W // pw)  # 8 * 14 * 14 = 1568

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, C, H, W)
            mask: Optional boolean mask (B, N) — True = keep, False = mask out.
                  Used by MAE to only process visible tokens.
        Returns:
            (B, N, D) or (B, N_visible, D) if mask is provided.
        """
        x, grid = self.patch_embed(x)  # (B, N, D)
        x = x + self.pos_embed

        if mask is not None:
            # Keep only visible tokens
            B, N, D = x.shape
            x = x[mask].reshape(B, -1, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def mean_pool(self, x):
        """Global average pooling: (B, N, D) → (B, D)."""
        return x.mean(dim=1)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
