"""VideoMAE: Masked Autoencoder for spatiotemporal representation learning."""

import torch
import torch.nn as nn
import random

from models.encoder import SpatioTemporalViT, TransformerBlock


class MAEDecoder(nn.Module):
    """Lightweight decoder to reconstruct masked patches."""

    def __init__(self, embed_dim=384, decoder_dim=192, decoder_depth=2,
                 decoder_heads=3, patch_size=(2, 16, 16), in_channels=11):
        super().__init__()
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(decoder_dim, decoder_heads, mlp_ratio=4.0)
            for _ in range(decoder_depth)
        ])
        self.norm = nn.LayerNorm(decoder_dim)

        # Reconstruct patch pixels
        pt, ph, pw = patch_size
        self.pred = nn.Linear(decoder_dim, in_channels * pt * ph * pw)

        self.patch_size = patch_size
        self.in_channels = in_channels

    def forward(self, x):
        x = self.decoder_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x = self.pred(x)  # (B, N, C*pt*ph*pw)
        return x


class VideoMAE(nn.Module):
    """
    Masked Autoencoder for video/spatiotemporal data.

    1. Tokenize input into tube patches.
    2. Mask a fraction (e.g., 75%) of tokens.
    3. Encode only visible tokens.
    4. Decode all tokens (with mask tokens) to reconstruct masked patches.
    """

    def __init__(
        self,
        encoder: SpatioTemporalViT,
        decoder_dim=192,
        decoder_depth=2,
        decoder_heads=3,
        mask_ratio=0.75,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio

        self.decoder = MAEDecoder(
            embed_dim=encoder.embed_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            patch_size=encoder.patch_embed.patch_size,
            in_channels=11,
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder positional embeddings
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, encoder.num_patches, decoder_dim)
        )
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

    def random_masking(self, x, mask_ratio):
        """
        Random masking: keep (1-mask_ratio) tokens.

        Args:
            x: (B, N, D) — all patch embeddings
        Returns:
            x_visible: (B, N_vis, D)
            mask: (B, N) — 0=keep, 1=mask
            ids_restore: (B, N) — indices to unshuffle
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random permutation
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep tokens
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Binary mask: 0=keep, 1=mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)

        return x_visible, mask, ids_restore, ids_keep

    def patchify(self, x):
        """
        Convert input to patch targets for reconstruction loss.

        x: (B, T, C, H, W)
        Returns: (B, N, patch_volume) where patch_volume = C * pt * ph * pw
        """
        B, T, C, H, W = x.shape
        pt, ph, pw = self.encoder.patch_embed.patch_size

        # Reshape into patches
        x = x.reshape(B, T // pt, pt, C, H // ph, ph, W // pw, pw)
        x = x.permute(0, 1, 4, 6, 3, 2, 5, 7)  # (B, Nt, Nh, Nw, C, pt, ph, pw)
        x = x.reshape(B, -1, C * pt * ph * pw)   # (B, N, patch_vol)
        return x

    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) — input video
        Returns:
            loss: reconstruction MSE on masked patches
            pred: (B, N, patch_vol) — predicted patches
            mask: (B, N) — 0=visible, 1=masked
        """
        # 1. Patch embed
        patches, grid = self.encoder.patch_embed(x)  # (B, N, D)
        patches = patches + self.encoder.pos_embed

        # 2. Mask
        visible, mask, ids_restore, ids_keep = self.random_masking(patches, self.mask_ratio)

        # 3. Encode only visible
        for blk in self.encoder.blocks:
            visible = blk(visible)
        visible = self.encoder.norm(visible)

        # 4. Prepare decoder input: embed visible + fill mask tokens
        B, N, _ = patches.shape
        decoder_visible = self.decoder.decoder_embed(visible)

        mask_tokens = self.mask_token.expand(B, N - decoder_visible.shape[1], -1)
        full = torch.cat([decoder_visible, mask_tokens], dim=1)

        # Unshuffle to original order
        full = torch.gather(
            full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[2])
        )
        full = full + self.decoder_pos_embed

        # 5. Decode
        for blk in self.decoder.blocks:
            full = blk(full)
        full = self.decoder.norm(full)
        pred = self.decoder.pred(full)  # (B, N, patch_vol)

        # 6. Loss on masked patches only
        target = self.patchify(x)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)         # (B, N) — per-patch MSE
        loss = (loss * mask).sum() / mask.sum()  # mean over masked patches

        return loss, pred, mask
