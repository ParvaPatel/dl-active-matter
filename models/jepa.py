"""Video-JEPA (Joint-Embedding Predictive Architecture)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.encoder import SpatioTemporalViT, TransformerBlock

class JEPAPredictor(nn.Module):
    """Narrow Transformer predictor to map context embeddings + target masking to target embeddings."""
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
    Video-JEPA: Predicts representations of target patches using context patches.
    Mitigates representation collapse via EMA target network and VICReg variance penalty.
    """
    def __init__(
        self,
        encoder: SpatioTemporalViT,
        predictor_dim=256,
        predictor_depth=3,
        predictor_heads=4,
        target_ratio=0.20,
        context_ratio=0.40,
    ):
        super().__init__()
        # Online encoder
        self.encoder = encoder
        
        # Target encoder (EMA of online encoder)
        self.target_encoder = copy.deepcopy(encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.target_ratio = target_ratio
        self.context_ratio = context_ratio

        # Predictor network
        self.predictor = JEPAPredictor(
            encoder_dim=encoder.embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_heads,
        )

        # Learnable mask token for the predictor
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        
        # We share positional embedding from the encoder for the predictor!
        
    @torch.no_grad()
    def update_target_encoder(self, momentum=0.996):
        """Exponential Moving Average update for target encoder."""
        for online_params, target_params in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_params.data = target_params.data * momentum + online_params.data * (1.0 - momentum)

    def generate_masks(self, B, N, device):
        """
        Generate disjoint random masks for Context and Target.
        Returns indices to gather.
        """
        noise = torch.rand(B, N, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        num_target = int(N * self.target_ratio)
        num_context = int(N * self.context_ratio)
        
        # Split tokens deterministically via shuffle
        ids_target = ids_shuffle[:, :num_target]
        ids_context = ids_shuffle[:, num_target:num_target+num_context]
        
        return ids_context, ids_target

    def variance_loss(self, x, gamma=1.0, eps=1e-4):
        """VICReg Variance hinge loss to explicitly prevent representation collapse."""
        std = torch.sqrt(x.var(dim=0) + eps)
        loss_var = torch.mean(F.relu(gamma - std))
        return loss_var

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        Returns: total_loss, mse_loss, variance_loss, target_std
        """
        B, T, C, H, W = x.shape
        
        # 1. Embed ALL patches using patch_embed & pos_embed
        # Encode online context
        patches_online, _ = self.encoder.patch_embed(x)
        patches_online = patches_online + self.encoder.pos_embed
        
        # Encode targets with EMA — entire target path must be in no_grad
        # to avoid wasting VRAM on intermediate activations
        with torch.no_grad():
            patches_target, _ = self.target_encoder.patch_embed(x)
            patches_target = patches_target + self.target_encoder.pos_embed

        N = patches_online.shape[1]
        
        # 2. Get disjoint target and context masks
        ids_context, ids_target = self.generate_masks(B, N, x.device)
        
        # 3. Process Target Encoder (fully inside no_grad)
        with torch.no_grad():
            target_input = torch.gather(patches_target, 1, ids_target.unsqueeze(-1).expand(-1, -1, patches_target.shape[-1]))
            target_features = target_input
            for blk in self.target_encoder.blocks:
                target_features = blk(target_features)
            target_features = self.target_encoder.norm(target_features) # (B, N_target, D)
        
        # 4. Process Context Encoder
        context_input = torch.gather(patches_online, 1, ids_context.unsqueeze(-1).expand(-1, -1, patches_online.shape[-1]))
        context_features = context_input
        for blk in self.encoder.blocks:
            context_features = blk(context_features)
        context_features = self.encoder.norm(context_features) # (B, N_context, D)
        
        # 5. Predictor Preparation: 
        # We must predict target tokens using context tokens.
        # Input to predictor: [Context Features] + [Mask Tokens placed at Target Pos]
        # Gather Target positional embeddings
        target_pos = torch.gather(self.encoder.pos_embed.expand(B, -1, -1), 1, ids_target.unsqueeze(-1).expand(-1, -1, patches_online.shape[-1]))
        
        mask_tokens = self.mask_token.expand(B, target_features.shape[1], -1)
        pred_target_tokens = mask_tokens + target_pos 
        
        # Feed combined sequence to predictor
        predictor_input = torch.cat([context_features, pred_target_tokens], dim=1)
        pred_output_full = self.predictor(predictor_input)
        
        # Extract only the predicted target tokens (they were concatenated at the end)
        pred_features = pred_output_full[:, context_features.shape[1]:, :] # (B, N_target, D)
        
        # 6. Loss Calculation
        # L2 Prediction loss (detach target to be explicit about stop-gradient)
        mse_loss = F.mse_loss(pred_features, target_features.detach())
        
        # VICReg Variance loss (collapse penalty applied to embeddings across the batch)
        # flatten (B, N_target, D) -> (B * N_target, D)
        flat_pred = pred_features.reshape(-1, pred_features.shape[-1])
        var_loss = self.variance_loss(flat_pred, gamma=1.0)
        
        # Scale variance penalty heavily since it's a regularizer
        total_loss = mse_loss + 1.0 * var_loss
        
        # Metrics for wandb
        target_std = torch.sqrt(target_features.reshape(-1, target_features.shape[-1]).var(dim=0) + 1e-4).mean().item()
        
        return total_loss, mse_loss, var_loss, target_std
