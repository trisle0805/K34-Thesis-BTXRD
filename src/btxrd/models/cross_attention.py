"""Visual-token projection and proposition-guided cross-attention."""

from __future__ import annotations

import torch
from torch import nn


def fixed_2d_sincos(rows: int, cols: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("Fixed 2D sine/cosine encoding requires knowledge.dim divisible by 4")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(quarter, 1)))
    y, x = torch.meshgrid(
        torch.arange(rows, device=device, dtype=torch.float32),
        torch.arange(cols, device=device, dtype=torch.float32),
        indexing="ij",
    )
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1).to(dtype=dtype)


class PropositionCrossAttention(nn.Module):
    def __init__(self, text_emb: torch.Tensor, image_dim: int, dim: int, num_heads: int, dropout: float, use_positional_embedding: bool = False):
        super().__init__()
        self.register_buffer("text_emb", text_emb.detach().float().clone())
        self.token_proj = nn.Linear(image_dim, dim)
        self.query_proj = nn.Linear(text_emb.shape[1], dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.use_positional_embedding = use_positional_embedding

    def forward(self, feature_map: torch.Tensor, spatial_shape=None) -> tuple[torch.Tensor, torch.Tensor]:
        if feature_map.ndim == 4:
            spatial_shape = feature_map.shape[-2:]
            feature_map = feature_map.flatten(2).transpose(1, 2)
        batch = feature_map.shape[0]
        visual = self.token_proj(feature_map)
        if self.use_positional_embedding:
            if spatial_shape is None:
                raise ValueError('Positional embedding requires spatial_shape')
            rows, cols = spatial_shape
            visual = visual + fixed_2d_sincos(rows, cols, visual.shape[-1], visual.device, visual.dtype).unsqueeze(0)
        query = self.query_proj(self.text_emb).unsqueeze(0).expand(batch, -1, -1)
        prop_features, weights = self.attn(
            query,
            visual,
            visual,
            need_weights=True,
            average_attn_weights=True,
        )
        return prop_features, weights
