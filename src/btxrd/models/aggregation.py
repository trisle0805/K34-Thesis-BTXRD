"""Knowledge proposition aggregation."""

from __future__ import annotations

import torch
from torch import nn


class MeanPooling(nn.Module):
    def forward(self, prop_features: torch.Tensor) -> tuple[torch.Tensor, None]:
        return prop_features.mean(dim=1), None


class LearnedAttentionPooling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.W_h = nn.Linear(dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, prop_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.w(torch.tanh(self.W_h(prop_features))).squeeze(-1).softmax(dim=-1)
        return torch.sum(prop_features * weights.unsqueeze(-1), dim=1), weights


def build_aggregation(name: str, dim: int, hidden_dim: int) -> nn.Module:
    if name == "mean":
        return MeanPooling()
    if name == "learned":
        return LearnedAttentionPooling(dim, hidden_dim)
    raise ValueError(f"Unsupported aggregation: {name}")
