"""Metadata and branch fusion modules."""

from __future__ import annotations

import torch
from torch import nn


class MetadataMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        return self.network(metadata)


class ClassifierHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float, head_type: str):
        super().__init__()
        if head_type == "linear":
            self.network: nn.Module = nn.Linear(input_dim, num_classes)
        elif head_type == "mlp":
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unsupported classifier head: {head_type}")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
