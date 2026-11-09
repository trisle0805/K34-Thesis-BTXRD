"""Class-prototype contrastive loss."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrototypeContrastiveLoss(nn.Module):
    def forward(self, u: torch.Tensor, prototypes: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        logits = (u.float() @ prototypes.float().T) / temperature
        return F.cross_entropy(logits, labels)
