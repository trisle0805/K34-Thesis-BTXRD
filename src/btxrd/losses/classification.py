"""Weighted classification loss."""

from __future__ import annotations

import torch
from torch import nn


def build_classification_loss(class_weights: torch.Tensor, device: torch.device) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=class_weights.to(device=device, dtype=torch.float32))
