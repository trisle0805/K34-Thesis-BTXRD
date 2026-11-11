"""Weighted classification loss."""

from __future__ import annotations

import torch
from torch import nn


def build_classification_loss(class_weights: torch.Tensor, device: torch.device) -> nn.CrossEntropyLoss:
    return nn.CrossEntropyLoss(weight=class_weights.to(device=device, dtype=torch.float32))


def branch_classification_loss(output, labels, criterion, image_weight=0.0, knowledge_weight=0.0):
    final = criterion(output['logits'], labels)
    branches = {name: criterion(output[f'logits_{name}'], labels)
                for name in ('img', 'kb') if f'logits_{name}' in output}
    total = final
    # Do not even add zero terms: preserve the historical loss bit for bit.
    if image_weight:
        total = total + image_weight * branches['img']
    if knowledge_weight:
        total = total + knowledge_weight * branches['kb']
    return total, final, branches
