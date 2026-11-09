"""DenseNet-169 feature encoder compatible with the image-only baseline."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet169_Weights, densenet169


class DenseNet169Encoder(nn.Module):
    output_dim = 1664

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = DenseNet169_Weights.DEFAULT if pretrained else None
        self.features = densenet169(weights=weights).features

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = F.relu(self.features(image), inplace=False)
        z_img = feature_map.mean(dim=(2, 3))
        return feature_map, z_img
