"""DenseNet-169 feature encoder compatible with the image-only baseline."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F
from torchvision.models import DenseNet169_Weights, densenet169


@dataclass
class EncoderOutput:
    visual_tokens: torch.Tensor
    global_feature: torch.Tensor
    spatial_shape: tuple[int, int] | None


class DenseNet169Encoder(nn.Module):
    output_dim = 1664
    token_dim = global_dim = 1664

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = DenseNet169_Weights.DEFAULT if pretrained else None
        self.features = densenet169(weights=weights).features

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = F.relu(self.features(image), inplace=False)
        z_img = feature_map.mean(dim=(2, 3))
        return EncoderOutput(feature_map.flatten(2).transpose(1, 2), z_img, tuple(feature_map.shape[-2:]))


class SKELEXEncoder(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        try:
            from transformers import ViTConfig, ViTModel
        except ImportError as exc:
            raise ImportError('SKELEX requires pip install -e .[text]') from exc
        saved = cfg.get_path('model.skelex.vit_config')
        if saved:
            self.vit = ViTModel(ViTConfig(**saved), add_pooling_layer=False)
        elif pretrained:
            self.vit, loading_info = ViTModel.from_pretrained(
                cfg.model.skelex.model_name, add_pooling_layer=False, output_loading_info=True)
            missing = [key for key in loading_info['missing_keys']
                       if key.startswith(('embeddings.', 'encoder.', 'layers.', 'layernorm.'))]
            if missing:
                raise ValueError(f'SKELEX checkpoint missing core weights: {missing}')
        else:
            self.vit = ViTModel(ViTConfig(**dict(cfg.model.skelex.test_config)), add_pooling_layer=False)
        self.token_dim = self.global_dim = self.vit.config.hidden_size

    def forward(self, image):
        hidden = self.vit(pixel_values=image).last_hidden_state
        size = self.vit.config.image_size // self.vit.config.patch_size
        return EncoderOutput(hidden[:, 1:], hidden[:, 0], (size, size))


def build_image_encoder(cfg, pretrained, densenet_class=None):
    if cfg.model.backbone == 'skelex':
        return SKELEXEncoder(cfg, pretrained)
    return (densenet_class or DenseNet169Encoder)(pretrained=pretrained)
