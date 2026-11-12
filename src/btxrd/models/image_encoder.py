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

    @property
    def num_layers(self):
        return self.vit.config.num_hidden_layers

    def layer_id(self, name):
        import re
        if '.embeddings.' in name:
            return 0
        match = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', name)
        return int(match.group(1)) + 1 if match else self.num_layers + 1

    def forward(self, image):
        hidden = self.vit(pixel_values=image, interpolate_pos_encoding=True).last_hidden_state
        patch_size = self.vit.config.patch_size
        patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        return EncoderOutput(hidden[:, 1:], hidden[:, 0],
                             (image.shape[-2] // patch_h, image.shape[-1] // patch_w))


class CNNEncoder(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super().__init__()
        from torchvision import models
        if backbone == 'resnet50':
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            self.features = nn.Sequential(*list(net.children())[:-2])
            self.token_dim = self.global_dim = 2048
        else:
            net = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
            self.features = net.features
            self.token_dim = self.global_dim = 1536

    def forward(self, image):
        features = self.features(image)
        return EncoderOutput(features.flatten(2).transpose(1, 2), features.mean((2, 3)), tuple(features.shape[-2:]))

class ViTEncoder(SKELEXEncoder):
    def __init__(self, cfg, pretrained=True):
        nn.Module.__init__(self)
        from transformers import ViTConfig, ViTModel
        self.vit = (ViTModel.from_pretrained('google/vit-base-patch16-224', add_pooling_layer=False)
                    if pretrained else ViTModel(ViTConfig(), add_pooling_layer=False))
        self.token_dim = self.global_dim = self.vit.config.hidden_size

class BiomedCLIPEncoder(nn.Module):
    def __init__(self, cfg, pretrained=True):
        super().__init__()
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError('BiomedCLIP requires pip install -e .[biomedclip]') from exc
        if pretrained:
            model = open_clip.create_model('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            self.trunk = model.visual.trunk
            from timm.layers import Format
            self.trunk.dynamic_img_size = True
            self.trunk.patch_embed.strict_img_size = False
            self.trunk.patch_embed.flatten = False
            self.trunk.patch_embed.output_fmt = Format.NHWC
        else:
            from timm.models.vision_transformer import VisionTransformer
            self.trunk = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=0, dynamic_img_size=True)
        self.token_dim = self.global_dim = 768

    @property
    def num_layers(self):
        return len(self.trunk.blocks)

    def layer_id(self, name):
        import re
        if any(name.startswith('trunk.' + key) for key in ('patch_embed', 'cls_token', 'pos_embed')):
            return 0
        match = re.search(r'blocks\.(\d+)\.', name)
        return int(match.group(1)) + 1 if match else self.num_layers + 1

    def forward(self, image):
        hidden = self.trunk.forward_features(image)
        patch_size = self.trunk.patch_embed.patch_size
        patch_h, patch_w = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        grid = (image.shape[-2] // patch_h, image.shape[-1] // patch_w)
        if grid[0] * grid[1] != hidden.shape[1] - 1:
            raise ValueError('BiomedCLIP token count does not match the input patch grid')
        return EncoderOutput(hidden[:, 1:], hidden[:, 0], grid)

def build_image_encoder(cfg, pretrained, densenet_class=None):
    backbone = cfg.model.backbone
    if backbone == 'skelex':
        return SKELEXEncoder(cfg, pretrained)
    if backbone == 'vit_b16':
        return ViTEncoder(cfg, pretrained)
    if backbone == 'biomedclip_vitb16':
        return BiomedCLIPEncoder(cfg, pretrained)
    if backbone in {'resnet50', 'efficientnet_b3'}:
        return CNNEncoder(backbone, pretrained)
    return (densenet_class or DenseNet169Encoder)(pretrained=pretrained)
