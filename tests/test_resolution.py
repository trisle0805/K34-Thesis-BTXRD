"""Input resolution and pretrained-grid compatibility contracts."""
import pytest
import torch

from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.models.image_encoder import build_image_encoder
from btxrd.models.model import BTXRDModel


@pytest.mark.parametrize('backbone,valid,invalid', [
    ('densenet169', 512, 240), ('resnet50', 512, 240),
    ('efficientnet_b3', 512, 240), ('vit_b16', 240, 232),
    ('biomedclip_vitb16', 240, 232), ('skelex', 240, 232),
])
def test_resolution_validation(config_overrides, backbone, valid, invalid):
    for size in (224, valid):
        cfg = load_config('configs/b0_image.yaml', config_overrides(**{
            'model.backbone': backbone, 'data.input_size': size}))
        assert cfg.data.input_size == size
    for size in (invalid, 0, -32, True, 224.0):
        with pytest.raises(ValueError, match='data.input_size'):
            load_config('configs/b0_image.yaml', config_overrides(**{
                'model.backbone': backbone, 'data.input_size': size}))


@pytest.mark.parametrize('backbone,dim,grid', [
    ('densenet169', 1664, 16), ('resnet50', 2048, 16),
    ('efficientnet_b3', 1536, 16), ('vit_b16', 768, 32),
    ('biomedclip_vitb16', 768, 32),
])
def test_real_encoder_512(config_overrides, backbone, dim, grid):
    if backbone == 'biomedclip_vitb16':
        pytest.importorskip('open_clip')
    if backbone == 'vit_b16':
        pytest.importorskip('transformers')
    cfg = load_config('configs/b0_image.yaml', config_overrides(**{
        'model.backbone': backbone, 'data.input_size': 512}))
    encoder = build_image_encoder(cfg, pretrained=False).eval()
    with torch.inference_mode():
        output = encoder(torch.zeros(1, 3, 512, 512))
    assert output.visual_tokens.shape == (1, grid * grid, dim)
    assert output.global_feature.shape == (1, dim)
    assert output.spatial_shape == (grid, grid)


@pytest.mark.parametrize('backbone', ['vit_b16', 'biomedclip_vitb16'])
def test_transformer_224_bitwise_legacy(config_overrides, backbone):
    if backbone == 'biomedclip_vitb16':
        pytest.importorskip('open_clip')
    else:
        pytest.importorskip('transformers')
    cfg = load_config('configs/b0_image.yaml', config_overrides(**{'model.backbone': backbone}))
    encoder = build_image_encoder(cfg, pretrained=False).eval()
    image = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        actual = encoder(image)
        if backbone == 'vit_b16':
            legacy = encoder.vit(pixel_values=image).last_hidden_state
        else:
            from timm.layers import Format
            encoder.trunk.dynamic_img_size = False
            encoder.trunk.patch_embed.flatten = True
            encoder.trunk.patch_embed.output_fmt = Format.NCHW
            legacy = encoder.trunk.forward_features(image)
    assert torch.equal(actual.visual_tokens, legacy[:, 1:])
    assert torch.equal(actual.global_feature, legacy[:, 0])


def test_a3_attention_512(config_overrides):
    cfg = load_config('configs/cv/a3_densenet169_kdecoder_pecl.yaml', config_overrides(**{
        'data.input_size': 512, 'model.pretrained': False}))
    ids = [f'p{i}' for i in range(65)]
    mapping = {name: ids[i::8] for i, name in enumerate(CLASS_NAMES)}
    model = BTXRDModel(cfg, torch.randn(65, 768), mapping, ids, pretrained=False).eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 512, 512))
    assert output['cross_attn_weights'].shape == (1, 65, 256)