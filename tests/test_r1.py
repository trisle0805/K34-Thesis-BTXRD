"""Offline R1 encoder, masked decoder, and optimizer contracts."""
import numpy as np
import pytest
import torch
import json
from pathlib import Path
from PIL import Image
from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.data.transforms import load_skelex_rgb
from btxrd.models.image_encoder import SKELEXEncoder
from btxrd.models.model import BTXRDModel
from btxrd.training.optim import build_optimizer, build_scheduler


def inputs():
    ids = [f'p{i}' for i in range(16)]
    return torch.randn(16, 768), {name: ids[2*i:2*i+2] for i, name in enumerate(CLASS_NAMES)}, ids


@pytest.mark.parametrize('name,path', [
    ('b0', 'configs/b0_image.yaml'), ('b1', 'configs/b1_metadata.yaml'),
    ('b2', 'configs/b2_knowledge.yaml'), ('b3', 'configs/b3_contrastive.yaml'),
    ('a1', 'configs/ablations/a1_mean_aggregation.yaml')])
def test_actual_baseline_manifests(config_overrides, name, path):
    cfg = load_config(path, config_overrides())
    ids = [f'p{i}' for i in range(65)]
    classes = {cls: ids[i::8] for i, cls in enumerate(CLASS_NAMES)}
    model = BTXRDModel(cfg, torch.zeros(65, 768), classes, ids, pretrained=False)
    manifest = json.loads((Path(__file__).parent / 'fixtures' / f'{name}_80db443.json').read_text())
    assert {key: list(value.shape) for key, value in model.state_dict().items()} == manifest['state_dict']
    assert [key for key, _ in model.named_parameters()] == manifest['parameters']
    optimizer = build_optimizer(model, cfg)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]['lr'] == 1e-4
    assert optimizer.param_groups[0]['weight_decay'] == 1e-4
    assert [id(p) for p in optimizer.param_groups[0]['params']] == [id(p) for p in model.parameters()]


@pytest.mark.parametrize('scorer', ['shared', 'per_class'])
def test_decoder_residual_mask_gradients(config_overrides, scorer):
    cfg = load_config('configs/r1_skelex_class_decoder.yaml', config_overrides(**{
        'knowledge.decoder.scorer': scorer, 'model.pretrained': False}))
    model = BTXRDModel(cfg, *inputs(), pretrained=False).eval()
    output = model(torch.randn(2, 3, 224, 224))
    assert torch.allclose(output['logits'], output['logits_img'] + model.knowledge_logit_scale * output['logits_kb'])
    attention = output['class_attn_weights']
    mask = model.class_decoder.attention_mask
    assert attention.shape == (2, 8, 16)
    assert torch.count_nonzero(attention[:, mask]) == 0
    assert torch.allclose(attention.sum(-1), torch.ones(2, 8))
    propositions = output['prop_features'].detach()
    changed = propositions.clone()
    changed[:, mask[0]] += 100
    assert torch.allclose(model.class_decoder(propositions)[0][:, 0], model.class_decoder(changed)[0][:, 0])
    before = {name: value.clone() for name, value in model.named_buffers()}
    output['logits'].square().sum().backward()
    for name, parameter in model.named_parameters():
        if name.startswith(('class_decoder.', 'cross_attention.query_proj', 'cross_attention.token_proj')) or name == 'knowledge_logit_scale':
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert all(torch.equal(value, before[name]) for name, value in model.named_buffers())


def test_tiny_vit_and_llrd(config_overrides):
    cfg = load_config('configs/r1_skelex_class_decoder.yaml', config_overrides(**{'model.pretrained': False}))
    model = BTXRDModel(cfg, *inputs(), pretrained=False).eval()
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        encoded = model.image_encoder(image)
        hidden = model.image_encoder.vit(image).last_hidden_state
    assert encoded.visual_tokens.shape == (1, 196, 32)
    assert encoded.spatial_shape == (14, 14)
    assert torch.equal(encoded.global_feature, hidden[:, 0])
    assert torch.equal(encoded.visual_tokens, hidden[:, 1:])
    optimizer = build_optimizer(model, cfg)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    assignments = {id(p): group for group in optimizer.param_groups for p in group['params']}
    block_lrs = {}
    for name, parameter in model.named_parameters():
        if name.startswith('image_encoder.vit.'):
            import re
            match = re.search(r'(?:encoder\.layer|layers)\.(\d+)\.', name)
            expected = int(match.group(1)) + 1 if match else (0 if '.embeddings.' in name else 3)
            assert assignments[id(parameter)]['layer_id'] == expected, name
            block_lrs[expected] = assignments[id(parameter)]['lr']
    assert set(block_lrs) == {0, 1, 2, 3}
    assert len(set(block_lrs.values())) == 4
    assert len(parameters) == len(set(map(id, parameters))) == len(list(model.parameters()))
    for group in optimizer.param_groups:
        assert group['lr'] == pytest.approx(cfg.train.lr * 0.75 ** (4 - group['layer_id']))
        if any(p is model.knowledge_logit_scale for p in group['params']):
            assert group['weight_decay'] == 0 and group['layer_id'] == 4
    scheduler = build_scheduler(optimizer, cfg, 3)
    assert max(group['lr'] for group in optimizer.param_groups) == pytest.approx(cfg.train.lr / 10)
    for _ in range(cfg.train.epochs * 2):
        optimizer.step()
        scheduler.step()
    assert max(group['lr'] for group in optimizer.param_groups) == pytest.approx(0)


def test_skelex_minmax():
    constant = np.full((5, 5), 10, dtype=np.uint8)
    assert np.asarray(load_skelex_rgb(Image.fromarray(constant))).max() == 0
    ramp = np.arange(10, 21, dtype=np.uint8).reshape(1, -1)
    actual = np.asarray(load_skelex_rgb(Image.fromarray(ramp)))
    expected = np.clip((ramp.astype(np.float32) - 10) / 10 * 255, 0, 255).astype(np.uint8)
    assert np.array_equal(actual[:, :, 0], expected)

def test_leaf_names_and_recipe_overrides(config_overrides):
    paths = sorted(Path('configs').rglob('*.yaml'))
    paths = [path for path in paths if path.name != 'base.yaml']
    names = []
    for path in paths:
        overrides = config_overrides(**{'run.name': None})
        cfg = load_config(path, overrides)
        assert cfg.run.name == path.stem
        names.append(cfg.run.name)
        assert load_config(path, config_overrides(run_name='notebook')).run.name == 'notebook'
    assert len(set(names)) == 8
    cfg = load_config('configs/r1_skelex_class_decoder.yaml', config_overrides(**{
        'train.early_stopping.enabled': True, 'train.lr': 0.002,
        'train.warmup_epochs': 3, 'recipes.skelex.train.epochs': 17}))
    assert cfg.train.early_stopping.enabled is True
    assert cfg.train.lr == 0.002 and cfg.train.warmup_epochs == 3
    assert cfg.train.epochs == 17

def test_skelex_configured_transforms():
    from btxrd.data.transforms import build_train_transform, build_eval_transform
    data = {'preprocessing': 'skelex_minmax_rgb', 'input_size': 48,
            'augmentation': {'degrees': 12, 'translate': [0.1, 0.2], 'scale': [0.8, 1.2]}}
    transform = build_train_transform(data)
    assert transform.transforms[1].size == (48, 48)
    affine = transform.transforms[2]
    assert affine.degrees == [-12.0, 12.0]
    assert affine.translate == (0.1, 0.2) and affine.scale == (0.8, 1.2)
    assert build_eval_transform(data)(Image.new('L', (10, 10))).shape == (3, 48, 48)

def test_missing_pretrained_core_weights(config_overrides, monkeypatch):
    from transformers import ViTModel
    cfg = load_config('configs/s0_skelex_image.yaml', config_overrides())
    tiny = SKELEXEncoder(cfg, pretrained=False).vit
    def fake_load(*args, **kwargs):
        assert kwargs == {'add_pooling_layer': False, 'output_loading_info': True}
        return tiny, {'missing_keys': ['embeddings.cls_token', 'layers.0.weight']}
    monkeypatch.setattr(ViTModel, 'from_pretrained', fake_load)
    with pytest.raises(ValueError, match='embeddings.cls_token.*layers.0.weight'):
        SKELEXEncoder(cfg, pretrained=True)

def test_llrd_incomplete_backbone_rejected(config_overrides):
    cfg = load_config('configs/s0_skelex_image.yaml', config_overrides())
    model = BTXRDModel(cfg, pretrained=False)
    for name, parameter in model.named_parameters():
        if 'layers.0.' in name or 'encoder.layer.0.' in name:
            parameter.requires_grad_(False)
    with pytest.raises(ValueError, match='Invalid ViT LLRD layer ids'):
        build_optimizer(model, cfg)


@pytest.mark.parametrize('changes', [
    {'model.residual_knowledge_logits': False}, {'model.use_metadata': True},
    {'model.use_knowledge': False},
    {'model.use_contrastive': True, 'loss.contrastive_weight': 0.1},
    {'knowledge.decoder.scorer': 'invalid'},
])
def test_decoder_validation(config_overrides, changes):
    with pytest.raises(ValueError):
        load_config('configs/r1_densenet_class_decoder.yaml', config_overrides(**changes))


def test_r1_config_derived_differences(config_overrides):
    dense = load_config('configs/r1_densenet_class_decoder.yaml', config_overrides()).to_dict()
    skelex = load_config('configs/r1_skelex_class_decoder.yaml', config_overrides()).to_dict()
    for key in ('backbone',):
        dense['model'].pop(key)
        skelex['model'].pop(key)
    for key in ('recipe', 'epochs', 'lr', 'weight_decay', 'grad_accum', 'clip_grad_norm'):
        dense['train'].pop(key)
        skelex['train'].pop(key)
    for key in ('preprocessing', 'batch_size'):
        dense['data'].pop(key)
        skelex['data'].pop(key)
    assert dense == skelex


def test_dense_r1_train_predict(config_overrides, tmp_path, monkeypatch):
    from test_integration import TinyImageEncoder
    from btxrd.training.trainer import train
    from btxrd.training.predict import predict
    monkeypatch.setattr('btxrd.models.model.DenseNet169Encoder', TinyImageEncoder)
    cfg = load_config('configs/r1_densenet_class_decoder.yaml', config_overrides(
        output_dir=tmp_path, run_name='dense', **{'train.epochs': 1, 'train.amp': False}))
    train(cfg)
    frame, extras = predict(cfg, tmp_path / 'dense' / 'best.pt', return_extras=True)
    assert len(frame) == 8 and extras['logits_kb'].shape == (8, 8)


@pytest.mark.parametrize('path', ['configs/s0_skelex_image.yaml', 'configs/r1_skelex_class_decoder.yaml'])
def test_skelex_train_predict_resume(config_overrides, tmp_path, path, monkeypatch):
    from btxrd.training.trainer import train
    from btxrd.training.predict import predict
    cfg = load_config(path, config_overrides(output_dir=tmp_path, run_name='tiny', **{
        'model.pretrained': False, 'train.epochs': 1, 'train.amp': False}))
    result = train(cfg)
    frame, extras = predict(cfg, tmp_path / 'tiny' / 'best.pt', return_extras=True)
    assert len(frame) == 8
    if cfg.model.use_knowledge:
        assert extras['class_attn_weights'].shape == (8, 8, 65)
    resumed = cfg.to_dict()
    resumed['train']['resume'] = True
    from btxrd.config import Config
    assert train(Config(resumed)) == result