from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torchvision import transforms
from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.data.dataset import build_dataloaders
from btxrd.data.transforms import build_train_transform, build_eval_transform
from btxrd.models.image_encoder import EncoderOutput, CNNEncoder, SKELEXEncoder
from btxrd.models.model import BTXRDModel
from btxrd.models.class_decoder import build_class_prototypes
from btxrd.training.trainer import train, _build_model_inputs
from btxrd.training.predict import predict
from btxrd.training.optim import build_optimizer


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path('scripts') / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TinyEncoder(nn.Module):
    token_dim = global_dim = 8
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
    def forward(self, image):
        x = self.conv(torch.nn.functional.adaptive_avg_pool2d(image, (7, 7)))
        return EncoderOutput(x.flatten(2).transpose(1, 2), x.mean((2, 3)), (7, 7))


@pytest.mark.parametrize('path', sorted(Path('configs/cv').glob('[ab]*.yaml')) + list(Path('configs/pilot').glob('*.yaml')))
def test_all_protocol_configs_train_predict_resume(path, config_overrides, monkeypatch):
    import btxrd.models.model as module
    from btxrd.models.image_encoder import SKELEXEncoder
    def encoder(cfg, pretrained, **kwargs):
        if cfg.model.backbone in {'vit_b16', 'biomedclip_vitb16'}:
            return SKELEXEncoder(cfg, False)
        return TinyEncoder()
    monkeypatch.setattr(module, 'build_image_encoder', encoder)
    overrides = config_overrides(run_name=path.stem, **{'train.epochs': 1, 'knowledge.dim': 8, 'knowledge.num_heads': 2})
    cfg = load_config(path, overrides)
    result = train(cfg)
    history = pd.read_csv(Path(result['run_dir']) / 'history.csv')
    assert len(history) == 1
    assert ('train_loss_pecl' in history) == cfg.model.use_pecl
    predictions = predict(cfg, Path(result['run_dir']) / 'best.pt')
    assert len(predictions) == 8
    resume = load_config(path, {**overrides, 'train.resume': True})
    assert train(resume)['epochs_ran'] == 1


@pytest.mark.parametrize('preprocessing', [None, 'imagenet_gray3', 'clip_gray3'])
def test_vit_preprocessing(config_overrides, preprocessing):
    from PIL import Image

    overrides = {} if preprocessing is None else {'data.preprocessing': preprocessing}
    cfg = load_config('configs/cv/b_vit_b16.yaml', config_overrides(**overrides))
    assert cfg.data.preprocessing == 'vit_gray3'
    for build_transform in (build_train_transform, build_eval_transform):
        transform = build_transform(cfg.data)
        assert any(isinstance(t, transforms.Grayscale) and t.num_output_channels == 3
                   for t in transform.transforms)
        normalization = transform.transforms[-1]
        assert isinstance(normalization, transforms.Normalize)
        assert list(normalization.mean) == [0.5, 0.5, 0.5]
        assert list(normalization.std) == [0.5, 0.5, 0.5]
        assert torch.equal(transform(Image.new('RGB', (32, 32), 0)),
                           -torch.ones(3, cfg.data.input_size, cfg.data.input_size))


def test_augmentation_sampling_and_weights(config_overrides):
    cfg = load_config('configs/b0_image.yaml', config_overrides())
    assert [type(t) for t in build_train_transform(cfg.data).transforms] == [transforms.Resize, transforms.RandomAffine, transforms.Grayscale, transforms.ToTensor, transforms.Normalize]
    cfg = load_config('configs/cv/b0_densenet169.yaml', config_overrides())
    assert [type(t) for t in build_train_transform(cfg.data).transforms] == [transforms.Resize, transforms.RandomHorizontalFlip, transforms.RandomAffine, transforms.ColorJitter, transforms.Grayscale, transforms.ToTensor, transforms.Normalize]
    data = build_dataloaders(cfg)
    sampler = data.loaders['train'].sampler
    assert sampler.replacement and sampler.num_samples == 8
    assert torch.equal(sampler.weights, torch.ones(8, dtype=torch.double))
    state = data.train_generator.get_state()
    expected = list(sampler)
    resumed = build_dataloaders(cfg, train_generator_state=state)
    assert list(resumed.loaders['train'].sampler) == expected
    cfg = load_config('configs/b0_image.yaml', config_overrides(**{'loss.class_weighting': 'train_split'}))
    assert torch.equal(build_dataloaders(cfg).class_weights, torch.ones(8))
    with pytest.raises(ValueError, match='double compensate'):
        load_config('configs/b0_image.yaml', config_overrides(**{'data.sampler': 'sqrt_inverse'}))


def test_pecl_independent_and_permuted(config_overrides, monkeypatch):
    import btxrd.models.model as module
    monkeypatch.setattr(module, 'build_image_encoder', lambda *a, **kw: TinyEncoder())
    cfg = load_config('configs/cv/a1_densenet169_pecl.yaml', config_overrides())
    emb, ids, mapping, _ = _build_model_inputs(cfg)
    model = BTXRDModel(cfg, emb, mapping, ids)
    assert model.cross_attention is None and not hasattr(model, 'class_decoder')
    assert not model.pecl_prototypes.requires_grad
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    labels = torch.tensor([0, 2])
    logits = output['pecl_u'] @ model.pecl_prototypes.T / .07
    manual = -torch.log_softmax(logits, 1)[torch.arange(2), labels].mean()
    assert torch.equal(manual, torch.nn.functional.cross_entropy(logits, labels))
    shuffled = load_config('configs/cv/a1_densenet169_pecl.yaml', config_overrides(**{'knowledge.embedding_permutation_seed': 0}))
    perm, _, _, _ = _build_model_inputs(shuffled)
    other = BTXRDModel(shuffled, perm, mapping, ids)
    assert torch.equal(other.pecl_prototypes, build_class_prototypes(perm, mapping, ids))
    assert not torch.equal(model.pecl_prototypes, other.pecl_prototypes)
    with pytest.raises(ValueError, match='mutually exclusive'):
        load_config('configs/cv/a1_densenet169_pecl.yaml', config_overrides(**{'model.use_contrastive': True, 'loss.contrastive_weight': .1}))


@pytest.mark.parametrize('name,dim', [('resnet50', 2048), ('efficientnet_b3', 1536)])
def test_cnn_shapes(name, dim):
    model = CNNEncoder(name, False).eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 224, 224))
    assert output.visual_tokens.shape == (1, 49, dim)
    assert output.global_feature.shape == (1, dim)


def test_fold_generation_and_report(tmp_path):
    maker = script('make_cv_folds')
    reporter = script('cv_report')
    rows = [{'image_id': f'{c}_{g}_{v}', 'class8': name, 'case': f'{c}_{g}'} for c, name in enumerate(CLASS_NAMES) for g in range(20) for v in range(2)]
    frame = pd.DataFrame(rows)
    splits, groups, out = [tmp_path / x for x in ['splits.csv', 'groups.csv', 'folds.csv']]
    frame[['image_id', 'class8']].to_csv(splits, index=False)
    frame[['image_id', 'case']].to_csv(groups, index=False)
    folds, audit = maker.make_folds(splits, groups, 'case', out, inner_val_splits=2)
    first = out.read_bytes()
    maker.make_folds(splits, groups, 'case', out, inner_val_splits=2)
    assert out.read_bytes() == first and audit['each_image_test_once']
    for k in range(5):
        assert folds.groupby('group_id')[f'split_fold{k}'].nunique().max() == 1
    frame.iloc[1:][['image_id', 'case']].to_csv(groups, index=False)
    with pytest.raises(ValueError, match='Missing groups'):
        maker.make_folds(splits, groups, 'case', out, inner_val_splits=2)
    missing, audit = maker.make_folds(splits, groups, 'case', out, inner_val_splits=2, allow_missing_groups=True)
    assert audit['missing_group_image_ids'] == ['0_0_0']
    folds.to_csv(out, index=False)
    configs = [reporter.BASELINES[0], reporter.A1, reporter.A2, reporter.A3, reporter.A4]
    for config in configs:
        for k in range(5):
            f = folds[folds.fold == k][['image_id', 'class8']].copy()
            f['y_true'] = f.class8.map(dict(zip(CLASS_NAMES, range(8))))
            f['y_pred'] = f.y_true
            for i, col in enumerate(reporter.PROBS):
                f[col] = (f.y_true == i).astype(float)
            target = tmp_path / 'cv' / config / f'{config}_fold{k}' / 'evaluation'
            target.mkdir(parents=True)
            f.to_csv(target / 'predictions_test.csv', index=False)
    with pytest.warns(UserWarning, match='incomplete'):
        _, pairs = reporter.report(tmp_path / 'cv', out, configs + ['missing'], tmp_path / 'report', n_boot=5)
    pooled = pd.read_csv(tmp_path / 'report' / 'metrics_pooled.csv')
    assert (pooled.macro_f1 == 1).all()
    assert any(p['question'] == 'Q3' and p['b'] == reporter.A2 for p in pairs)
    assert all(p['macro_f1_diff'] == p['ci_low'] == p['ci_high'] == 0 for p in pairs)
    assert all(np.array_equal(a, b) for a, b in zip(reporter.bootstrap_indices(folds.group_id, 3, 2026), reporter.bootstrap_indices(folds.group_id, 3, 2026)))
    # A non-perfect comparator catches pairing/alignment and metric regressions.
    from sklearn.metrics import f1_score
    for k in range(5):
        path = tmp_path / 'cv' / configs[0] / f'{configs[0]}_fold{k}' / 'evaluation' / 'predictions_test.csv'
        f = pd.read_csv(path)
        f['y_pred'] = (f.y_true + 1) % 8
        for i, col in enumerate(reporter.PROBS):
            f[col] = (f.y_pred == i).astype(float)
        f.iloc[::-1].to_csv(path, index=False)
    complete, pairs = reporter.report(tmp_path / 'cv', out, configs, tmp_path / 'imperfect', n_boot=7)
    q1 = next(p for p in pairs if p['question'] == 'Q1')
    assert q1['macro_f1_diff'] == f1_score(complete[reporter.A2].y_true, complete[reporter.A2].y_pred, average='macro')
    assert q1['ci_low'] == q1['ci_high'] == 1
    assert q1['p_one_sided'] == 0 and q1['folds_won'] == 5

@pytest.mark.parametrize('backbone', ['vit_b16', 'biomedclip_vitb16'])
def test_transformer_shapes_and_llrd(backbone, config_overrides, monkeypatch):
    import sys
    import types
    import transformers
    from transformers import ViTConfig, ViTModel
    from btxrd.models.image_encoder import ViTEncoder, BiomedCLIPEncoder
    cfg = load_config('configs/cv/b_vit_b16.yaml', config_overrides())
    if backbone == 'vit_b16':
        small_config = ViTConfig(image_size=224, patch_size=16, hidden_size=32,
                                num_hidden_layers=2, num_attention_heads=4, intermediate_size=64)
        monkeypatch.setattr(transformers, 'ViTConfig', lambda: small_config)
        encoder = ViTEncoder(cfg, False)
        dim = 32
    else:
        class Trunk(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Linear(3, 768)
                self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
                self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
                self.blocks = nn.ModuleList([nn.Linear(768, 768), nn.Linear(768, 768)])
                self.norm = nn.LayerNorm(768)
            def forward_features(self, image):
                x = self.patch_embed(torch.nn.functional.adaptive_avg_pool2d(image, (14,14)).flatten(2).transpose(1,2))
                x = torch.cat([self.cls_token.expand(len(image), -1, -1), x], 1) + self.pos_embed
                for block in self.blocks:
                    x = block(x)
                return self.norm(x)
        monkeypatch.setitem(sys.modules, 'open_clip', types.SimpleNamespace(create_model=lambda *a: types.SimpleNamespace(visual=types.SimpleNamespace(trunk=Trunk()))))
        encoder = BiomedCLIPEncoder(cfg, True)
        dim = 768
    model = nn.Module()
    model.image_encoder = encoder
    model.head = nn.Linear(dim, 8)
    optimizer = build_optimizer(model, cfg)
    assert {g['layer_id'] for g in optimizer.param_groups} == set(range(5))
    with torch.no_grad():
        output = encoder(torch.zeros(1, 3, 224, 224))
    assert output.visual_tokens.shape == (1, 196, dim)
    assert output.global_feature.shape == (1, dim)


def test_unequal_counts_and_legacy_rng(config_overrides, synthetic_assets):
    from PIL import Image
    from btxrd.constants import IMAGENET_MEAN, IMAGENET_STD
    cfg = load_config('configs/b0_image.yaml', config_overrides())
    current = build_train_transform(cfg.data)
    legacy = transforms.Compose([transforms.Resize((cfg.data.input_size, cfg.data.input_size)),
        transforms.RandomAffine(degrees=cfg.data.augmentation.degrees,
            translate=tuple(cfg.data.augmentation.translate), scale=tuple(cfg.data.augmentation.scale)),
        transforms.Grayscale(num_output_channels=3), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    image = Image.fromarray(np.arange(32*32, dtype=np.uint8).reshape(32,32))
    torch.manual_seed(123)
    a = current(image)
    state = torch.get_rng_state()
    torch.manual_seed(123)
    b = legacy(image)
    assert torch.equal(a,b) and torch.equal(state, torch.get_rng_state())
    path = cfg.paths.splits_csv
    frame = pd.read_csv(path)
    extra = frame[frame['split'] == 'train'].iloc[[0]].copy()
    extra['image_id'] = 'extra_1'
    extra2 = extra.copy()
    extra2['image_id'] = 'extra_2'
    pd.concat([frame, extra, extra2]).to_csv(path, index=False)
    for image_id in ('extra_1', 'extra_2'):
        image.save(Path(cfg.paths.image_dir) / (image_id + cfg.data.image_ext))
    cfg = load_config('configs/cv/b0_densenet169.yaml', config_overrides())
    bundle = build_dataloaders(cfg)
    weights = bundle.loaders['train'].sampler.weights
    assert torch.isclose(weights.min(), torch.tensor(3**-.5, dtype=torch.double))
    assert weights.max() == 1
    cfg = load_config('configs/b0_image.yaml', config_overrides(**{'loss.class_weighting':'train_split'}))
    ce = build_dataloaders(cfg).class_weights
    assert torch.isclose(ce.mean(), torch.tensor(1.))
    assert torch.isclose(ce.max()/ce.min(), torch.tensor(3**.5))
