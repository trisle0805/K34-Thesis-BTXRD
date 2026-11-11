"""Dual-branch auxiliary supervision and shuffled-control regressions."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from btxrd.config import load_config
from btxrd.knowledge.permutation import permute_embeddings
from btxrd.losses.classification import branch_classification_loss
from btxrd.training.predict import predict
from btxrd.training.trainer import train


def test_weighted_loss_and_zero_exact():
    torch.manual_seed(12)
    output = {key: torch.randn(5, 8, requires_grad=True)
              for key in ('logits', 'logits_img', 'logits_kb')}
    labels = torch.tensor([0, 3, 4, 7, 1])
    weights = torch.arange(1, 9, dtype=torch.float32)
    ce = torch.nn.CrossEntropyLoss(weight=weights)
    old = ce(output['logits'], labels)
    zero, final, branches = branch_classification_loss(output, labels, ce)
    assert torch.equal(old, zero) and torch.equal(old, final)
    assert torch.equal(torch.autograd.grad(old, output['logits'], retain_graph=True)[0],
                       torch.autograd.grad(zero, output['logits'], retain_graph=True)[0])
    actual, _, _ = branch_classification_loss(output, labels, ce, 0.7, 1.3)
    def manual(logits):
        selected = -logits.log_softmax(1)[torch.arange(5), labels]
        return (selected * weights[labels]).sum() / weights[labels].sum()
    assert torch.allclose(actual, manual(output['logits']) +
                          0.7 * manual(output['logits_img']) + 1.3 * manual(output['logits_kb']))
    actual.backward()
    assert all(value.grad is not None for value in output.values())


def test_permutation():
    ids = [f'P{i:03}' for i in range(1, 66)]
    vectors = torch.arange(65 * 4).reshape(65, 4)
    original = vectors.clone()
    rng = torch.get_rng_state().clone()
    shuffled, mapping = permute_embeddings(vectors, ids, 0)
    again, again_mapping = permute_embeddings(vectors, ids, 0)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(vectors, original)
    assert torch.equal(shuffled, again) and mapping == again_mapping
    assert set(mapping) == set(mapping.values()) == set(ids)
    assert not torch.equal(shuffled, vectors)
    for index, target in enumerate(ids):
        assert torch.equal(shuffled[index], vectors[ids.index(mapping[target])])
    unchanged, identity = permute_embeddings(vectors, ids, None)
    assert unchanged is vectors and all(key == value for key, value in identity.items())


@pytest.mark.parametrize('changes', [
    {'loss.aux_image_weight': -1}, {'loss.aux_knowledge_weight': -1},
    {'loss.aux_image_weight': 1}, {'knowledge.embedding_permutation_seed': True},
])
def test_invalid_aux_config(config_overrides, changes):
    with pytest.raises(ValueError):
        load_config('configs/b0_image.yaml', config_overrides(**changes))


@pytest.mark.parametrize('shuffled', [False, True])
def test_aux_smoke(config_overrides, synthetic_assets, tmp_path, monkeypatch, shuffled):
    from test_integration import TinyImageEncoder
    monkeypatch.setattr('btxrd.models.model.DenseNet169Encoder', TinyImageEncoder)
    name = 'r1_densenet_class_decoder_aux' + ('_shuffled' if shuffled else '')
    cfg = load_config(f'configs/{name}.yaml', config_overrides(
        output_dir=tmp_path, run_name=name, **{'train.epochs': 1, 'train.amp': False}))
    cache_before = synthetic_assets['cache'].read_bytes()
    train(cfg)
    run = tmp_path / name
    history = pd.read_csv(run / 'history.csv')
    for phase in ('train', 'val'):
        for branch in ('final', 'img', 'kb'):
            assert np.isfinite(history[f'{phase}_loss_ce_{branch}']).all()
        assert np.allclose(history[f'{phase}_loss_total'],
                           sum(history[f'{phase}_loss_ce_{branch}'] for branch in ('final', 'img', 'kb')))
    for branch in ('img', 'kb', 'avg'):
        assert history[f'val_macro_f1_{branch}'].between(0, 1).all()
    info = json.loads((run / 'run_info.json').read_text())
    assert info['knowledge_permuted'] is shuffled
    mapping = info['knowledge_embedding_source_ids']
    assert len(mapping) == 65 and set(mapping) == set(mapping.values())
    assert synthetic_assets['cache'].read_bytes() == cache_before
    for split in ('val', 'test'):
        frame, extras = predict(cfg, run / 'best.pt', split, True)
        assert extras['logits_img'].shape == extras['logits_kb'].shape == (8, 8)
        logits = extras['logits_img'] + extras['beta'] * extras['logits_kb']
        assert np.array_equal(logits.argmax(1), frame.y_pred.to_numpy())
        from scripts.evaluate import main
        import sys
        overrides = config_overrides(output_dir=tmp_path, run_name=name,
                                     **{'train.epochs': 1, 'train.amp': False})
        arguments = [f'{key}={json.dumps(value)}' for key, value in overrides.items()]
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--config', f'configs/{name}.yaml',
                            '--checkpoint', str(run / 'best.pt'), '--split', split,
                            '--out', str(run / 'evaluation'), '--set', *arguments])
        main()
        with np.load(run / 'evaluation' / f'branch_logits_{split}.npz') as artifact:
            for key in ('logits_img', 'logits_kb', 'beta'):
                assert np.array_equal(artifact[key], extras[key])


def test_old_r1_checkpoint_prediction_equal(config_overrides, tmp_path, monkeypatch):
    from test_integration import TinyImageEncoder
    monkeypatch.setattr('btxrd.models.model.DenseNet169Encoder', TinyImageEncoder)
    cfg = load_config('configs/r1_densenet_class_decoder.yaml', config_overrides(
        output_dir=tmp_path, run_name='old', **{'train.epochs': 1, 'train.amp': False}))
    train(cfg)
    checkpoint = tmp_path / 'old' / 'best.pt'
    expected, extras = predict(cfg, checkpoint, return_extras=True)
    saved = torch.load(checkpoint, weights_only=False)
    saved['config']['loss'].pop('aux_image_weight')
    saved['config']['loss'].pop('aux_knowledge_weight')
    saved['config']['knowledge'].pop('embedding_permutation_seed')
    legacy = tmp_path / 'legacy.pt'
    torch.save(saved, legacy)
    actual, old_extras = predict(cfg, legacy, return_extras=True)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    for key in ('logits_img', 'logits_kb', 'beta'):
        assert np.array_equal(extras[key], old_extras[key])