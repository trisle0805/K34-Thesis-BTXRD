"""PG invariants using CPU synthetic features only."""
import pytest
import torch
from btxrd.constants import CLASS_NAMES
from btxrd.models.proposition_guided import PropositionGuided, ImageTextConcat
from btxrd.data.patient_text import load_patient_cache
from btxrd.utils.io import sha256_file


def make_pg(use_text=True, pooling='learned'):
    ids = [f'P{i}' for i in range(65)]
    mapping = {name: ids[c::8] for c, name in enumerate(CLASS_NAMES)}
    return PropositionGuided(torch.randn(65, 768), mapping, ids, use_text=use_text, pooling=pooling)


@pytest.mark.parametrize('use_text', [False, True])
@pytest.mark.parametrize('pooling', ['learned', 'mean'])
def test_pg_shapes_gradients_masks(use_text, pooling):
    torch.set_num_threads(1)
    model = make_pg(use_text, pooling).eval()
    image, text = torch.randn(2, 3, 1664), torch.randn(2, 4, 768)
    mask = torch.tensor([[True, True, False, False]] * 2)
    out = model(image, text, mask)
    assert out['logits'].shape == (2, 8)
    assert out['attn_img'].shape == (2, 8, 65, 3)
    assert out['proposition_scores'].shape == (2, 65)
    alpha = out['class_pool_weights']
    assert torch.allclose(alpha.sum(-1), torch.ones(2, 8))
    assert (alpha[:, ~model.A] == 0).all()
    changed = text.clone()
    changed[:, 2:] = 1000
    assert torch.equal(out['logits'], model(image, changed, mask)['logits'])
    out['logits'].sum().backward()
    assert model.Q.grad is None
    assert model.image_attention.key.weight.grad is not None
    assert not any('query' in name for name, _ in model.named_parameters())


def test_concat_mask_and_empty():
    model = ImageTextConcat().eval()
    image, text = torch.randn(2, 1664), torch.randn(2, 3, 768)
    mask = torch.tensor([[True, False, False]] * 2)
    first = model(image, text, mask)['logits']
    text[:, 1:] = 999
    assert torch.equal(first, model(image, text, mask)['logits'])
    with pytest.raises(ValueError):
        model(image, text, torch.zeros_like(mask))


@pytest.mark.parametrize('mutation', ['encoder_name', 'max_length', 'text_csv_sha256', 'tokens', 'mask', 'image_ids', 'coverage'])
def test_cache_rejects_corruption(tmp_path, mutation):
    csv = tmp_path / 'text.csv'
    csv.write_text('image_id,text\na,example\n')
    cache = dict(image_ids=['a'], tokens=torch.ones(1, 2, 768).half(), mask=torch.ones(1, 2).bool(),
                 encoder_name='encoder', max_length=2, text_csv_sha256=sha256_file(csv))
    path = tmp_path / 'cache.pt'
    torch.save(cache, path)
    assert 'a' in load_patient_cache(path, csv, 'encoder', 2, ['a'])
    if mutation == 'tokens':
        cache['tokens'] = cache['tokens'].float()
    elif mutation == 'mask':
        cache['mask'].zero_()
    elif mutation == 'image_ids':
        cache['image_ids'] = ['']
    elif mutation != 'coverage':
        cache[mutation] = 'bad'
    torch.save(cache, path)
    with pytest.raises(ValueError):
        load_patient_cache(path, csv, 'encoder', 2, ['missing'] if mutation == 'coverage' else ['a'])


def test_two_epoch_cpu_smoke():
    torch.set_num_threads(1)
    model = make_pg()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad()
        output = model(torch.randn(2, 2, 1664), torch.randn(2, 3, 768), torch.ones(2, 3).bool())
        loss = torch.nn.functional.cross_entropy(output['logits'], torch.tensor([0, 1]))
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)


@pytest.mark.parametrize('name', ['k0_densenet169_imgtxt', 'k1_densenet169_pg', 'k2_densenet169_pg_shuffled'])
def test_train_resume_predict(config_overrides, synthetic_assets, tmp_path, monkeypatch, name):
    import json
    import pandas as pd
    from test_integration import TinyImageEncoder
    from btxrd.config import load_config
    from btxrd.training.trainer import train
    from btxrd.training.predict import predict
    monkeypatch.setattr('btxrd.models.model.DenseNet169Encoder', TinyImageEncoder)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    csv = tmp_path / 'patient.csv'
    ids = pd.read_csv(synthetic_assets['splits_csv']).image_id.tolist()
    pd.DataFrame({'image_id': ids, 'text': ['synthetic'] * len(ids)}).to_csv(csv, index=False)
    cache = tmp_path / 'patient.pt'
    torch.save(dict(image_ids=ids, tokens=torch.randn(len(ids), 4, 768).half(),
                    mask=torch.ones(len(ids), 4).bool(), encoder_name='ncbi/MedCPT-Query-Encoder',
                    max_length=4, text_csv_sha256=sha256_file(csv)), cache)
    changes = {'train.epochs': 1, 'train.amp': False}
    changes.update({'paths.text_csv': str(csv), 'paths.text_token_cache': str(cache),
                    'text.max_length': 4})
    path = f'configs/cv/{name}.yaml'
    def config(**extra):
        return load_config(path, config_overrides(output_dir=tmp_path, run_name=name, **(changes | extra)))
    train(config())
    train(config(**{'train.epochs': 2, 'train.resume': True}))
    run = tmp_path / name
    assert len(pd.read_csv(run / 'history.csv')) == 2
    info = json.loads((run / 'run_info.json').read_text())
    assert info['parameters']['blocks']
    assert info['use_text_branch'] is True
    assert {'text_csv', 'text_token_cache'} <= info['sha256'].keys()
    if name != 'k0_densenet169_imgtxt':
        assert set(info['parameters']['blocks']) == {'backbone', 'image_attention', 'text_attention', 'scorer', 'pooling', 'bias'}
        assert sum(b['total'] for b in info['parameters']['blocks'].values()) == info['parameters']['total']
    frame, extras = predict(config(), run / 'last.pt', return_extras=True)
    assert len(frame) == 8
    if True:
        import sys
        import numpy as np
        from scripts.evaluate import main
        overrides = config_overrides(output_dir=tmp_path, run_name=name, **changes)
        monkeypatch.setattr(sys, 'argv', ['evaluate', '--config', path, '--checkpoint', str(run / 'last.pt'),
            '--out', str(run / 'eval'), '--set', *[f'{k}={json.dumps(v)}' for k, v in overrides.items()]])
        main()
        if name == 'k0_densenet169_imgtxt':
            return
        with np.load(run / 'eval' / 'proposition_scores_test.npz') as artifact:
            assert set(artifact.files) == {'s', 'alpha', 'image_ids', 'prop_ids'}
            assert np.array_equal(artifact['s'], extras['proposition_scores'])
        assert extras['proposition_scores'].shape == (8, 65)
        assert extras['class_pool_weights'].shape == (8, 8, 65)
        saved = torch.load(run / 'last.pt', weights_only=False)
        saved['model_state']['pg.A'] = ~saved['model_state']['pg.A']
        torch.save(saved, run / 'corrupt.pt')
        with pytest.raises(ValueError, match='PG A'):
            predict(config(), run / 'corrupt.pt')

def test_exact_pg_configs_and_counts(config_overrides):
    from btxrd.config import load_config
    from btxrd.models.model import BTXRDModel
    configs = [load_config(f'configs/cv/{name}.yaml', config_overrides()) for name in
               ('k1_densenet169_pg', 'k2_densenet169_pg_shuffled')]
    left, right = [cfg.to_dict() for cfg in configs]
    left['run']['name'] = right['run']['name']
    left['knowledge']['embedding_permutation_seed'] = 0
    assert left == right
    ids = [f'P{i}' for i in range(65)]
    mapping = {name: ids[c::8] for c, name in enumerate(CLASS_NAMES)}
    counts = [sum(p.numel() for p in BTXRDModel(cfg, torch.randn(65, 768), mapping, ids, pretrained=False).parameters()) for cfg in configs]
    assert counts[0] == counts[1]