from pathlib import Path

import pytest
import torch
from torch import nn

from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.taxonomy import load_taxonomy
from btxrd.models.image_encoder import EncoderOutput
from btxrd.training.trainer import train
from btxrd.training.predict import predict


class TinyEncoder(nn.Module):
    global_dim = token_dim = 16

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 16)

    def forward(self, image):
        tokens = self.proj(torch.nn.functional.adaptive_avg_pool2d(image, (7, 7)).flatten(2).transpose(1, 2))
        return EncoderOutput(tokens, tokens.mean(1), (7, 7))


@pytest.mark.parametrize('run,mixed', [
    ('f1_flat_decoder', None), ('h1_hier_decoder', None),
    ('h2_hier_pecl', None), ('h2n_hier_name', None),
    ('h2_hier_pecl', 'name'), ('h2n_hier_name', 'sub'),
])
def test_query_train_resume_predict(run, mixed, config_overrides, synthetic_assets, monkeypatch, tmp_path):
    monkeypatch.setattr('btxrd.models.model.build_image_encoder', lambda *a, **kw: TinyEncoder())
    taxonomy = load_taxonomy('configs/taxonomy/btxrd_8class.yaml', CLASS_NAMES)
    name_cache = tmp_path / 'names.pt'
    torch.save(dict(node_ids=taxonomy.node_ids, texts=[taxonomy.names[n] for n in taxonomy.node_ids],
                    embeddings=torch.randn(12, 768), model_name='ncbi/MedCPT-Query-Encoder',
                    max_length=64, pooling='cls', taxonomy_sha256=taxonomy.sha256), name_cache)
    values = config_overrides(run_name=run, **{'paths.node_name_cache': str(name_cache), 'train.epochs': 1,
                  'knowledge.dim': 16, 'knowledge.query_decoder.num_layers': 1, 'knowledge.query_decoder.ffn_dim': 32})
    if mixed:
        values['knowledge.pecl_prototype'] = mixed
    if run == 'h2n_hier_name' and mixed is None:
        values.update({'paths.kb_json': None, 'paths.text_embedding_cache': None})
    cfg = load_config(f'configs/{run}.yaml', values)
    result = train(cfg)
    checkpoint = Path(result['run_dir']) / 'last.pt'
    saved = torch.load(checkpoint, weights_only=False)
    if run == 'h2n_hier_name' and mixed is None:
        from btxrd.models.model import BTXRDModel
        from btxrd.training.predict import checkpoint_query_assets
        first = BTXRDModel(cfg, torch.randn(65, 768), query_assets=checkpoint_query_assets(saved), pretrained=False)
        second = BTXRDModel(cfg, torch.randn(65, 768) * 100, query_assets=checkpoint_query_assets(saved), pretrained=False)
        first.load_state_dict(saved['model_state'])
        second.load_state_dict(saved['model_state'])
        first.eval()
        second.eval()
        image = torch.randn(2, 3, 32, 32)
        assert torch.equal(first(image)['logits'], second(image)['logits'])
        normalized = torch.nn.functional.normalize(torch.load(name_cache, weights_only=False)['embeddings'], dim=-1)
        assert torch.equal(first.query_decoder.node_text, normalized)
        assert torch.equal(first.node_prototypes, normalized)
        assert saved['proposition_assignment'] == {}
        import json
        info = json.loads((Path(result['run_dir']) / 'run_info.json').read_text())
        assert info['query_source'] == info['pecl_prototype'] == 'name'
        assert info['uses_proposition_content'] is False
        assert not {'kb_json', 'text_embedding_cache'} & info['sha256'].keys()
    assert 'val_loss_pecl' in saved['history'][0]
    assert saved['node_ids'] == (CLASS_NAMES if run.startswith('f1') else taxonomy.node_ids)
    from btxrd.training.predict import verify_query_assets
    from btxrd.config import Config
    if not run.startswith('f1'):
        relocated = tmp_path / 'relocated.yaml'
        relocated.write_bytes(Path('configs/taxonomy/btxrd_8class.yaml').read_bytes())
        audit = load_config(f'configs/{run}.yaml', {
            **values, 'knowledge.taxonomy_file': str(relocated),
            **{'paths.' + n: None for n in ('kb_json', 'text_embedding_cache', 'node_name_cache')},
        }, require_training_assets=False)
        from btxrd.training.predict import _prediction_config
        _prediction_config(audit, saved['config'])
        verify_query_assets(audit, saved)
        relocated.unlink()
        with pytest.raises(FileNotFoundError):
            verify_query_assets(audit, saved)
    # Every consumed asset is independently auditable without its companions.
    for name in saved['asset_sha256']:
        audit = Config({'paths': {name: values['paths.' + name]}}, cfg.source_path)
        verify_query_assets(audit, saved)
    values['train.resume'] = True
    values['train.epochs'] = 2
    assert train(load_config(f'configs/{run}.yaml', values))['epochs_ran'] == 2
    values.update({'paths.kb_json': None, 'paths.text_embedding_cache': None, 'paths.node_name_cache': None})
    runtime = load_config(f'configs/{run}.yaml', values, require_training_assets=False)
    frame, extras = predict(runtime, checkpoint, return_extras=True)
    assert len(frame) == 8
    assert extras['node_logits'].shape == (8, len(saved['node_ids']))
    assert extras['cross_attn_weights'].shape == (8, len(saved['node_ids']), 49)
    sums = torch.tensor(frame.filter(like='p_').to_numpy().sum(1))
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


@pytest.mark.parametrize('name', ['kb_json', 'text_embedding_cache', 'node_name_cache'])
def test_query_asset_hash_audit(name, tmp_path):
    from btxrd.config import Config
    from btxrd.training.predict import verify_query_assets
    from btxrd.utils.io import sha256_file

    path = tmp_path / name
    path.write_bytes(b'original')
    checkpoint = {'asset_sha256': {name: sha256_file(path)},
                  'config': {'paths': {name: str(tmp_path / 'missing-stored')}}}
    cfg = Config({'paths': {name: str(path)}}, tmp_path / 'config.yaml')
    verify_query_assets(cfg, checkpoint)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash does not match'):
        verify_query_assets(cfg, checkpoint)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        verify_query_assets(cfg, checkpoint)
    cfg = Config({'paths': {}}, tmp_path / 'config.yaml')
    verify_query_assets(cfg, checkpoint)
    path.write_bytes(b'unrecorded')
    cfg = Config({'paths': {name: str(path)}}, tmp_path / 'config.yaml')
    with pytest.raises(ValueError, match='hash does not match'):
        verify_query_assets(cfg, {'asset_sha256': {}})