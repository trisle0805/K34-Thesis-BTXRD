"""CPU invariants and full synthetic M1–M4 lifecycle."""
import json
import subprocess
import pandas as pd
import pytest
import torch
from btxrd.constants import CLASS_NAMES
from btxrd.config import load_config
from btxrd.models.kb_multitask import soft_targets
from btxrd.data.locations import LOCATION_IDS, load_locations, location_definition
from btxrd.utils.io import sha256_file

NAMES = ['m1_densenet169_kb_multitask', 'm2_densenet169_kb_multitask_shuffled',
         'm3_densenet169_kb_multitask_no_location', 'm4_densenet169_kb_multitask_location_only']


def test_soft_jaccard():
    mapping = {name: [] for name in CLASS_NAMES}
    mapping[CLASS_NAMES[1]] = ['a', 'b']
    mapping[CLASS_NAMES[2]] = ['b', 'c']
    matrix = soft_targets(mapping, .1)
    assert torch.allclose(matrix.sum(1), torch.ones(8))
    assert matrix[1, 2] == .1 and matrix[1, 1] == .9
    assert torch.equal(matrix[0], torch.eye(8)[0])
    assert torch.equal(soft_targets(mapping, 0), torch.eye(8))


def test_original_attention_exact():
    from btxrd.models.proposition_guided import FixedQueryAttention
    original = subprocess.check_output(['git', 'show', '61b0623:src/btxrd/models/proposition_guided.py'], text=True)
    scope = {}
    exec(original, scope)
    torch.manual_seed(42)
    reference = scope['FixedQueryAttention'](1664).eval()
    torch.manual_seed(42)
    actual = FixedQueryAttention(1664).eval()
    queries, tokens = torch.randn(65, 768), torch.randn(2, 3, 1664)
    for left, right in zip(reference(queries, tokens), actual(queries, tokens)):
        assert torch.equal(left, right)


@pytest.mark.parametrize('invalid', ['missing', 'duplicate', 'fractional', 'normal', 'tumor'])
def test_location_invalid(tmp_path, invalid):
    frame = pd.DataFrame({'image_id': ['a', 'b'], 'label': ['normal', 'tumor']})
    table = pd.DataFrame({'image_id': ['a', 'b'], **{key: [0, 1] for key in LOCATION_IDS}})
    if invalid == 'missing': table = table.iloc[:1]
    if invalid == 'duplicate': table.loc[1, 'image_id'] = 'a'
    if invalid == 'fractional':
        table['L01'] = table['L01'].astype(float)
        table.loc[1, 'L01'] = .5
    if invalid == 'normal': table.loc[0, 'L01'] = 1
    if invalid == 'tumor': table.loc[1, LOCATION_IDS] = 0
    path = tmp_path / 'labels.csv'
    table.to_csv(path, index=False)
    with pytest.raises(ValueError): load_locations(path, frame, 'image_id', 'label')


@pytest.mark.parametrize('name', NAMES)
def test_lifecycle(config_overrides, synthetic_assets, tmp_path, monkeypatch, name):
    from test_integration import TinyImageEncoder
    from btxrd.models.model import BTXRDModel
    from btxrd.training.trainer import train, _build_model_inputs
    from btxrd.training.predict import predict
    monkeypatch.setattr('btxrd.models.model.DenseNet169Encoder', TinyImageEncoder)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    frame = pd.read_csv(synthetic_assets['splits_csv'])
    ids = frame.image_id.tolist()
    csv, cache, labels, loc_cache = [tmp_path / item for item in ('text.csv', 'text.pt', 'labels.csv', 'loc.pt')]
    pd.DataFrame({'image_id': ids, 'text': ['synthetic'] * len(ids)}).to_csv(csv, index=False)
    torch.save(dict(image_ids=ids, tokens=torch.randn(len(ids), 64, 768).half(), mask=torch.ones(len(ids), 64).bool(),
                    encoder_name='ncbi/MedCPT-Query-Encoder', max_length=64, text_csv_sha256=sha256_file(csv)), cache)
    table = pd.DataFrame({'image_id': ids, **{key: [int(str(label).lower() != 'normal') for label in frame.class8] for key in LOCATION_IDS}})
    table.to_csv(labels, index=False)
    definition = 'kb/location_propositions_v1.csv'
    torch.save(location_definition(definition) | {'embeddings': torch.randn(15, 768)}, loc_cache)
    changes = {'train.epochs': 1, 'train.amp': False, 'paths.text_csv': str(csv), 'paths.text_token_cache': str(cache),
               'paths.location_csv': definition, 'paths.location_labels_csv': str(labels), 'paths.location_embedding_cache': str(loc_cache)}
    path = f'configs/cv/{name}.yaml'
    def config(**extra):
        return load_config(path, config_overrides(output_dir=tmp_path, run_name=name, **(changes | extra)))
    cfg = config()
    embeddings, prop_ids, mapping, _ = _build_model_inputs(cfg)
    torch.manual_seed(7)
    actual = BTXRDModel(cfg, embeddings, mapping, prop_ids, pretrained=False).eval()
    torch.manual_seed(7)
    k0 = BTXRDModel(load_config('configs/cv/k0_densenet169_imgtxt.yaml', config_overrides()), pretrained=False).eval()
    assert all(torch.equal(value, actual.concat.state_dict()[key]) for key, value in k0.concat.state_dict().items())
    image, tokens, mask = torch.randn(2, 3, 32, 32), torch.randn(2, 4, 768), torch.ones(2, 4).bool()
    first = actual(image, text_tokens=tokens, text_mask=mask)
    assert torch.equal(first['logits'], k0(image, text_tokens=tokens, text_mask=mask)['logits'])
    with torch.no_grad(): actual.proposition_branch.scorer[-1].weight.add_(10)
    assert torch.equal(first['logits'], actual(image, text_tokens=tokens, text_mask=mask)['logits'])
    assert 'Q' not in dict(actual.proposition_branch.named_parameters())
    train(cfg)
    train(config(**{'train.epochs': 2, 'train.resume': True}))
    run = tmp_path / name
    history = pd.read_csv(run / 'history.csv')
    assert len(history) == 2 and 'train_loss_soft_ce' in history
    prediction, extras = predict(config(), run / 'last.pt', return_extras=True)
    assert extras['proposition_scores'].shape == (len(prediction), 80)
    assert 'auxiliary_metrics' in extras
    info = json.loads((run / 'run_info.json').read_text())
    assert 'soft_targets_sha256' in info
    import sys
    import numpy as np
    from scripts.evaluate import main
    overrides = config_overrides(output_dir=tmp_path, run_name=name, **changes)
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--config', path, '--checkpoint', str(run / 'last.pt'),
        '--out', str(run / 'eval'), '--set', *[f'{key}={json.dumps(value)}' for key, value in overrides.items()]])
    main()
    with np.load(run / 'eval' / 'proposition_scores_test.npz') as artifact:
        assert artifact['scores'].shape == (8, 80)
        assert artifact['prop_ids'].tolist()[-15:] == LOCATION_IDS
    saved = torch.load(run / 'last.pt', weights_only=False)
    saved['model_state']['proposition_branch.Q'][0, 0] += 1
    torch.save(saved, run / 'corrupt.pt')
    with pytest.raises(ValueError, match='mismatch'):
        predict(config(), run / 'corrupt.pt')