import pytest
import torch
from torch.nn import functional as F

from btxrd.config import load_config
from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.taxonomy import load_taxonomy, flat_taxonomy
from btxrd.models.hierarchy import hierarchical_log_probs
from btxrd.models.query_decoder import QueryDecoder
from btxrd.losses.pecl import pecl_loss
from btxrd.training.metrics import hierarchical_metrics
from btxrd.utils.io import sanitize_for_json


def tree():
    return load_taxonomy('configs/taxonomy/btxrd_8class.yaml', CLASS_NAMES)


def test_weighted_path_ce_and_pecl():
    taxonomy = tree()
    logits = torch.randn(8, 12, requires_grad=True)
    labels = torch.arange(8)
    leaf, _ = hierarchical_log_probs(logits, taxonomy)
    index = {n: i for i, n in enumerate(taxonomy.node_ids)}
    manual = []
    for row, c in enumerate(CLASS_NAMES):
        terms = []
        for n in taxonomy.paths[taxonomy.leaf_of_class[c]]:
            siblings = taxonomy.children[taxonomy.parent[n]]
            terms.append(logits[row, [index[s] for s in siblings]].log_softmax(0)[siblings.index(n)])
        manual.append(-torch.stack(terms).sum())
    manual = torch.stack(manual)
    weights = torch.arange(1., 9.)
    assert torch.allclose(F.cross_entropy(leaf, labels, weight=weights), (manual * weights).sum() / weights.sum())
    u, prototypes = F.normalize(torch.randn(8, 24), dim=-1), F.normalize(torch.randn(12, 24), dim=-1)
    expected = hierarchical_log_probs(u @ prototypes.T / .07, taxonomy)[0]
    assert torch.allclose(pecl_loss(u, prototypes, labels, taxonomy), F.nll_loss(expected, labels))
    flat = flat_taxonomy(CLASS_NAMES)
    assert torch.allclose(pecl_loss(u, prototypes[:8], labels, flat), F.cross_entropy(u @ prototypes[:8].T / .07, labels))


@pytest.mark.parametrize('tokens,nodes', [(49, 8), (196, 12)])
def test_decoder_gradients_and_frozen_vectors(tokens, nodes):
    decoder = QueryDecoder(16, F.normalize(torch.randn(nodes, 768), dim=-1), dim=16,
                           num_layers=2, num_heads=4, ffn_dim=32, dropout=0)
    original = decoder.node_text.clone()
    output, weights = decoder(torch.randn(2, tokens, 16))
    assert output.shape == (2, nodes) and weights.shape == (2, nodes, tokens)
    output.square().sum().backward()
    for name in ('token_proj', 'query_proj', 'layers', 'head'):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in getattr(decoder, name).parameters())
    torch.optim.SGD(decoder.parameters(), lr=.1).step()
    assert torch.equal(original, decoder.node_text)
    assert 'node_text' not in dict(decoder.named_parameters())
    assert sum(isinstance(m, torch.nn.Linear) and m.out_features == 1 for m in decoder.modules()) == 1


def test_clinical_metrics_normal_prediction_is_false_negative():
    normal, benign, malignant = [CLASS_NAMES.index(c) for c in ('normal', 'osteochondroma', 'osteosarcoma')]
    values = hierarchical_metrics([normal, benign, malignant], [normal, normal, malignant], tree())
    assert values['benign_vs_malignant_f1'] == .5
    assert values['severe_error_rate'] == pytest.approx(1 / 3)
    assert values['node_accuracy_tumour'] == .5
    values = hierarchical_metrics([normal], [normal], tree())
    assert sanitize_for_json(values)['benign_vs_malignant_f1'] is None


@pytest.mark.parametrize('overrides', [
    {'knowledge.query_source': 'name'}, {'knowledge.query_source': 'own'},
    {'knowledge.hierarchy': 'tree'}, {'model.use_metadata': True},
    {'knowledge.query_decoder.num_layers': 0}, {'loss.pecl_temperature': 0},
    {'loss.pecl_weight': -1},
])
def test_query_invalid_configs(config_overrides, overrides):
    with pytest.raises(ValueError):
        load_config('configs/f1_flat_decoder.yaml', config_overrides(**overrides))