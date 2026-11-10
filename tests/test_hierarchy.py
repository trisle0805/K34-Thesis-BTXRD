"""CPU checks for standalone hierarchy and decoder primitives."""
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.taxonomy import flat_taxonomy, load_taxonomy
from btxrd.models.hierarchy import hierarchical_log_probs
from btxrd.models.query_decoder import QueryDecoder
from btxrd.losses.pecl import pecl_loss


def test_clinical_tree():
    tree = load_taxonomy(Path(__file__).resolve().parents[1] / 'configs/taxonomy/btxrd_8class.yaml', CLASS_NAMES)
    assert len(tree.node_ids) == 12
    assert len(tree.decision_nodes) == 5
    logits = torch.randn(3, 12, requires_grad=True)
    leaf, node = hierarchical_log_probs(logits, tree)
    torch.testing.assert_close(leaf.exp().sum(-1), torch.ones(3))
    tumour = tree.node_ids.index('tumour')
    selected = [i for i, c in enumerate(CLASS_NAMES) if c != 'normal']
    torch.testing.assert_close(node[:, tumour].exp(), leaf[:, selected].exp().sum(-1))
    targets = torch.tensor([0, 2, 7])
    weight = torch.arange(1, 9).float()
    expected = -(weight[targets] * leaf[torch.arange(3), targets]).sum() / weight[targets].sum()
    torch.testing.assert_close(F.cross_entropy(leaf, targets, weight), expected)
    expected.backward()
    assert logits.grad.abs().sum() > 0


def test_flat_pecl():
    tree = flat_taxonomy(CLASS_NAMES)
    logits = torch.randn(3, 8)
    leaf, _ = hierarchical_log_probs(logits, tree)
    torch.testing.assert_close(leaf, logits.log_softmax(-1))
    u, prototypes = F.normalize(torch.randn(3, 16), dim=-1), F.normalize(torch.randn(8, 16), dim=-1)
    targets = torch.tensor([0, 1, 2])
    torch.testing.assert_close(pecl_loss(u, prototypes, targets, tree), F.cross_entropy(u @ prototypes.T / 0.07, targets))


@pytest.mark.parametrize('nodes,tokens', [(8, 49), (12, 196)])
def test_query_decoder(nodes, tokens):
    decoder = QueryDecoder(24, torch.randn(nodes, 32), dim=16, num_layers=2, num_heads=4, ffn_dim=32, dropout=0)
    fixed = decoder.node_text.clone()
    optimizer = torch.optim.Adam(decoder.parameters())
    logits, attention = decoder(torch.randn(2, tokens, 24))
    assert logits.shape == (2, nodes)
    assert attention.shape == (2, nodes, tokens)
    logits.square().sum().backward()
    for module in [decoder.query_proj, decoder.token_proj, decoder.head]:
        assert module.weight.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(fixed, decoder.node_text)
    assert 'node_text' not in dict(decoder.named_parameters())