"""Unweighted path contrastive objective on fixed node prototypes."""
import torch
from torch.nn import functional as F
from btxrd.models.hierarchy import hierarchical_log_probs


def pecl_loss(u, node_prototypes, targets, taxonomy, temperature=0.07):
    if temperature <= 0:
        raise ValueError('PECL temperature must be positive')
    with torch.autocast(device_type=u.device.type, enabled=False):
        sims = u.float() @ node_prototypes.float().T / temperature
        leaf, _ = hierarchical_log_probs(sims, taxonomy)
        return F.nll_loss(leaf, targets)