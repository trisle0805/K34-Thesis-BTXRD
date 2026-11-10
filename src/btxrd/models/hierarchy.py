"""Float32 sibling softmax and path marginalization."""
import torch
from torch import nn


class HierarchicalLogProbs(nn.Module):
    def __init__(self, taxonomy):
        super().__init__()
        self.taxonomy = taxonomy
        index = {n: i for i, n in enumerate(taxonomy.node_ids)}
        for i, p in enumerate(taxonomy.decision_nodes):
            self.register_buffer(f'children_{i}', torch.tensor([index[n] for n in taxonomy.children[p]]))
        matrix = torch.zeros(len(index), len(index))
        for n, i in index.items():
            matrix[i, [index[p] for p in taxonomy.paths[n]]] = 1
        self.register_buffer('path_matrix', matrix)
        self.register_buffer('leaf_indices', torch.tensor([index[n] for n in taxonomy.leaf_of_class.values()]))

    def forward(self, node_logits):
        with torch.autocast(device_type=node_logits.device.type, enabled=False):
            logits = node_logits.float()
            conditional = torch.zeros_like(logits)
            for i in range(len(self.taxonomy.decision_nodes)):
                indices = getattr(self, f'children_{i}')
                conditional = conditional.index_copy(1, indices, logits.index_select(1, indices).log_softmax(-1))
            marginals = conditional @ self.path_matrix.T
            return marginals.index_select(1, self.leaf_indices), marginals


def hierarchical_log_probs(node_logits, taxonomy):
    return HierarchicalLogProbs(taxonomy).to(node_logits.device)(node_logits)