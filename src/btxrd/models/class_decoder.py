"""Class-conditioned masked proposition decoding."""
import torch
from torch import nn
from torch.nn import functional as F
from btxrd.constants import CLASS_NAMES


def build_class_prototypes(text_embeddings, class_to_ids, kb_ids):
    normalized = F.normalize(text_embeddings.float(), dim=-1)
    indices = {name: index for index, name in enumerate(kb_ids)}
    return torch.stack([
        F.normalize(normalized[[indices[item] for item in class_to_ids[name]]].mean(0), dim=0)
        for name in CLASS_NAMES
    ])


class ClassConditionedDecoder(nn.Module):
    def __init__(self, cfg, embeddings, class_to_ids, kb_ids):
        super().__init__()
        self.register_buffer('class_prototypes', build_class_prototypes(embeddings, class_to_ids, kb_ids))
        self.register_buffer('attention_mask', torch.tensor([
            [item not in class_to_ids[name] for item in kb_ids] for name in CLASS_NAMES
        ], dtype=torch.bool))
        dim = cfg.knowledge.dim
        self.query_proj = nn.Linear(embeddings.shape[-1], dim)
        self.attn = nn.MultiheadAttention(dim, cfg.knowledge.decoder.num_heads,
                                        dropout=cfg.knowledge.attention_dropout, batch_first=True)
        self.scorer_type = cfg.knowledge.decoder.scorer
        if self.scorer_type == 'shared':
            self.scorer = nn.Linear(dim, 1)
        else:
            self.weight = nn.Parameter(torch.empty(len(CLASS_NAMES), dim))
            self.bias = nn.Parameter(torch.zeros(len(CLASS_NAMES)))
            nn.init.xavier_uniform_(self.weight)

    def forward(self, propositions):
        query = self.query_proj(self.class_prototypes).unsqueeze(0).expand(propositions.shape[0], -1, -1)
        decoded, weights = self.attn(query, propositions, propositions, attn_mask=self.attention_mask)
        logits = (self.scorer(decoded).squeeze(-1) if self.scorer_type == 'shared'
                  else (decoded * self.weight).sum(-1) + self.bias)
        return logits, weights