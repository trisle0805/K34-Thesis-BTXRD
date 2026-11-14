"""Fixed MedCPT queries with independent image and patient-text attention."""
from __future__ import annotations

import torch
from torch import nn

from btxrd.constants import CLASS_NAMES


class FixedQueryAttention(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.key = nn.Linear(input_dim, 768)
        self.value = nn.Linear(input_dim, 768)
        self.output = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.norm = nn.LayerNorm(768)

    def forward(self, queries, tokens, mask=None):
        batch, length, _ = tokens.shape
        key = self.key(tokens).reshape(batch, length, 8, 96).transpose(1, 2)
        value = self.value(tokens).reshape(batch, length, 8, 96).transpose(1, 2)
        query = queries.reshape(65, 8, 96).permute(1, 0, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / (96 ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(~mask[:, None, None, :], float('-inf'))
        weights = scores.softmax(-1)
        context = torch.matmul(weights, value).transpose(1, 2).reshape(batch, 65, 768)
        return self.norm(queries + self.dropout(self.output(context))), weights


class PropositionGuided(nn.Module):
    def __init__(self, queries, class_to_ids, prop_ids, *, use_text=True, pooling='learned'):
        super().__init__()
        if queries is None or queries.shape != (65, 768) or not torch.isfinite(queries).all():
            raise ValueError('PG requires finite 65x768 cached CLS queries')
        if prop_ids is None or len(prop_ids) != 65 or len(set(prop_ids)) != 65:
            raise ValueError('PG requires 65 unique proposition IDs')
        if pooling not in {'learned', 'mean'}:
            raise ValueError('Invalid PG pooling')
        membership = torch.zeros(8, 65, dtype=torch.bool)
        for c, name in enumerate(CLASS_NAMES):
            ids = class_to_ids[name]
            if not ids or len(set(ids)) != len(ids):
                raise ValueError('Class membership must be nonempty and unique')
            for prop_id in ids:
                membership[c, prop_ids.index(prop_id)] = True
        self.register_buffer('Q', queries.detach().float().clone())
        self.register_buffer('A', membership)
        self.image_attention = FixedQueryAttention(1664)
        self.text_attention = FixedQueryAttention(768) if use_text else None
        dim = 1536 if use_text else 768
        self.scorer = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Dropout(.1), nn.Linear(256, 1))
        self.pool = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1)) if pooling == 'learned' else None
        self.class_bias = nn.Parameter(torch.zeros(8))

    def forward(self, image_tokens, text_tokens=None, text_mask=None):
        image, attn_img = self.image_attention(self.Q, image_tokens)
        attn_txt = None
        features = image
        if self.text_attention is not None:
            validate_text(text_tokens, text_mask)
            text, attn_txt = self.text_attention(self.Q, text_tokens.float(), text_mask)
            features = torch.cat((image, text), -1)
        scores = self.scorer(features).squeeze(-1)
        pool_logits = self.pool(features).squeeze(-1) if self.pool is not None else torch.zeros_like(scores)
        alpha = pool_logits[:, None, :].expand(-1, 8, -1).masked_fill(~self.A, float('-inf')).softmax(-1)
        return dict(logits=(alpha * scores[:, None, :]).sum(-1) + self.class_bias,
                    proposition_scores=scores, class_pool_weights=alpha, attn_img=attn_img, attn_txt=attn_txt)


def validate_text(tokens, mask):
    if tokens is None or mask is None or tokens.ndim != 3 or tokens.shape[-1] != 768:
        raise ValueError('Patient tokens must have shape BxLx768')
    if mask.dtype != torch.bool or mask.shape != tokens.shape[:2] or not mask.any(-1).all():
        raise ValueError('Patient mask must be boolean, aligned and nonempty')
    if not torch.isfinite(tokens).all():
        raise ValueError('Patient tokens must be finite')


class ImageTextConcat(nn.Module):
    def __init__(self):
        super().__init__()
        self.text = nn.Sequential(nn.Linear(768, 256), nn.LayerNorm(256))
        self.classifier = nn.Linear(1920, 8)

    def forward(self, image, tokens, mask):
        validate_text(tokens, mask)
        pooled = (tokens.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        return {'logits': self.classifier(torch.cat((image, self.text(pooled)), -1))}