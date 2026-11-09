"""Configurable BTXRD multimodal classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from btxrd.constants import CLASS_NAMES, IMAGE_FEATURE_DIM
from btxrd.models.aggregation import build_aggregation
from btxrd.models.cross_attention import PropositionCrossAttention
from btxrd.models.fusion import ClassifierHead, MetadataMLP
from btxrd.models.image_encoder import DenseNet169Encoder


class BTXRDModel(nn.Module):
    def __init__(self, cfg, text_embeddings: torch.Tensor | None = None, class_to_ids: dict[str, list[str]] | None = None, kb_ids: list[str] | None = None, pretrained: bool | None = None):
        super().__init__()
        self.use_metadata = bool(cfg.model.use_metadata)
        self.use_knowledge = bool(cfg.model.use_knowledge)
        if pretrained is None:
            pretrained = bool(cfg.model.pretrained)
        self.image_encoder = DenseNet169Encoder(pretrained=pretrained)
        feature_dim = IMAGE_FEATURE_DIM

        self.metadata_mlp: MetadataMLP | None = None
        if self.use_metadata:
            input_dim = 1 + sum(len(cfg.metadata.categories[name]) for name in ("gender", "region", "view"))
            self.metadata_mlp = MetadataMLP(input_dim, cfg.metadata.hidden, cfg.metadata.hidden, cfg.metadata.dropout)
            feature_dim += cfg.metadata.hidden

        self.cross_attention: PropositionCrossAttention | None = None
        self.knowledge_aggregation: nn.Module | None = None
        if self.use_knowledge:
            if text_embeddings is None:
                raise ValueError("text_embeddings are required when model.use_knowledge=true")
            self.cross_attention = PropositionCrossAttention(
                text_emb=text_embeddings,
                image_dim=IMAGE_FEATURE_DIM,
                dim=cfg.knowledge.dim,
                num_heads=cfg.knowledge.num_heads,
                dropout=cfg.knowledge.attention_dropout,
                use_positional_embedding=cfg.model.use_positional_embedding,
            )
            self.knowledge_aggregation = build_aggregation(cfg.model.aggregation, cfg.knowledge.dim, cfg.knowledge.pool_hidden)
            feature_dim += cfg.knowledge.dim

        self.classifier = ClassifierHead(
            feature_dim,
            cfg.model.head.hidden,
            len(CLASS_NAMES),
            cfg.model.head.dropout,
            cfg.model.head.type,
        )

        if cfg.model.use_contrastive:
            if class_to_ids is None or kb_ids is None or text_embeddings is None:
                raise ValueError("class_to_ids and kb_ids are required for contrastive initialization")
            normalized = F.normalize(text_embeddings.float(), dim=-1)
            id_to_index = {item: index for index, item in enumerate(kb_ids)}
            prototypes = torch.stack([
                F.normalize(normalized[[id_to_index[item] for item in class_to_ids[name]]].mean(0), dim=0)
                for name in CLASS_NAMES
            ])
            self.register_buffer("prototypes", prototypes)
            self.phi = nn.Linear(cfg.knowledge.dim, cfg.knowledge.text_dim)

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None) -> dict[str, torch.Tensor | None]:
        feature_map, z_img = self.image_encoder(image)
        fusion_features = [z_img]
        result: dict[str, torch.Tensor | None] = {"z_img": z_img}
        if self.use_metadata:
            if meta is None or self.metadata_mlp is None:
                raise ValueError("Metadata tensor is required for this model")
            z_meta = self.metadata_mlp(meta)
            fusion_features.append(z_meta)
            result["z_meta"] = z_meta
        if self.use_knowledge:
            if self.cross_attention is None or self.knowledge_aggregation is None:
                raise RuntimeError("Knowledge modules are not initialized")
            prop_features, spatial_attention = self.cross_attention(feature_map)
            z_kb, aggregation_weights = self.knowledge_aggregation(prop_features)
            fusion_features.append(z_kb)
            result.update(
                prop_features=prop_features,
                cross_attn_weights=spatial_attention,
                z_kb=z_kb,
                aggregation_weights=aggregation_weights,
            )
            if hasattr(self, "phi"):
                result["u"] = F.normalize(self.phi(z_kb), dim=-1)
        result["logits"] = self.classifier(torch.cat(fusion_features, dim=-1))
        return result

    def checkpoint_buffers(self) -> dict[str, Any]:
        if self.cross_attention is None:
            return {}
        buffers: dict[str, Any] = {"text_embeddings": self.cross_attention.text_emb.detach().cpu()}
        if hasattr(self, "prototypes"):
            buffers["prototypes"] = self.prototypes.detach().cpu()
        return buffers
