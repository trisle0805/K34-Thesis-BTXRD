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
from btxrd.models.image_encoder import build_image_encoder, EncoderOutput
from btxrd.models.class_decoder import ClassConditionedDecoder, build_class_prototypes


class BTXRDModel(nn.Module):
    def __init__(self, cfg, text_embeddings: torch.Tensor | None = None, class_to_ids: dict[str, list[str]] | None = None, kb_ids: list[str] | None = None, pretrained: bool | None = None):
        super().__init__()
        self.use_metadata = bool(cfg.model.use_metadata)
        self.use_knowledge = bool(cfg.model.use_knowledge)
        if pretrained is None:
            pretrained = bool(cfg.model.pretrained)
        self.image_encoder = build_image_encoder(cfg, pretrained, densenet_class=DenseNet169Encoder)
        feature_dim = getattr(self.image_encoder, 'global_dim', IMAGE_FEATURE_DIM)
        self.architecture = cfg.get_path('model.architecture', 'legacy')
        if self.architecture != 'legacy':
            from btxrd.models.proposition_guided import PropositionGuided, ImageTextConcat
            if feature_dim != 1664:
                raise ValueError('PG and image_text_concat require DenseNet169 features')
            if self.architecture == 'pg':
                self.pg = PropositionGuided(text_embeddings, class_to_ids, kb_ids,
                    use_text=cfg.get_path('model.pg_use_text', True), pooling=cfg.model.aggregation)
            elif self.architecture == 'image_text_concat':
                self.concat = ImageTextConcat()
            else:
                raise ValueError('Unknown architecture')
            return
        self.class_conditioned = cfg.get_path('model.knowledge_decoder', 'global') == 'class_conditioned'

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
                image_dim=getattr(self.image_encoder, 'token_dim', IMAGE_FEATURE_DIM),
                dim=cfg.knowledge.dim,
                num_heads=cfg.knowledge.num_heads,
                dropout=cfg.knowledge.attention_dropout,
                use_positional_embedding=cfg.model.use_positional_embedding,
            )
            if self.class_conditioned:
                self.class_decoder = ClassConditionedDecoder(cfg, text_embeddings, class_to_ids, kb_ids)
                self.knowledge_logit_scale = nn.Parameter(torch.tensor(float(cfg.knowledge.decoder.beta_init)))
            else:
                self.knowledge_aggregation = build_aggregation(cfg.model.aggregation, cfg.knowledge.dim, cfg.knowledge.pool_hidden)
                feature_dim += cfg.knowledge.dim

        self.classifier = ClassifierHead(
            feature_dim,
            cfg.model.head.hidden,
            len(CLASS_NAMES),
            cfg.model.head.dropout,
            cfg.model.head.type,
        )

        if cfg.get_path('model.use_pecl', False):
            if text_embeddings is None or class_to_ids is None or kb_ids is None:
                raise ValueError('PECL requires embeddings and class mappings')
            self.register_buffer('pecl_prototypes', build_class_prototypes(text_embeddings, class_to_ids, kb_ids))
            self.pecl_phi = nn.Linear(self.image_encoder.global_dim, cfg.knowledge.text_dim)
            self.register_buffer('pecl_text_embeddings', text_embeddings.detach().clone())

        if cfg.model.use_contrastive:
            if class_to_ids is None or kb_ids is None or text_embeddings is None:
                raise ValueError("class_to_ids and kb_ids are required for contrastive initialization")
            prototypes = build_class_prototypes(text_embeddings, class_to_ids, kb_ids)
            self.register_buffer("prototypes", prototypes)
            self.phi = nn.Linear(cfg.knowledge.dim, cfg.knowledge.text_dim)

    def forward(self, image: torch.Tensor, meta: torch.Tensor | None = None, text_tokens=None, text_mask=None) -> dict[str, torch.Tensor | None]:
        encoded = self.image_encoder(image)
        if isinstance(encoded, EncoderOutput):
            feature_map, z_img = encoded.visual_tokens, encoded.global_feature
            spatial_shape = encoded.spatial_shape
        else:
            feature_map, z_img = encoded
            spatial_shape = None
        if self.architecture == 'pg':
            if feature_map.ndim == 4:
                feature_map = feature_map.flatten(2).transpose(1, 2)
            return self.pg(feature_map, text_tokens, text_mask)
        if self.architecture == 'image_text_concat':
            return self.concat(z_img, text_tokens, text_mask)
        fusion_features = [z_img]
        result: dict[str, torch.Tensor | None] = {"z_img": z_img}
        if hasattr(self, 'pecl_phi'):
            result['pecl_u'] = F.normalize(self.pecl_phi(z_img), dim=-1)
        if self.use_metadata:
            if meta is None or self.metadata_mlp is None:
                raise ValueError("Metadata tensor is required for this model")
            z_meta = self.metadata_mlp(meta)
            fusion_features.append(z_meta)
            result["z_meta"] = z_meta
        if self.use_knowledge:
            if self.cross_attention is None:
                raise RuntimeError("Knowledge modules are not initialized")
            prop_features, spatial_attention = self.cross_attention(feature_map, spatial_shape)
            if self.class_conditioned:
                logits_kb, class_weights = self.class_decoder(prop_features)
                logits_img = self.classifier(z_img)
                result.update(logits=logits_img + self.knowledge_logit_scale * logits_kb,
                              logits_img=logits_img, logits_kb=logits_kb,
                              knowledge_logit_scale=self.knowledge_logit_scale,
                              prop_features=prop_features, cross_attn_weights=spatial_attention,
                              class_attn_weights=class_weights)
                return result
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
        if self.architecture == 'pg':
            return {'text_embeddings': self.pg.Q.detach().cpu(), 'Q': self.pg.Q.detach().cpu(), 'A': self.pg.A.detach().cpu()}
        if self.architecture == 'image_text_concat':
            return {}
        if self.cross_attention is None:
            return {'text_embeddings': self.pecl_text_embeddings.detach().cpu()} if hasattr(self, 'pecl_text_embeddings') else {}
        buffers: dict[str, Any] = {"text_embeddings": self.cross_attention.text_emb.detach().cpu()}
        if hasattr(self, "prototypes"):
            buffers["prototypes"] = self.prototypes.detach().cpu()
        return buffers

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.architecture == 'pg':
            for name in ('Q', 'A'):
                key = f'pg.{name}'
                if key not in state_dict or not torch.equal(state_dict[key].cpu(), getattr(self.pg, name).cpu()):
                    raise ValueError(f'Checkpoint PG {name} mismatch')
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
