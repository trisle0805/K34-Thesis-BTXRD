from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision
import yaml
from torch import nn
from torch.nn import functional as F

from btxrd.config import Config, load_config
from btxrd.constants import CLASS_NAMES
from btxrd.data.metadata import MetadataPreprocessor
from btxrd.knowledge.kb import load_kb
from btxrd.knowledge.text_encoder import load_text_embeddings, save_text_embedding_cache
from btxrd.losses.contrastive import PrototypeContrastiveLoss
from btxrd.models.aggregation import LearnedAttentionPooling, MeanPooling
from btxrd.models.cross_attention import PropositionCrossAttention
from btxrd.models.image_encoder import DenseNet169Encoder
from btxrd.models.model import BTXRDModel
from btxrd.training.metrics import compute_metrics
from btxrd.utils.io import find_repo_root, git_commit


def make_kb(path: Path):
    propositions = []
    prototypes = {name: [] for name in CLASS_NAMES}
    for index in range(1, 69):
        name = CLASS_NAMES[(index - 1) % len(CLASS_NAMES)]
        prop_id = f"P{index:03d}"
        propositions.append({"id": prop_id, "text": f"text {index}", "classes": [name]})
        prototypes[name].append(prop_id)
    payload = {
        "classes": CLASS_NAMES,
        "meta": {"version": "test-1"},
        "propositions": propositions,
        "query_ids": [item["id"] for item in propositions],
        "prototypes": prototypes,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def model_config(use_knowledge=False, use_metadata=False, use_contrastive=False, aggregation="learned"):
    raw = yaml.safe_load(Path("configs/base.yaml").read_text(encoding="utf-8"))
    raw["model"].update(
        pretrained=False,
        use_knowledge=use_knowledge,
        use_metadata=use_metadata,
        use_contrastive=use_contrastive,
        aggregation=aggregation,
    )
    raw["model"]["head"]["type"] = "mlp" if use_knowledge else "linear"
    return Config(raw)

def knowledge_inputs():
    ids = [f"P{index:03d}" for index in range(1, 9)]
    class_to_ids = {name: [ids[index]] for index, name in enumerate(CLASS_NAMES)}
    return torch.randn(8, 768), class_to_ids, ids

class TinyGradientEncoder(nn.Module):
    def forward(self, image: torch.Tensor):
        spatial = F.adaptive_avg_pool2d(image.mean(dim=1, keepdim=True), (7, 7))
        feature_map = spatial.expand(-1, 1664, -1, -1)
        return feature_map, feature_map.mean(dim=(2, 3))


def test_config_merge_override_and_unknown():
    required = [
        "paths.image_dir=x", "paths.splits_csv=x", "paths.class_weights_csv=x", "paths.output_dir=x",
        "train.epochs=2",
    ]
    cfg = load_config("configs/b0_image.yaml", required)
    assert cfg.train.epochs == 2
    with pytest.raises(KeyError):
        load_config("configs/b0_image.yaml", required + ["train.typo=1"])

def test_git_commit_is_discovered_from_package_file():
    repo_root = find_repo_root(Path(__file__).resolve())
    assert repo_root == Path(__file__).resolve().parents[1]
    assert git_commit(Path(__file__).resolve())


def test_prediction_config_can_omit_knowledge_assets():
    overrides = [
        "paths.image_dir=x", "paths.splits_csv=x", "paths.class_weights_csv=x", "paths.output_dir=x",
    ]
    with pytest.raises(ValueError, match="kb_json"):
        load_config("configs/b2_knowledge.yaml", overrides)
    cfg = load_config("configs/b2_knowledge.yaml", overrides, require_training_assets=False)
    assert cfg.paths.kb_json is None
    assert cfg.paths.text_embedding_cache is None


def test_locked_config_validation_rules():
    base = [
        "paths.image_dir=x", "paths.splits_csv=x", "paths.class_weights_csv=x", "paths.output_dir=x",
        "paths.kb_json=x", "paths.text_embedding_cache=x",
    ]
    with pytest.raises(ValueError, match="contrastive_weight"):
        load_config("configs/b3_contrastive.yaml", base + ["loss.contrastive_weight=0"])
    with pytest.raises(ValueError, match="text_encoder_frozen"):
        load_config("configs/b2_knowledge.yaml", base + ["knowledge.text_encoder_frozen=false"])
    with pytest.raises(ValueError, match="divisible"):
        load_config("configs/b2_knowledge.yaml", base + ["knowledge.dim=255"])


def test_metadata_preprocessor_train_stats_and_order():
    cfg = model_config(use_metadata=True)
    frame = pd.DataFrame({
        "image_id": ["a", "b"], "age": [10, 20], "gender": ["M", "F"],
        "region": ["upper limb", "pelvis"], "view": ["frontal", "oblique"],
    })
    preprocessor = MetadataPreprocessor.from_config(cfg).fit(frame)
    encoded = preprocessor.transform(frame)
    assert encoded.shape == (2, 9)
    assert np.allclose(encoded[:, 0], [-1, 1])
    assert encoded[0].tolist() == [-1, 1, 0, 1, 0, 0, 1, 0, 0]


def test_kb_and_embedding_cache_strict_order(tmp_path):
    kb_path = tmp_path / "kb.json"
    make_kb(kb_path)
    kb = load_kb(kb_path, [], CLASS_NAMES)
    embeddings = torch.randn(68, 768)
    cache_path = tmp_path / "cache.pt"
    save_text_embedding_cache(cache_path, kb, embeddings, "fake", 64)
    assert torch.equal(load_text_embeddings(cache_path, kb), embeddings)
    cache = torch.load(cache_path, weights_only=False)
    cache["ids"] = list(reversed(cache["ids"]))
    torch.save(cache, cache_path)
    with pytest.raises(ValueError, match="IDs/order"):
        load_text_embeddings(cache_path, kb)


def test_kb_requires_exactly_68_propositions(tmp_path):
    kb_path = tmp_path / "kb.json"
    payload = make_kb(kb_path)
    removed = payload["propositions"].pop()
    payload["query_ids"].remove(removed["id"])
    payload["prototypes"][removed["classes"][0]].remove(removed["id"])
    kb_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 68"):
        load_kb(kb_path, [], CLASS_NAMES)


def test_real_kb_contract_when_available():
    path = os.getenv("BTXRD_KB_JSON")
    if not path:
        pytest.skip("Set BTXRD_KB_JSON to validate the real knowledge base")
    kb = load_kb(path, ["P066", "P067", "P068"], CLASS_NAMES)
    assert kb.all_ids == [f"P{index:03d}" for index in range(1, 69)]
    assert kb.ids == [f"P{index:03d}" for index in range(1, 66)]
    expected = {
        "normal": 12, "osteochondroma": 8, "multiple_osteochondromas": 10,
        "simple_bone_cyst": 13, "giant_cell_tumor": 12, "osteofibroma": 7,
        "synovial_osteochondroma": 9, "osteosarcoma": 16,
    }
    assert {name: len(ids) for name, ids in kb.class_to_ids.items()} == expected


def test_attention_and_aggregation_shapes():
    module = PropositionCrossAttention(torch.randn(5, 768), 1664, 32, 4, 0.0)
    features, weights = module(torch.randn(2, 1664, 7, 7))
    assert features.shape == (2, 5, 32)
    assert weights.shape == (2, 5, 49)
    pooled, alpha = LearnedAttentionPooling(32, 16)(features)
    assert pooled.shape == (2, 32)
    assert torch.allclose(alpha.sum(1), torch.ones(2))
    mean, no_alpha = MeanPooling()(features)
    assert mean.shape == (2, 32) and no_alpha is None


def test_image_encoder_matches_torchvision_pooling():
    encoder = DenseNet169Encoder(pretrained=False).eval()
    ref = torchvision.models.densenet169(weights=None)
    ref.classifier = nn.Identity()
    ref.features.load_state_dict(encoder.features.state_dict())
    ref.eval()
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feature_map, pooled = encoder(image)
        expected = ref(image)
    assert feature_map.shape == (1, 1664, 7, 7)
    assert torch.allclose(pooled, expected, atol=1e-5)

def test_knowledge_buffers_are_frozen_and_excluded_from_optimizer(monkeypatch):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", lambda pretrained=False: TinyGradientEncoder())
    cfg = model_config(use_knowledge=True, use_contrastive=True)
    embeddings, class_to_ids, ids = knowledge_inputs()
    model = BTXRDModel(cfg, embeddings, class_to_ids, ids, pretrained=False)
    parameter_names = dict(model.named_parameters())
    buffer_names = dict(model.named_buffers())
    assert "cross_attention.text_emb" in buffer_names
    assert "prototypes" in buffer_names
    assert "cross_attention.text_emb" not in parameter_names
    assert "prototypes" not in parameter_names
    assert not model.cross_attention.text_emb.requires_grad
    assert not model.prototypes.requires_grad
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert id(model.cross_attention.text_emb) not in optimizer_ids
    assert id(model.prototypes) not in optimizer_ids

def test_b2_b3_common_state_is_identical_with_same_seed():
    embeddings, class_to_ids, ids = knowledge_inputs()
    torch.manual_seed(2026)
    b2 = BTXRDModel(model_config(use_knowledge=True), embeddings, class_to_ids, ids, pretrained=False)
    torch.manual_seed(2026)
    b3 = BTXRDModel(
        model_config(use_knowledge=True, use_contrastive=True),
        embeddings,
        class_to_ids,
        ids,
        pretrained=False,
    )
    assert not hasattr(b2, "phi")
    assert hasattr(b3, "phi")
    b2_state = b2.state_dict()
    b3_state = b3.state_dict()
    common_keys = b2_state.keys() & b3_state.keys()
    assert common_keys
    assert all(torch.equal(b2_state[key], b3_state[key]) for key in common_keys)

def test_b3_backward_reaches_all_knowledge_branches_without_mutating_buffers(monkeypatch):
    monkeypatch.setattr("btxrd.models.model.DenseNet169Encoder", lambda pretrained=False: TinyGradientEncoder())
    torch.manual_seed(2026)
    cfg = model_config(use_knowledge=True, use_contrastive=True)
    embeddings, class_to_ids, ids = knowledge_inputs()
    model = BTXRDModel(cfg, embeddings, class_to_ids, ids, pretrained=False).eval()
    text_before = model.cross_attention.text_emb.clone()
    prototypes_before = model.prototypes.clone()
    output = model(torch.randn(4, 3, 32, 32))
    labels = torch.tensor([0, 1, 2, 3])
    loss = nn.CrossEntropyLoss()(output["logits"], labels)
    loss = loss + 0.1 * PrototypeContrastiveLoss()(output["u"], model.prototypes, labels, 0.07)
    loss.backward()

    gradients = {
        "query_proj": model.cross_attention.query_proj.weight.grad,
        "attn.in_proj_weight": model.cross_attention.attn.in_proj_weight.grad,
        "token_proj": model.cross_attention.token_proj.weight.grad,
        "pooling.W_h": model.knowledge_aggregation.W_h.weight.grad,
        "phi": model.phi.weight.grad,
        "head": model.classifier.network[-1].weight.grad,
    }
    assert all(gradient is not None and torch.count_nonzero(gradient) > 0 for gradient in gradients.values())
    assert torch.equal(model.cross_attention.text_emb, text_before)
    assert torch.equal(model.prototypes, prototypes_before)


def test_model_variants_output_contract():
    image = torch.randn(1, 3, 224, 224)
    metadata_cfg = model_config(use_metadata=True)
    metadata_model = BTXRDModel(metadata_cfg, pretrained=False).eval()
    with torch.no_grad():
        output = metadata_model(image, torch.randn(1, 9))
    assert output["logits"].shape == (1, 8)
    with pytest.raises(ValueError):
        metadata_model(image)

    knowledge_cfg = model_config(use_knowledge=True, use_contrastive=True)
    embeddings, class_to_ids, ids = knowledge_inputs()
    knowledge_model = BTXRDModel(knowledge_cfg, embeddings, class_to_ids, ids, pretrained=False).eval()
    with torch.no_grad():
        output = knowledge_model(image)
    assert output["prop_features"].shape == (1, 8, 256)
    assert output["cross_attn_weights"].shape == (1, 8, 49)
    assert output["aggregation_weights"].shape == (1, 8)
    assert output["u"].shape == (1, 768)
    assert torch.allclose(knowledge_model.prototypes.norm(dim=1), torch.ones(8), atol=1e-6)


def test_contrastive_zero_weight_equivalence_and_metrics():
    loss_fn = PrototypeContrastiveLoss()
    u = torch.nn.functional.normalize(torch.randn(4, 768), dim=-1)
    prototypes = torch.nn.functional.normalize(torch.randn(8, 768), dim=-1)
    labels = torch.tensor([0, 1, 2, 3])
    ce = torch.tensor(1.25)
    assert torch.equal(ce + 0 * loss_fn(u, prototypes, labels, 0.07), ce)
    probabilities = np.full((3, 8), 1 / 8)
    metrics = compute_metrics([0, 0, 1], [0, 1, 1], probabilities)
    assert metrics["n_auroc_classes"] == 2
    assert np.isnan(metrics["osteosarcoma_auroc"])
