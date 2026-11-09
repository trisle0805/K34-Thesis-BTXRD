"""Checkpoint-driven prediction isolated from training and test leakage."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from btxrd.config import Config
from btxrd.constants import CLASS_NAMES
from btxrd.data.dataset import build_dataloaders
from btxrd.knowledge.kb import load_kb
from btxrd.knowledge.text_encoder import load_text_embeddings
from btxrd.models.model import BTXRDModel
from btxrd.training.checkpoint import load_checkpoint


_ARCHITECTURE_KEYS = (
    "model",
    "metadata",
    "knowledge",
    "data.columns",
    "data.image_ext",
    "data.input_size",
)
_RUNTIME_KEYS = (
    "paths",
    "data.batch_size",
    "data.num_workers",
    "data.pin_memory",
    "data.expected_split_counts",
    "debug",
)


def _get_dotted(data: dict[str, Any], key: str) -> Any:
    value: Any = data
    for part in key.split("."):
        value = value[part]
    return value


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _prediction_config(runtime_cfg, checkpoint_config: dict[str, Any]) -> Config:
    runtime = runtime_cfg.to_dict()
    mismatches = [key for key in _ARCHITECTURE_KEYS if _get_dotted(runtime, key) != _get_dotted(checkpoint_config, key)]
    if mismatches:
        raise ValueError(f"Prediction architecture differs from checkpoint at: {', '.join(mismatches)}")
    resolved = copy.deepcopy(checkpoint_config)
    for key in _RUNTIME_KEYS:
        _set_dotted(resolved, key, copy.deepcopy(_get_dotted(runtime, key)))
    return Config(resolved, runtime_cfg.source_path)


def _verify_optional_knowledge(cfg, checkpoint: dict[str, Any]) -> None:
    kb_path = cfg.paths.kb_json
    cache_path = cfg.paths.text_embedding_cache
    if kb_path:
        kb = load_kb(kb_path, cfg.knowledge.exclude_ids, CLASS_NAMES)
        if kb.ids != checkpoint.get("kb_ids") or kb.sha256 != checkpoint.get("kb_sha256"):
            raise ValueError("Provided KB does not match checkpoint kb_ids/sha256")
        if cache_path:
            cached = load_text_embeddings(cache_path, kb)
            expected = checkpoint["model_buffers"]["text_embeddings"]
            if cached.shape != expected.shape or not torch.equal(cached, expected):
                raise ValueError("Provided text embedding cache does not match checkpoint embeddings")
    elif cache_path:
        raise ValueError("paths.text_embedding_cache cannot be verified without paths.kb_json")


def predict(cfg, checkpoint_path: str | Path, split: str = "test", return_extras: bool = False):
    checkpoint = load_checkpoint(checkpoint_path, "cpu")
    resolved = _prediction_config(cfg, checkpoint["config"])
    if resolved.model.use_knowledge:
        _verify_optional_knowledge(resolved, checkpoint)
        embeddings = checkpoint.get("model_buffers", {}).get("text_embeddings")
        if embeddings is None:
            raise ValueError("Knowledge checkpoint is missing embedded text_embeddings")
    else:
        embeddings = None
    data = build_dataloaders(resolved, (split,), checkpoint.get("metadata_stats"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BTXRDModel(
        resolved,
        embeddings,
        checkpoint.get("class_to_ids"),
        checkpoint.get("kb_ids"),
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_ids: list[str] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    probabilities: list[np.ndarray] = []
    cross_attn: list[np.ndarray] = []
    aggregation: list[np.ndarray] = []
    with torch.no_grad():
        for batch in data.loaders[split]:
            image = batch["image"].to(device, non_blocking=True)
            metadata = batch.get("meta")
            if metadata is not None:
                metadata = metadata.to(device, non_blocking=True)
            output = model(image, metadata)
            prob = output["logits"].softmax(dim=1)
            prediction = prob.argmax(dim=1)
            image_ids.extend(batch["image_id"])
            y_true.extend(batch["label"].tolist())
            y_pred.extend(prediction.cpu().tolist())
            probabilities.append(prob.cpu().numpy())
            if "cross_attn_weights" in output:
                cross_attn.append(output["cross_attn_weights"].cpu().numpy())
            if output.get("aggregation_weights") is not None:
                aggregation.append(output["aggregation_weights"].cpu().numpy())

    probability_array = np.concatenate(probabilities, axis=0)
    frame = pd.DataFrame({
        "image_id": image_ids,
        "y_true": y_true,
        "y_pred": y_pred,
        "true_class": [CLASS_NAMES[index] for index in y_true],
        "pred_class": [CLASS_NAMES[index] for index in y_pred],
    })
    for index, name in enumerate(CLASS_NAMES):
        frame[f"p_{name}"] = probability_array[:, index]
    if not return_extras:
        return frame
    extras = {
        "aggregation_weights": np.concatenate(aggregation) if aggregation else None,
        "cross_attn_weights": np.concatenate(cross_attn) if cross_attn else None,
        "kb_ids": checkpoint.get("kb_ids") if resolved.model.use_knowledge else None,
    }
    return frame, extras
