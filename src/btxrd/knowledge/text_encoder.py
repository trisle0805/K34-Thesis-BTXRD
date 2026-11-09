"""Offline MedCPT encoding and strict cache loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from btxrd.knowledge.kb import KnowledgeBase


def encode_texts(texts: list[str], model_name: str, max_length: int = 64, device: str = "cpu", batch_size: int = 32) -> torch.Tensor:
    """Encode short biomedical texts with CLS pooling; transformers is imported lazily."""
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError("Text encoding requires 'pip install -e .[text]'") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            encoded = tokenizer(
                texts[start : start + batch_size],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            embeddings = model(**encoded).last_hidden_state[:, 0, :].float().cpu()
            outputs.append(embeddings)
    return torch.cat(outputs, dim=0)


def save_text_embedding_cache(path: str | Path, kb: KnowledgeBase, embeddings: torch.Tensor, model_name: str, max_length: int) -> None:
    if embeddings.shape != (len(kb.all_ids), 768):
        raise ValueError(f"Expected full cache shape [{len(kb.all_ids)}, 768], got {list(embeddings.shape)}")
    payload: dict[str, Any] = {
        "ids": kb.all_ids,
        "texts": kb.all_texts,
        "embeddings": embeddings.detach().float().cpu(),
        "model_name": model_name,
        "max_length": max_length,
        "pooling": "cls",
        "kb_sha256": kb.sha256,
        "kb_version": kb.version,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_text_embeddings(cache_path: str | Path, kb: KnowledgeBase, expected_model_name: str) -> torch.Tensor:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    required = {"ids", "texts", "embeddings", "model_name", "max_length", "pooling", "kb_sha256", "kb_version"}
    missing = required - set(cache)
    if missing:
        raise ValueError(f"Embedding cache missing field(s): {sorted(missing)}")
    if cache["model_name"] != expected_model_name:
        raise ValueError(
            f"Embedding cache model_name mismatch: cache={cache['model_name']!r}, "
            f"expected={expected_model_name!r}"
        )
    if cache["kb_sha256"] != kb.sha256:
        raise ValueError("Embedding cache KB sha256 mismatch; rebuild the cache")
    if cache["kb_version"] != kb.version:
        raise ValueError("Embedding cache KB version mismatch; rebuild the cache")
    if cache["pooling"] != "cls":
        raise ValueError("Embedding cache pooling must be 'cls'")
    ids = list(cache["ids"])
    if ids != kb.all_ids:
        raise ValueError("Embedding cache IDs/order must exactly match the full KB query_ids")
    if list(cache["texts"]) != kb.all_texts:
        raise ValueError("Embedding cache texts must exactly match the full KB proposition texts")
    embeddings = cache["embeddings"]
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape != (len(ids), 768):
        raise ValueError("Embedding cache must contain FloatTensor[len(ids), 768]")
    if not torch.isfinite(embeddings).all():
        raise ValueError("Embedding cache contains non-finite values")
    index = {item: position for position, item in enumerate(ids)}
    missing_ids = [item for item in kb.ids if item not in index]
    if missing_ids:
        raise ValueError(f"Embedding cache missing KB IDs: {missing_ids[:5]}")
    return torch.stack([embeddings[index[item]] for item in kb.ids]).float()
