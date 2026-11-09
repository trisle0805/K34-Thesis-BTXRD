from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image

from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.kb import load_kb
from btxrd.knowledge.text_encoder import save_text_embedding_cache


@pytest.fixture
def synthetic_assets(tmp_path: Path) -> dict[str, Path]:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    rows: list[dict[str, str]] = []
    metadata: list[dict[str, object]] = []
    index = 0
    for split in ("train", "val", "test"):
        for class_index, class_name in enumerate(CLASS_NAMES):
            image_id = f"{split}_{class_index}"
            Image.new("L", (32, 32), color=20 + index).save(image_dir / f"{image_id}.png")
            rows.append({"image_id": image_id, "class8": class_name, "split": split})
            metadata.append({
                "image_id": image_id,
                "class8": class_name,
                "split": split,
                "age": 10 + index,
                "gender": ("M", "F")[index % 2],
                "region": ("upper limb", "lower limb", "pelvis")[index % 3],
                "view": ("frontal", "lateral", "oblique")[index % 3],
            })
            index += 1
    splits_csv = tmp_path / "splits.csv"
    metadata_csv = tmp_path / "metadata.csv"
    weights_csv = tmp_path / "weights.csv"
    pd.DataFrame(rows).to_csv(splits_csv, index=False)
    pd.DataFrame(metadata).to_csv(metadata_csv, index=False)
    pd.DataFrame({"class8": CLASS_NAMES, "weight_sqrt_inv": [1.0] * 8}).to_csv(weights_csv, index=False)

    propositions = []
    prototypes = {name: [] for name in CLASS_NAMES}
    for proposition_index in range(68):
        prop_id = f"P{proposition_index + 1:03d}"
        class_name = CLASS_NAMES[proposition_index % len(CLASS_NAMES)]
        propositions.append({"id": prop_id, "text": f"proposition {prop_id}", "classes": [class_name]})
        prototypes[class_name].append(prop_id)
    kb_json = tmp_path / "kb.json"
    kb_json.write_text(json.dumps({
        "classes": CLASS_NAMES,
        "meta": {"version": "test-1"},
        "propositions": propositions,
        "query_ids": [item["id"] for item in propositions],
        "prototypes": prototypes,
    }), encoding="utf-8")
    kb = load_kb(kb_json, [], CLASS_NAMES)
    cache = tmp_path / "text_cache.pt"
    save_text_embedding_cache(cache, kb, torch.randn(68, 768), "fake", 64)
    return {
        "root": tmp_path,
        "image_dir": image_dir,
        "splits_csv": splits_csv,
        "metadata_csv": metadata_csv,
        "weights_csv": weights_csv,
        "kb_json": kb_json,
        "cache": cache,
    }


@pytest.fixture
def config_overrides(synthetic_assets: dict[str, Path]):
    assets = synthetic_assets

    def build(*, output_dir: Path | None = None, run_name: str = "test", **extra):
        values: dict[str, object] = {
            "paths.image_dir": str(assets["image_dir"]),
            "paths.splits_csv": str(assets["splits_csv"]),
            "paths.class_weights_csv": str(assets["weights_csv"]),
            "paths.metadata_csv": str(assets["metadata_csv"]),
            "paths.kb_json": str(assets["kb_json"]),
            "paths.text_embedding_cache": str(assets["cache"]),
            "paths.output_dir": str(output_dir or assets["root"] / "runs"),
            "data.expected_split_counts": None,
            "data.batch_size": 8,
            "data.num_workers": 0,
            "data.pin_memory": False,
            "model.pretrained": False,
            "run.name": run_name,
        }
        values.update(extra)
        return values

    return build
