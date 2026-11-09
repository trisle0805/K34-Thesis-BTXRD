"""Strict BTXRD knowledge-base loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from btxrd.constants import EXPECTED_KB_PROPOSITIONS
from btxrd.utils.io import sha256_file


@dataclass(frozen=True)
class KnowledgeBase:
    ids: list[str]
    texts: list[str]
    all_texts: list[str]
    class_to_ids: dict[str, list[str]]
    all_ids: list[str]
    version: str
    sha256: str


def load_kb(path: str | Path, exclude_ids: list[str], class_names: list[str]) -> KnowledgeBase:
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {"classes", "meta", "propositions", "query_ids", "prototypes"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"KB missing field(s): {sorted(missing)}")
    if raw["classes"] != class_names:
        raise ValueError("KB classes must exactly match CLASS_NAMES in order")

    propositions = raw["propositions"]
    if len(propositions) != EXPECTED_KB_PROPOSITIONS:
        raise ValueError(f"KB must contain exactly {EXPECTED_KB_PROPOSITIONS} propositions")
    proposition_ids = [item.get("id") for item in propositions]
    if len(proposition_ids) != len(set(proposition_ids)) or any(not item for item in proposition_ids):
        raise ValueError("KB proposition IDs must be non-empty and unique")
    query_ids = list(raw["query_ids"])
    if len(query_ids) != len(set(query_ids)) or set(query_ids) != set(proposition_ids):
        raise ValueError("KB query_ids must contain each proposition ID exactly once")
    by_id = {item["id"]: item for item in propositions}

    recomputed: dict[str, list[str]] = {name: [] for name in class_names}
    for proposition_id in query_ids:
        item = by_id[proposition_id]
        if not isinstance(item.get("text"), str) or not item["text"].strip():
            raise ValueError(f"KB proposition {proposition_id} has invalid text")
        mapped_classes = item.get("classes")
        if not isinstance(mapped_classes, list) or len(mapped_classes) != len(set(mapped_classes)):
            raise ValueError(f"KB proposition {proposition_id} has invalid classes")
        unknown = set(mapped_classes) - set(class_names)
        if unknown:
            raise ValueError(f"KB proposition {proposition_id} has unknown classes: {sorted(unknown)}")
        for class_name in mapped_classes:
            recomputed[class_name].append(proposition_id)

    prototypes = raw["prototypes"]
    if set(prototypes) != set(class_names):
        raise ValueError("KB prototypes must define exactly CLASS_NAMES")
    for class_name in class_names:
        if list(prototypes[class_name]) != recomputed[class_name]:
            raise ValueError(f"KB prototypes mismatch recomputed mapping for {class_name}")

    excluded = set(exclude_ids)
    unknown_excluded = excluded - set(proposition_ids)
    if unknown_excluded:
        raise ValueError(f"knowledge.exclude_ids contains unknown IDs: {sorted(unknown_excluded)}")
    ids = [item for item in query_ids if item not in excluded]
    class_to_ids = {name: [item for item in recomputed[name] if item not in excluded] for name in class_names}
    empty = [name for name, values in class_to_ids.items() if not values]
    if empty:
        raise ValueError(f"Classes have empty prototypes after exclusion: {empty}")

    return KnowledgeBase(
        ids=ids,
        texts=[by_id[item]["text"] for item in ids],
        all_texts=[by_id[item]["text"] for item in query_ids],
        class_to_ids=class_to_ids,
        all_ids=query_ids,
        version=str(raw["meta"].get("version", "")),
        sha256=sha256_file(path),
    )
