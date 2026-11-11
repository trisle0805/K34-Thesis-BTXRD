"""YAML configuration loading, inheritance, overrides, and validation."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class Config(Mapping[str, Any]):
    """Read-only-at-the-root mapping with attribute access for nested values."""

    def __init__(self, data: dict[str, Any], source_path: str | Path | None = None):
        self._data = data
        self.source_path = Path(source_path).resolve() if source_path else None

    def __getitem__(self, key: str) -> Any:
        value = self._data[key]
        return Config(value, self.source_path) if isinstance(value, dict) else value

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def get_path(self, dotted_key: str, default: Any = None) -> Any:
        current: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in child.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def migrate_config(data: dict[str, Any]) -> dict[str, Any]:
    """Supply architecture/protocol defaults absent from pre-R1 checkpoints."""
    result = deepcopy(data)
    model = result['model']
    model.setdefault('knowledge_decoder', 'global')
    model.setdefault('residual_knowledge_logits', False)
    model.setdefault('skelex', {'model_name': 'skhoha/SKELEX', 'vit_config': None,
        'test_config': {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 2,
                        'intermediate_size': 64, 'patch_size': 16, 'image_size': 224}})
    result['knowledge'].setdefault('decoder', {'num_heads': 1, 'scorer': 'shared', 'beta_init': 0.1})
    result['knowledge'].setdefault('embedding_permutation_seed', None)
    result['loss'].setdefault('aux_image_weight', 0.0)
    result['loss'].setdefault('aux_knowledge_weight', 0.0)
    result['data'].setdefault('preprocessing', 'imagenet_gray3')
    recipes = _read_with_bases(Path(__file__).resolve().parents[2] / 'configs' / 'base.yaml')['recipes']
    recipe_name = recipes['backbones'][model['backbone']]
    defaults = {'recipe': recipe_name, **recipes[recipe_name]['train']}
    for key, value in defaults.items():
        result['train'].setdefault(key, value)
    return result


def _read_with_bases(path: Path, chain: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in chain:
        cycle = " -> ".join(str(item) for item in (*chain, path))
        raise ValueError(f"Config base cycle detected: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    base_ref = loaded.pop("base", None)
    if base_ref is None:
        return loaded
    if not isinstance(base_ref, str):
        raise ValueError(f"Config 'base' must be a path string: {path}")
    base = _read_with_bases((path.parent / base_ref).resolve(), (*chain, path))
    return _deep_merge(base, loaded)


def _flatten_keys(data: dict[str, Any], prefix: str = "") -> set[str]:
    keys: set[str] = set()
    for key, value in data.items():
        dotted = f"{prefix}.{key}" if prefix else key
        keys.add(dotted)
        if isinstance(value, dict):
            keys.update(_flatten_keys(value, dotted))
    return keys


def _set_dotted(data: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current: MutableMapping[str, Any] = data
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, MutableMapping):
            raise KeyError(f"Cannot override unknown config key: {dotted_key}")
        current = child
    if parts[-1] not in current:
        raise KeyError(f"Unknown config key: {dotted_key}")
    current[parts[-1]] = value


def _apply_overrides(data: dict[str, Any], overrides: dict[str, Any] | list[str] | None) -> None:
    if overrides is None:
        return
    if isinstance(overrides, dict):
        items = overrides.items()
    elif isinstance(overrides, list):
        parsed: list[tuple[str, Any]] = []
        for item in overrides:
            if "=" not in item:
                raise ValueError(f"Override must have key=value form: {item!r}")
            key, raw = item.split("=", 1)
            parsed.append((key, yaml.safe_load(raw)))
        items = parsed
    else:
        raise TypeError("overrides must be a dict, list[str], or None")
    for key, value in items:
        _set_dotted(data, str(key), value)


def _require_positive(name: str, value: Any, allow_zero: bool = False) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    if value < 0 if allow_zero else value <= 0:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {comparator}")


def validate_config(data: dict[str, Any], schema: dict[str, Any] | None = None, *, require_training_assets: bool = True) -> None:
    """Validate schema and all locked experiment invariants."""
    if schema is not None:
        unknown = sorted(_flatten_keys(data) - _flatten_keys(schema))
        if unknown:
            raise KeyError(f"Unknown config key(s): {', '.join(unknown)}")

    model = data["model"]
    knowledge = data["knowledge"]
    loss = data["loss"]
    for key in ('aux_image_weight', 'aux_knowledge_weight'):
        _require_positive(f'loss.{key}', loss[key], allow_zero=True)
        if loss[key] > 0 and not (data['model']['knowledge_decoder'] == 'class_conditioned'
                                  and data['model']['residual_knowledge_logits']):
            raise ValueError('Positive auxiliary weights require class_conditioned residual_knowledge_logits')
    permutation_seed = knowledge['embedding_permutation_seed']
    if permutation_seed is not None:
        if isinstance(permutation_seed, bool) or not isinstance(permutation_seed, int) or permutation_seed < 0:
            raise ValueError('knowledge.embedding_permutation_seed must be null or a non-negative integer')
        if not model['use_knowledge']:
            raise ValueError('Embedding permutation requires knowledge')
    train_cfg = data["train"]
    data_cfg = data["data"]
    metadata = data["metadata"]

    if model['backbone'] not in {'densenet169', 'skelex'}:
        raise ValueError('Unsupported model.backbone')
    decoder = model.get('knowledge_decoder', 'global')
    if decoder not in {'global', 'class_conditioned'}:
        raise ValueError('Invalid model.knowledge_decoder')
    if (decoder == 'class_conditioned') != model.get('residual_knowledge_logits', False):
        raise ValueError('class_conditioned requires residual_knowledge_logits and vice versa')
    if decoder == 'class_conditioned' and (not model['use_knowledge'] or model['use_metadata'] or model['use_contrastive']):
        raise ValueError('class_conditioned requires knowledge without metadata or contrastive')
    if knowledge.get('decoder', {}).get('scorer', 'shared') not in {'shared', 'per_class'}:
        raise ValueError('Invalid knowledge.decoder.scorer')
    if knowledge['decoder']['num_heads'] < 1 or knowledge['dim'] % knowledge['decoder']['num_heads']:
        raise ValueError('knowledge.dim must be divisible by decoder.num_heads')
    if data_cfg['preprocessing'] not in {'imagenet_gray3', 'skelex_minmax_rgb'}:
        raise ValueError('Invalid data.preprocessing')
    if train_cfg['recipe'] not in {'densenet', 'skelex'}:
        raise ValueError('Invalid train.recipe')
    if not isinstance(train_cfg['grad_accum'], int) or train_cfg['grad_accum'] < 1:
        raise ValueError('train.grad_accum must be a positive integer')
    if data_cfg["input_size"] != 224:
        raise ValueError("data.input_size must be 224")
    if knowledge["text_dim"] != 768:
        raise ValueError("knowledge.text_dim must be 768")
    if knowledge["text_encoder_frozen"] is not True:
        raise ValueError("knowledge.text_encoder_frozen must be true")
    if knowledge["dim"] % knowledge["num_heads"] != 0:
        raise ValueError("knowledge.dim must be divisible by knowledge.num_heads")
    if not 0 <= knowledge["attention_dropout"] < 1:
        raise ValueError("knowledge.attention_dropout must be in [0, 1)")
    if model["aggregation"] not in {"learned", "mean"}:
        raise ValueError("model.aggregation must be 'learned' or 'mean'")
    if model["head"]["type"] not in {"linear", "mlp"}:
        raise ValueError("model.head.type must be 'linear' or 'mlp'")
    if model["use_contrastive"] and not model["use_knowledge"]:
        raise ValueError("model.use_contrastive requires model.use_knowledge")
    weight = loss["contrastive_weight"]
    if model["use_contrastive"] and weight <= 0:
        raise ValueError("loss.contrastive_weight must be > 0 when contrastive is enabled")
    if not model["use_contrastive"] and weight != 0:
        raise ValueError("loss.contrastive_weight must be 0 when contrastive is disabled")
    _require_positive("loss.contrastive_temperature", loss["contrastive_temperature"])
    _require_positive("train.epochs", train_cfg["epochs"])
    _require_positive("train.lr", train_cfg["lr"])
    _require_positive("train.weight_decay", train_cfg["weight_decay"], allow_zero=True)
    _require_positive("train.step_size", train_cfg["step_size"])
    _require_positive("train.gamma", train_cfg["gamma"])
    _require_positive("data.batch_size", data_cfg["batch_size"])
    if not isinstance(data_cfg["num_workers"], int) or data_cfg["num_workers"] < 0:
        raise ValueError("data.num_workers must be a non-negative integer")
    if train_cfg["resume"] and data["run"]["overwrite"]:
        raise ValueError("train.resume and run.overwrite cannot both be true")
    if train_cfg["early_stopping"]["enabled"]:
        _require_positive("train.early_stopping.patience", train_cfg["early_stopping"]["patience"])

    categories = metadata["categories"]
    metadata_dim = 1 + sum(len(categories[name]) for name in ("gender", "region", "view"))
    if metadata_dim != 9:
        raise ValueError(f"metadata categories must produce 9 features, got {metadata_dim}")
    if metadata["features"] != ["age", "gender", "region", "view"]:
        raise ValueError("metadata.features must be [age, gender, region, view]")

    required_paths = ["image_dir", "splits_csv", "class_weights_csv", "output_dir"]
    if model["use_metadata"]:
        required_paths.append("metadata_csv")
    if model["use_knowledge"] and require_training_assets:
        required_paths.extend(["kb_json", "text_embedding_cache"])
    missing = [name for name in required_paths if not data["paths"].get(name)]
    if missing:
        raise ValueError(f"Missing required path(s): {', '.join('paths.' + name for name in missing)}")


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | list[str] | None = None,
    *,
    require_training_assets: bool = True,
) -> Config:
    """Load a YAML config, recursively merge bases, apply dotted overrides, and validate."""
    path = Path(path).resolve()
    data = _read_with_bases(path)

    root = path
    while True:
        raw = yaml.safe_load(root.read_text(encoding="utf-8")) or {}
        base_ref = raw.get("base")
        if base_ref is None:
            break
        root = (root.parent / base_ref).resolve()
    schema = _read_with_bases(root)

    _apply_overrides(data, overrides)
    if data['train'].get('recipe') == 'auto':
        data['train']['recipe'] = data['recipes']['backbones'][data['model']['backbone']]
    recipe = data['recipes'][data['train']['recipe']]
    for section in ('train', 'data'):
        for field, value in recipe[section].items():
            if data[section].get(field) == 'auto':
                data[section][field] = deepcopy(value)
    if data["run"]["name"] is None:
        data["run"]["name"] = path.stem
    validate_config(data, schema, require_training_assets=require_training_assets)
    return Config(data, path)
