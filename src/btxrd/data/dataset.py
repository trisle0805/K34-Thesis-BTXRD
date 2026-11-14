"""Dataset validation, datasets, and dataloader construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from btxrd.constants import CLASS_NAMES, VALID_SPLITS
from btxrd.data.metadata import MetadataPreprocessor
from btxrd.data.transforms import build_eval_transform, build_train_transform


@dataclass
class DataBundle:
    loaders: dict[str, DataLoader]
    class_weights: torch.Tensor
    metadata_preprocessor: MetadataPreprocessor | None
    train_generator: torch.Generator | None
    balance_info: dict


class BTXRDDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_dir: str | Path, columns: dict[str, str], image_ext: str, transform, metadata: np.ndarray | None = None):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.columns = columns
        self.image_ext = image_ext
        self.transform = transform
        self.metadata = metadata
        self.patient_cache = None
        self.class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, image_id: str) -> Path:
        return self.image_dir / f"{Path(str(image_id)).stem}{self.image_ext}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_id = str(row[self.columns["image_id"]])
        with Image.open(self._image_path(image_id)) as image:
            tensor = self.transform(image.convert("L"))
        sample: dict[str, Any] = {
            "image": tensor,
            "label": self.class_to_index[str(row[self.columns["label"]])],
            "image_id": image_id,
        }
        if self.metadata is not None:
            sample["meta"] = torch.from_numpy(self.metadata[index])
        if self.patient_cache is not None:
            sample['text_tokens'], sample['text_mask'] = self.patient_cache[image_id]
        return sample


def load_splits_frame(cfg) -> pd.DataFrame:
    columns = cfg.data.columns.to_dict()
    frame = pd.read_csv(cfg.paths.splits_csv)
    required = [columns[name] for name in ("image_id", "label", "split")]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Splits CSV missing column(s): {', '.join(missing)}")
    image_col, label_col, split_col = required
    if frame[image_col].isna().any() or frame[image_col].duplicated().any():
        raise ValueError("splits image_id values must be non-null and unique")
    invalid_labels = sorted(set(frame[label_col]) - set(CLASS_NAMES))
    if invalid_labels:
        raise ValueError(f"Unknown class labels: {invalid_labels}")
    invalid_splits = sorted(set(frame[split_col]) - VALID_SPLITS)
    if invalid_splits:
        raise ValueError(f"Unknown split values: {invalid_splits}")
    expected = cfg.data.expected_split_counts
    if expected is not None:
        counts = frame[split_col].value_counts().to_dict()
        expected_dict = expected.to_dict()
        mismatches = {name: (counts.get(name, 0), count) for name, count in expected_dict.items() if counts.get(name, 0) != count}
        if mismatches:
            raise ValueError(f"Split counts do not match expected values: {mismatches}")
    return frame


def load_class_weights(cfg) -> torch.Tensor:
    columns = cfg.data.columns.to_dict()
    frame = pd.read_csv(cfg.paths.class_weights_csv)
    label_col = columns["label"]
    weight_col = columns["class_weight"]
    missing = [column for column in (label_col, weight_col) if column not in frame.columns]
    if missing:
        raise ValueError(f"Class-weights CSV missing column(s): {', '.join(missing)}")
    labels = frame[label_col].tolist()
    if len(frame) != len(CLASS_NAMES) or frame[label_col].duplicated().any() or labels != CLASS_NAMES:
        raise ValueError("Class-weights rows must match CLASS_NAMES exactly and in order")
    weights = pd.to_numeric(frame[weight_col], errors="coerce").to_numpy(dtype=np.float32)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Class weights must be finite and > 0")
    return torch.from_numpy(weights)


def _merge_metadata(frame: pd.DataFrame, cfg) -> pd.DataFrame:
    columns = cfg.data.columns.to_dict()
    metadata = pd.read_csv(cfg.paths.metadata_csv)
    image_col = columns["image_id"]
    if image_col not in metadata.columns or metadata[image_col].isna().any() or metadata[image_col].duplicated().any():
        raise ValueError("Metadata image_id values must exist, be non-null, and unique")
    merged = frame.merge(metadata, on=image_col, how="left", validate="one_to_one", suffixes=("", "__meta"), indicator=True)
    if not (merged["_merge"] == "both").all():
        missing_ids = merged.loc[merged["_merge"] != "both", image_col].astype(str).head(5).tolist()
        raise ValueError(f"Metadata missing for image_id examples: {missing_ids}")
    for key in ("label", "split"):
        column = columns[key]
        metadata_column = f"{column}__meta"
        if metadata_column not in merged.columns or not merged[column].equals(merged[metadata_column]):
            raise ValueError(f"Metadata {column} values do not match splits CSV")
        merged = merged.drop(columns=[metadata_column])
    return merged.drop(columns=["_merge"])


def _validate_images(frame: pd.DataFrame, cfg, splits: tuple[str, ...]) -> None:
    columns = cfg.data.columns.to_dict()
    selected = frame[frame[columns["split"]].isin(splits)]
    image_dir = Path(cfg.paths.image_dir)
    missing = []
    for image_id in selected[columns["image_id"]].astype(str):
        path = image_dir / f"{Path(image_id).stem}{cfg.data.image_ext}"
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} image file(s); examples: {missing[:5]}")


def build_dataloaders(cfg, splits: tuple[str, ...] = ("train", "val"), metadata_stats: dict[str, float] | None = None, train_generator_state: torch.Tensor | None = None) -> DataBundle:
    if not splits or any(split not in VALID_SPLITS for split in splits):
        raise ValueError(f"Invalid requested splits: {splits}")
    frame = load_splits_frame(cfg)
    patient_cache = None
    if cfg.get_path('model.architecture', 'legacy') == 'image_text_concat' or (
        cfg.get_path('model.architecture', 'legacy') == 'pg' and cfg.get_path('model.pg_use_text', True)):
        from btxrd.data.patient_text import load_patient_cache
        patient_cache = load_patient_cache(cfg.paths.patient_token_cache, cfg.paths.patient_text_csv,
            cfg.patient_text.encoder_name, cfg.patient_text.max_length,
            frame.loc[frame[cfg.data.columns.split].isin(splits), cfg.data.columns.image_id].astype(str))
    class_weights = load_class_weights(cfg) if cfg.loss.class_weighting == "csv" else None
    columns = cfg.data.columns.to_dict()
    split_col = columns["split"]
    if cfg.model.use_metadata:
        frame = _merge_metadata(frame, cfg)
    _validate_images(frame, cfg, splits)

    limit = cfg.debug.limit_per_split
    split_frames: dict[str, pd.DataFrame] = {}
    for split in splits:
        subset = frame[frame[split_col] == split].copy().reset_index(drop=True)
        if limit is not None:
            subset = subset.head(int(limit)).reset_index(drop=True)
        split_frames[split] = subset

    train_frame = split_frames.get('train', frame[frame[split_col] == 'train'])
    counts = train_frame[columns['label']].value_counts().reindex(CLASS_NAMES, fill_value=0)
    inverse = counts.clip(lower=1).to_numpy(dtype=float) ** -0.5
    if cfg.loss.class_weighting == 'train_split':
        if (counts == 0).any():
            raise ValueError('train_split weighting requires every class in train')
        class_weights = torch.tensor(inverse / inverse.mean(), dtype=torch.float32)
    balance_info = {'train_class_counts': counts.to_dict(),
                    'ce_class_weights': class_weights.tolist() if class_weights is not None else None,
                    'sampler_class_weights': dict(zip(CLASS_NAMES, inverse.tolist())) if cfg.data.sampler == 'sqrt_inverse' else None}

    preprocessor: MetadataPreprocessor | None = None
    encoded: dict[str, np.ndarray | None] = {split: None for split in splits}
    if cfg.model.use_metadata:
        preprocessor = MetadataPreprocessor.from_config(cfg, metadata_stats)
        if metadata_stats is None:
            if "train" not in split_frames:
                raise ValueError("metadata_stats are required when building metadata loaders without train split")
            preprocessor.fit(split_frames["train"])
        for split, subset in split_frames.items():
            encoded[split] = preprocessor.transform(subset)

    train_transform = build_train_transform(cfg.data)
    eval_transform = build_eval_transform(cfg.data)
    generator: torch.Generator | None = None
    if "train" in splits:
        generator = torch.Generator().manual_seed(cfg.seed)
        if train_generator_state is not None:
            generator.set_state(train_generator_state)

    loaders: dict[str, DataLoader] = {}
    for split, subset in split_frames.items():
        dataset = BTXRDDataset(
            subset,
            cfg.paths.image_dir,
            columns,
            cfg.data.image_ext,
            train_transform if split == "train" else eval_transform,
            encoded[split],
        )
        dataset.patient_cache = patient_cache
        sampler = None
        if split == 'train' and cfg.data.sampler == 'sqrt_inverse':
            sample_weights = subset[columns['label']].map(dict(zip(CLASS_NAMES, inverse))).tolist()
            sampler = WeightedRandomSampler(sample_weights, len(subset), replacement=True, generator=generator)
        loaders[split] = DataLoader(
            dataset,
            batch_size=cfg.data.batch_size,
            shuffle=split == "train" and sampler is None,
            sampler=sampler,
            generator=generator if split == "train" else None,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
            persistent_workers=cfg.data.num_workers > 0,
        )
    return DataBundle(loaders, class_weights, preprocessor, generator, balance_info)
