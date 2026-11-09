"""Strict metadata preprocessing fitted only on the training split."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MetadataPreprocessor:
    categories: dict[str, list[str]]
    columns: dict[str, str]
    age_mean: float | None = None
    age_std: float | None = None

    @classmethod
    def from_config(cls, cfg, stats: dict[str, float] | None = None) -> "MetadataPreprocessor":
        metadata = cfg["metadata"] if isinstance(cfg, dict) else cfg.metadata.to_dict()
        data = cfg["data"] if isinstance(cfg, dict) else cfg.data.to_dict()
        instance = cls(
            categories={key: list(value) for key, value in metadata["categories"].items()},
            columns=dict(data["columns"]),
        )
        if stats is not None:
            instance.age_mean = float(stats["age_mean"])
            instance.age_std = float(stats["age_std"])
            if not np.isfinite(instance.age_mean) or not np.isfinite(instance.age_std) or instance.age_std <= 0:
                raise ValueError("Invalid metadata_stats in checkpoint")
        return instance

    @property
    def output_dim(self) -> int:
        return 1 + sum(len(self.categories[name]) for name in ("gender", "region", "view"))

    def _validate_frame(self, frame: pd.DataFrame) -> pd.Series:
        required = [self.columns[name] for name in ("image_id", "age", "gender", "region", "view")]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Metadata missing column(s): {', '.join(missing)}")
        image_col = self.columns["image_id"]
        if frame[image_col].isna().any() or frame[image_col].duplicated().any():
            raise ValueError("Metadata image_id values must be non-null and unique")
        age = pd.to_numeric(frame[self.columns["age"]], errors="coerce")
        if age.isna().any() or not np.isfinite(age.to_numpy(dtype=np.float64)).all():
            raise ValueError("Metadata age values must be numeric and finite")
        for feature in ("gender", "region", "view"):
            column = self.columns[feature]
            invalid = sorted(set(frame[column].dropna()) - set(self.categories[feature]))
            if frame[column].isna().any() or invalid:
                raise ValueError(f"Invalid {feature} category values: {invalid or ['<missing>']}")
        return age

    def fit(self, train_df: pd.DataFrame) -> "MetadataPreprocessor":
        age = self._validate_frame(train_df)
        self.age_mean = float(age.mean())
        self.age_std = float(age.std(ddof=0))
        if not np.isfinite(self.age_mean) or not np.isfinite(self.age_std) or self.age_std <= 0:
            raise ValueError("Training age standard deviation must be finite and > 0")
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.age_mean is None or self.age_std is None:
            raise RuntimeError("MetadataPreprocessor must be fitted before transform")
        age = self._validate_frame(frame).to_numpy(dtype=np.float32)
        parts = [((age - self.age_mean) / self.age_std).reshape(-1, 1)]
        for feature in ("gender", "region", "view"):
            values = frame[self.columns[feature]].to_numpy()
            parts.append(
                np.column_stack([(values == category).astype(np.float32) for category in self.categories[feature]])
            )
        encoded = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        if encoded.shape[1] != 9:
            raise ValueError(f"Metadata encoding must have dimension 9, got {encoded.shape[1]}")
        return encoded

    def fit_transform(self, train_df: pd.DataFrame) -> np.ndarray:
        return self.fit(train_df).transform(train_df)

    def state_dict(self) -> dict[str, float]:
        if self.age_mean is None or self.age_std is None:
            raise RuntimeError("MetadataPreprocessor has not been fitted")
        return {"age_mean": self.age_mean, "age_std": self.age_std}
