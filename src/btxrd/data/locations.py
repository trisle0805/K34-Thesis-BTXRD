"""Strict location annotations and proposition cache contracts."""
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from btxrd.constants import CLASS_NAMES
from btxrd.utils.io import sha256_file

LOCATION_IDS = [f'L{i:02d}' for i in range(1, 16)]


def load_locations(path, frame, image_column, label_column):
    table = pd.read_csv(path, dtype={image_column: str})
    if image_column not in table or table[image_column].isna().any() or table[image_column].duplicated().any():
        raise ValueError('Location image IDs must be non-null and unique')
    if not set(LOCATION_IDS).issubset(table.columns):
        raise ValueError('Location labels require L01 through L15')
    values = table[LOCATION_IDS].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError('Location labels must be strict binary values without missing entries')
    indexed = table.set_index(image_column)
    ids = frame[image_column].astype(str)
    if not set(ids).issubset(indexed.index):
        raise ValueError('Location labels do not cover every split image')
    selected = indexed.loc[ids, LOCATION_IDS].to_numpy(dtype=np.float32)
    normal = frame[label_column].astype(str).str.lower().eq('normal').to_numpy()
    if (selected[normal].sum(1) != 0).any() or (selected[~normal].sum(1) < 1).any():
        raise ValueError('Normal images require zero locations; tumors require at least one')
    return dict(zip(ids, torch.from_numpy(selected)))


def location_definition(path):
    frame = pd.read_csv(path)
    if list(frame.get('id', [])) != LOCATION_IDS or 'text' not in frame or frame.text.isna().any() or not frame.text.str.strip().ne('').all():
        raise ValueError('Location CSV must define ordered L01..L15 and nonempty texts')
    return dict(ids=LOCATION_IDS, texts=frame.text.tolist(), model_name='ncbi/MedCPT-Query-Encoder',
                max_length=64, pooling='cls', csv_sha256=sha256_file(path))


def load_location_embeddings(cache_path, csv_path):
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    for key, value in location_definition(csv_path).items():
        if cache.get(key) != value:
            raise ValueError(f'Location embedding cache mismatch: {key}')
    embeddings = cache.get('embeddings')
    if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (15, 768) or not torch.isfinite(embeddings).all():
        raise ValueError('Location embeddings must be finite 15x768')
    return embeddings.float()