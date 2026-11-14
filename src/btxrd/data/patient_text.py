"""Strict patient token cache validation (no online encoding in training)."""
from pathlib import Path
import torch
from btxrd.utils.io import sha256_file


def load_patient_cache(path, csv_path, encoder_name, max_length, required_ids):
    cache = torch.load(Path(path), map_location='cpu', weights_only=False)
    required = {'image_ids', 'tokens', 'mask', 'encoder_name', 'max_length', 'text_csv_sha256'}
    if not isinstance(cache, dict) or not required <= cache.keys():
        raise ValueError('Invalid patient cache fields')
    ids = cache['image_ids']
    if not isinstance(ids, list) or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Patient cache IDs must be unique nonempty strings')
    tokens, mask = cache['tokens'], cache['mask']
    if not isinstance(tokens, torch.Tensor) or tokens.dtype != torch.float16 or tokens.shape != (len(ids), max_length, 768):
        raise ValueError('Patient tokens must be float16 NxLx768')
    if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool or mask.shape != (len(ids), max_length):
        raise ValueError('Patient mask must be bool NxL')
    if not torch.isfinite(tokens).all() or not mask.any(-1).all():
        raise ValueError('Patient cache contains nonfinite tokens or empty masks')
    if cache['encoder_name'] != encoder_name or cache['max_length'] != max_length:
        raise ValueError('Patient cache encoder/length mismatch')
    if cache['text_csv_sha256'] != sha256_file(csv_path):
        raise ValueError('Patient text CSV SHA256 mismatch')
    if not set(map(str, required_ids)) <= set(ids):
        raise ValueError('Patient cache missing split coverage')
    return {image_id: (tokens[i], mask[i]) for i, image_id in enumerate(ids)}