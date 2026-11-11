"""Reassign cached post-exclusion vectors without modifying cache or taxonomy."""

import torch


def permute_embeddings(embeddings, ids, seed):
    indices = (torch.arange(len(ids)) if seed is None else
               torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)))
    mapping = {target: ids[source] for target, source in zip(ids, indices.tolist())}
    return (embeddings if seed is None else embeddings[indices]), mapping