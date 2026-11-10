"""Fixed node vectors, with no proposition dependency for name-only runs."""
from pathlib import Path

import torch
from torch.nn import functional as F

from btxrd.constants import CLASS_NAMES
from btxrd.knowledge.taxonomy import load_taxonomy, flat_taxonomy, assign_propositions
from btxrd.models.class_decoder import build_class_prototypes


def taxonomy_path(cfg):
    path = Path(cfg.get_path('knowledge.taxonomy_file', 'taxonomy/btxrd_8class.yaml'))
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / 'configs' / path


def load_node_names(path, taxonomy, model_name):
    cache = torch.load(path, map_location='cpu', weights_only=False)
    for key, expected in {'node_ids': taxonomy.node_ids, 'texts': [taxonomy.names[n] for n in taxonomy.node_ids],
                          'taxonomy_sha256': taxonomy.sha256, 'model_name': model_name, 'pooling': 'cls'}.items():
        if cache.get(key) != expected:
            raise ValueError(f'Node name cache {key} mismatch')
    vectors = cache.get('embeddings')
    if not isinstance(vectors, torch.Tensor) or vectors.shape != (len(taxonomy.node_ids), 768) or not torch.isfinite(vectors).all():
        raise ValueError('Invalid node name embeddings')
    if not isinstance(cache.get('max_length'), int) or cache['max_length'] <= 0:
        raise ValueError('Invalid node name max_length')
    return F.normalize(vectors.float(), dim=-1)


def build_node_assets(cfg, embeddings=None, class_to_ids=None, kb_ids=None, kb=None):
    taxonomy = (flat_taxonomy(CLASS_NAMES) if cfg.knowledge.hierarchy == 'flat'
                else load_taxonomy(taxonomy_path(cfg), CLASS_NAMES))
    source = cfg.knowledge.query_source
    prototype = cfg.knowledge.pecl_prototype
    active = cfg.loss.pecl_weight > 0
    assignment = {}
    names = None
    if source == 'name' or (active and prototype == 'name'):
        names = load_node_names(cfg.paths.node_name_cache, taxonomy, cfg.knowledge.text_encoder_name)
    if source != 'name' or (active and prototype != 'name'):
        if embeddings is None or kb_ids is None or class_to_ids is None:
            raise ValueError('Proposition assets required')
        if taxonomy.decision_nodes == ['root']:
            flat = build_class_prototypes(embeddings, class_to_ids, kb_ids)
            own = sub = flat
        else:
            from types import SimpleNamespace
            assignment = assign_propositions(kb or SimpleNamespace(ids=kb_ids, class_to_ids=class_to_ids), taxonomy)
            normalized = F.normalize(embeddings.float(), dim=-1)
            index = {p: i for i, p in enumerate(kb_ids)}
            sub_ids = {n: [p for child, props in assignment.items() if child == n or n in taxonomy.paths[child] for p in props]
                       for n in taxonomy.node_ids}
            def aggregate(groups):
                rows = []
                for n in taxonomy.node_ids:
                    ids = groups[n] or sub_ids[n]
                    if not ids:
                        raise ValueError(f'No proposition vectors for node {n}')
                    rows.append(normalized[[index[p] for p in ids]].mean(0))
                return F.normalize(torch.stack(rows), dim=-1)
            own, sub = aggregate(assignment), aggregate(sub_ids)
    text = names if source == 'name' else (own if source == 'own' else flat)
    prototypes = (names if prototype == 'name' else sub) if active else None
    return taxonomy, text, prototypes, assignment