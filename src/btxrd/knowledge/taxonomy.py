"""Validated, serializable classification trees and deterministic LCA assignment."""
from dataclasses import dataclass, asdict
import hashlib
import logging
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Taxonomy:
    node_ids: list[str]
    parent: dict[str, str]
    children: dict[str, list[str]]
    decision_nodes: list[str]
    leaf_of_class: dict[str, str]
    paths: dict[str, list[str]]
    names: dict[str, str]
    sha256: str

    def to_dict(self):
        return asdict(self)


def _build(nodes, class_names, digest):
    ids = [n['id'] for n in nodes]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate taxonomy node')
    roots = [n['id'] for n in nodes if n.get('parent') is None]
    if len(roots) != 1:
        raise ValueError('Taxonomy requires exactly one root')
    root = roots[0]
    parent = {n['id']: n['parent'] for n in nodes if n['id'] != root}
    children = {}
    for child, p in parent.items():
        if p not in ids:
            raise ValueError('Unknown parent')
        children.setdefault(p, []).append(child)
    if any(len(c) < 2 for c in children.values()):
        raise ValueError('Decision nodes require at least two children')
    paths = {}
    for node in ids:
        path, current = [], node
        while current != root:
            if current in path:
                raise ValueError('Taxonomy cycle')
            path.append(current)
            current = parent[current]
        paths[node] = path[::-1]
    leaves = {}
    for n in nodes:
        if n['id'] not in children:
            label = n.get('leaf_class')
            if label is None or label in leaves:
                raise ValueError('Missing or duplicate leaf_class')
            leaves[label] = n['id']
        elif 'leaf_class' in n:
            raise ValueError('Internal node cannot have leaf_class')
    if set(leaves) != set(class_names):
        raise ValueError('Leaves must cover class_names exactly')
    return Taxonomy([n for n in ids if n != root], parent, children,
                    [n for n in ids if n in children],
                    {c: leaves[c] for c in class_names}, paths,
                    {n['id']: n['name'] for n in nodes}, digest)


def load_taxonomy(path, class_names):
    raw = Path(path).read_bytes()
    return _build(yaml.safe_load(raw)['nodes'], class_names, hashlib.sha256(raw).hexdigest())


def flat_taxonomy(class_names):
    nodes = [{'id': 'root', 'parent': None, 'name': 'bone radiograph'}]
    nodes += [dict(id=c, parent='root', name=c.replace('_', ' '), leaf_class=c) for c in class_names]
    digest = hashlib.sha256(yaml.safe_dump(nodes, sort_keys=True).encode()).hexdigest()
    return _build(nodes, class_names, digest)


def assign_propositions(kb, taxonomy):
    result = {n: [] for n in taxonomy.node_ids}
    for prop in kb.ids:
        paths = [taxonomy.paths[taxonomy.leaf_of_class[c]] for c, values in kb.class_to_ids.items() if prop in values]
        common = []
        for level in zip(*paths):
            if len(set(level)) != 1:
                break
            common.append(level[0])
        if common:
            result[common[-1]].append(prop)
        else:
            logging.warning('Proposition %s has root LCA; excluded from node assignment', prop)
    return result