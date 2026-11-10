"""Locked eight-class classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from btxrd.constants import CLASS_NAMES

def clinical_taxonomy():
    from pathlib import Path
    from btxrd.knowledge.taxonomy import load_taxonomy
    return load_taxonomy(Path(__file__).resolve().parents[3] / 'configs/taxonomy/btxrd_8class.yaml', CLASS_NAMES)


def hierarchical_metrics(y_true, y_pred, taxonomy):
    from sklearn.metrics import f1_score
    true = [CLASS_NAMES[int(i)] for i in y_true]
    pred = [CLASS_NAMES[int(i)] for i in y_pred]
    def group(c):
        path = taxonomy.paths[taxonomy.leaf_of_class[c]]
        return 0 if 'normal' in path else (1 if 'benign' in path else 2)
    a, b = np.array([group(c) for c in true]), np.array([group(c) for c in pred])
    tumour = a != 0
    result = {
        'normal_vs_tumour_f1': float(f1_score(a != 0, b != 0, labels=[False, True], average='macro', zero_division=0)) if len(a) else float('nan'),
        'benign_vs_malignant_f1': float(f1_score(a[tumour], b[tumour], labels=[1, 2], average='macro', zero_division=0)) if tumour.any() else float('nan'),
        'severe_error_rate': float((a != b).mean()) if len(a) else float('nan'),
    }
    for p in taxonomy.decision_nodes:
        def child(c):
            path = taxonomy.paths[taxonomy.leaf_of_class[c]]
            return next((n for n in path if taxonomy.parent[n] == p), None)
        pairs = [(child(t), child(pr)) for t, pr in zip(true, pred) if child(t) is not None]
        result[f'node_accuracy_{p}'] = float(np.mean([t == pr for t, pr in pairs])) if pairs else float('nan')
    return result


def compute_metrics(y_true, y_pred, y_prob=None, class_names: list[str] = CLASS_NAMES) -> dict[str, Any]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
    }
    for index, name in enumerate(class_names):
        metrics.update({
            f"{name}_precision": float(precision[index]),
            f"{name}_recall": float(recall[index]),
            f"{name}_f1": float(f1[index]),
            f"{name}_support": int(support[index]),
        })
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=np.float64)
        if y_prob.shape != (len(y_true), len(class_names)):
            raise ValueError(f"y_prob must have shape [{len(y_true)}, {len(class_names)}]")
        aucs: list[float] = []
        for index, name in enumerate(class_names):
            binary = (y_true == index).astype(np.int64)
            if np.unique(binary).size < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(binary, y_prob[:, index]))
                aucs.append(auc)
            metrics[f"{name}_auroc"] = auc
        metrics["macro_auroc_ovr"] = float(np.mean(aucs)) if aucs else float("nan")
        metrics["n_auroc_classes"] = len(aucs)
    return metrics
