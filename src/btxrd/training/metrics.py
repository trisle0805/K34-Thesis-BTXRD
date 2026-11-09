"""Locked eight-class classification metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score

from btxrd.constants import CLASS_NAMES


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
