"""Columnwise auxiliary AUROC with explicit valid-column/sample counts."""
import numpy as np
from sklearn.metrics import roc_auc_score


def auxiliary_auroc(targets, scores):
    targets, scores = np.asarray(targets), np.asarray(scores)
    valid = [i for i in range(targets.shape[1]) if len(np.unique(targets[:, i])) == 2]
    values = [float(roc_auc_score(targets[:, i], scores[:, i])) for i in valid]
    return dict(auroc=float(np.mean(values)) if values else None,
                valid_columns=valid, valid_column_count=len(valid), sample_count=len(targets))