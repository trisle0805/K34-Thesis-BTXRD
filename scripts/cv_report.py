"""Validate OOF predictions and report case-bootstrap CV comparisons."""
import argparse
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_recall_fscore_support, confusion_matrix
from btxrd.constants import CLASS_NAMES

PROBS = ['p_' + c for c in CLASS_NAMES]
LABELS = list(range(8))
BASELINES = ['b0_densenet169', 'b_resnet50', 'b_efficientnet_b3', 'b_vit_b16', 'b_biomedclip_vitb16']
A1, A2, A3, A4 = ['a1_densenet169_pecl', 'a2_densenet169_kdecoder', 'a3_densenet169_kdecoder_pecl', 'a4_densenet169_kdecoder_pecl_shuffled']


def metrics(frame):
    y, pred = frame.y_true, frame.y_pred
    try:
        auc = roc_auc_score(y, frame[PROBS], labels=LABELS, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')
    return {'accuracy': accuracy_score(y, pred), 'balanced_accuracy': recall_score(y, pred, labels=LABELS, average='macro', zero_division=0),
            'macro_f1': f1_score(y, pred, labels=LABELS, average='macro', zero_division=0), 'macro_auroc': auc}


def bootstrap_indices(groups, n_boot, seed):
    unique = np.unique(groups)
    members = [np.flatnonzero(np.asarray(groups) == g) for g in unique]
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])


def interval(values):
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    return np.quantile(finite, [.025, .975]).tolist() if len(finite) else [float('nan')] * 2


def report(cv_dir, folds_csv, configs, out, n_boot=2000, seed=2026):
    if n_boot < 1:
        raise ValueError('n_boot must be positive')
    folds = pd.read_csv(folds_csv, dtype={'image_id': str, 'group_id': str})
    if folds.image_id.duplicated().any() or folds[['image_id', 'group_id', 'class8', 'fold']].isna().any().any():
        raise ValueError('Invalid folds manifest')
    if set(folds.fold) != set(range(5)):
        raise ValueError('Expected five folds')
    if (folds.groupby('group_id').fold.nunique() != 1).any():
        raise ValueError('Groups cross outer folds')
    folds = folds.sort_values('image_id').reset_index(drop=True)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pooled, per_fold, per_class, complete = [], [], [], {}
    notes = []
    for config in configs:
        paths = [Path(cv_dir) / config / f'{config}_fold{k}' / 'evaluation' / 'predictions_test.csv' for k in range(5)]
        if not all(p.is_file() for p in paths):
            message = f'{config}: incomplete five-fold predictions; skipped'
            warnings.warn(message)
            notes.append(message)
            continue
        pieces = []
        for k, path in enumerate(paths):
            f = pd.read_csv(path, dtype={'image_id': str})
            if f.image_id.duplicated().any() or set(f.image_id) != set(folds.loc[folds.fold == k, 'image_id']):
                raise ValueError(f'{config} fold {k}: test IDs mismatch')
            probabilities = f[PROBS].to_numpy()
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5):
                raise ValueError('Invalid probabilities')
            if not set(f.y_pred) <= set(LABELS):
                raise ValueError('Invalid predicted labels')
            f = f.drop(columns=['fold', 'group_id', 'class8'], errors='ignore').merge(folds[['image_id', 'group_id', 'fold', 'class8']], on='image_id', validate='one_to_one')
            expected = f.class8.map(dict(zip(CLASS_NAMES, LABELS)))
            if not np.array_equal(f.y_true, expected):
                raise ValueError('Prediction labels disagree with folds manifest')
            per_fold.append({'config': config, 'fold': k, **metrics(f)})
            pieces.append(f)
        oof = pd.concat(pieces).sort_values('image_id').reset_index(drop=True)
        if len(oof) != len(folds) or oof.image_id.duplicated().any():
            raise ValueError('OOF coverage mismatch')
        complete[config] = oof
        oof.to_csv(out / f'oof_{config}.csv', index=False)
        point = metrics(oof)
        boot = [metrics(oof.iloc[ix]) for ix in bootstrap_indices(oof.group_id, n_boot, seed)]
        row = {'config': config, **point}
        for key in point:
            row[key + '_ci_low'], row[key + '_ci_high'] = interval([b[key] for b in boot])
        pooled.append(row)
        precision, recall, f1, support = precision_recall_fscore_support(oof.y_true, oof.y_pred, labels=LABELS, zero_division=0)
        per_class.extend({'config': config, 'class': c, 'precision': p, 'recall': r, 'f1': f, 'support': int(s)} for c, p, r, f, s in zip(CLASS_NAMES, precision, recall, f1, support))
        pd.DataFrame(confusion_matrix(oof.y_true, oof.y_pred, labels=LABELS), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out / f'confusion_{config}.csv')
    pd.DataFrame(pooled).to_csv(out / 'metrics_pooled.csv', index=False)
    fold_metrics = pd.DataFrame(per_fold)
    fold_metrics.to_csv(out / 'metrics_per_fold.csv', index=False)
    summary = fold_metrics.groupby('config').macro_f1.agg(['mean', 'std']) if per_fold else pd.DataFrame(columns=['mean', 'std'])
    summary.to_csv(out / 'macro_f1_mean_sd.csv')
    pd.DataFrame(per_class).to_csv(out / 'per_class_pooled.csv', index=False)
    pairs = [('Q1', A2, BASELINES[0]), ('Q2', A1, BASELINES[0]), ('Q2', A3, A2),
             ('Q3', A3, BASELINES[0]), ('Q3', A3, A1), ('Q3', A3, A2), ('Q4', A3, A4)]
    pairs += [('Q5', A3, b) for b in BASELINES]
    diffs = []
    for q, a, b in pairs:
        if a not in complete or b not in complete:
            continue
        fa, fb = complete[a], complete[b]
        values = [metrics(fa.iloc[ix])['macro_f1'] - metrics(fb.iloc[ix])['macro_f1'] for ix in bootstrap_indices(fa.group_id, n_boot, seed)]
        low, high = interval(values)
        wins = sum(metrics(fa[fa.fold == k])['macro_f1'] > metrics(fb[fb.fold == k])['macro_f1'] for k in range(5))
        diffs.append({'question': q, 'a': a, 'b': b, 'macro_f1_diff': metrics(fa)['macro_f1'] - metrics(fb)['macro_f1'], 'ci_low': low, 'ci_high': high, 'p_one_sided': float(np.mean(np.asarray(values) <= 0)), 'folds_won': wins})
    pd.DataFrame(diffs).to_csv(out / 'paired_diff.csv', index=False)
    text = ['# Grouped five-fold CV', '', 'Groups are heuristic cases, not verified patient IDs. CIs resample cases with replacement.',
            'AUROC bootstrap draws missing any class are undefined and excluded from AUROC intervals.', '', '## Baselines', '| Config | Pooled Macro-F1 | 95% CI |', '|---|---:|---|']
    text += [f"| {r['config']} | {r['macro_f1']:.4f} | [{r['macro_f1_ci_low']:.4f}, {r['macro_f1_ci_high']:.4f}] |" for r in pooled if r['config'] in BASELINES]
    text += ['', '## Ablation (RAD Table 5 style)', '| Config | Decoder + auxiliary CE | PECL | Knowledge | Macro-F1 |', '|---|---|---|---|---:|']
    for r in pooled:
        c = r['config']
        if c in [BASELINES[0], A1, A2, A3, A4]:
            text.append(f"| {c} | {'✓' if c in [A2,A3,A4] else '×'} | {'✓' if c in [A1,A3,A4] else '×'} | {'shuffled' if c == A4 else ('real' if c != BASELINES[0] else '—')} | {r['macro_f1']:.4f} |")
    text += ['', '## Q1–Q5 paired comparisons', '| Question | A − B | Difference [95% CI] | Wins / 5 | Conclusion |', '|---|---|---|---|---|']
    for r in diffs:
        conclusion = 'significantly better' if r['ci_low'] > 0 else ('significantly worse' if r['ci_high'] < 0 else 'no statistically significant difference')
        text.append(f"| {r['question']} | {r['a']} − {r['b']} | {r['macro_f1_diff']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] | {r['folds_won']} | {conclusion} |")
    text += ['', 'Q2 requires both point differences positive; significance is assessed separately for each interval.', 'Q3 requires A3 highest among A0/A1/A2/A3 and its A0 interval excluding zero.', 'See per_class_pooled.csv for every comparison, including giant_cell_tumor, osteofibroma and synovial_osteochondroma.', '', '## Warnings'] + notes
    (out / 'REPORT.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    return complete, diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cv-dir', 'folds-csv', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--configs', nargs='+', required=True)
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=2026)
    report(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
