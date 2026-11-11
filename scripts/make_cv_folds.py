"""Create reproducible, heuristic-case-grouped outer and inner CV splits."""
import argparse
import hashlib
import json
from pathlib import Path
import warnings

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from btxrd.constants import CLASS_NAMES


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_folds(splits_csv, groups_csv, group_col, out, n_folds=5, inner_val_splits=10, seed=2026, allow_missing_groups=False, label_col='class8'):
    frame = pd.read_csv(splits_csv, dtype={'image_id': str})[['image_id', label_col]].rename(columns={label_col: 'class8'})
    groups = pd.read_csv(groups_csv, dtype={'image_id': str})[['image_id', group_col]].rename(columns={group_col: 'group_id'})
    if frame.isna().any().any() or frame.image_id.duplicated().any() or groups.image_id.duplicated().any():
        raise ValueError('Input image IDs must be unique and labels non-null')
    if not set(frame.class8) <= set(CLASS_NAMES):
        raise ValueError('Unknown class labels')
    frame = frame.merge(groups, on='image_id', how='left', validate='one_to_one')
    missing = frame.loc[frame.group_id.isna(), 'image_id'].tolist()
    if missing and not allow_missing_groups:
        raise ValueError(f'Missing groups for image IDs: {missing}')
    frame['group_id'] = frame.group_id.map(lambda x: 'group:' + str(x) if pd.notna(x) else x)
    frame.loc[frame.group_id.isna(), 'group_id'] = ['singleton:' + x for x in missing]
    outer = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    frame['fold'] = -1
    for k, (_, test) in enumerate(outer.split(frame, frame.class8, frame.group_id)):
        frame.loc[test, 'fold'] = k
    report = {'seed': seed, 'missing_group_image_ids': missing, 'groups_are_heuristic': True, 'folds': {}, 'warnings': []}
    for k in range(n_folds):
        col = f'split_fold{k}'
        frame[col] = 'test'
        remaining = frame[frame.fold != k]
        inner = StratifiedGroupKFold(n_splits=inner_val_splits, shuffle=True, random_state=seed + k)
        train, val = next(inner.split(remaining, remaining.class8, remaining.group_id))
        frame.loc[remaining.index[train], col] = 'train'
        frame.loc[remaining.index[val], col] = 'val'
        no_leak = bool((frame.groupby('group_id')[col].nunique() == 1).all())
        if not no_leak:
            raise ValueError('Group leakage detected')
        counts = pd.crosstab(frame.class8, frame[col]).reindex(index=CLASS_NAMES, columns=['train', 'val', 'test'], fill_value=0)
        absent = [(c, s) for c in CLASS_NAMES for s in counts.columns if counts.loc[c, s] == 0]
        if absent:
            message = f'Fold {k}: missing class/split combinations {absent}'
            warnings.warn(message)
            report['warnings'].append(message)
        report['folds'][str(k)] = {'counts': counts.to_dict(), 'no_group_leakage': no_leak}
    report['each_image_test_once'] = bool((frame[[f'split_fold{k}' for k in range(n_folds)]].eq('test').sum(axis=1) == 1).all())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    report['sha256'] = {'splits': digest(splits_csv), 'groups': digest(groups_csv), 'output': digest(out)}
    Path(str(out) + '_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return frame, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('splits-csv', 'groups-csv', 'group-col', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--inner-val-splits', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--label-col', default='class8')
    parser.add_argument('--allow-missing-groups', action='store_true')
    make_folds(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
