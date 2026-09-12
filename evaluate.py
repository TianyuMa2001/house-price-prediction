"""Reproducible random/group/time experiments, ablations and artifact export."""
import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.inspection import permutation_importance
from proj import (create_pipeline, load_csv, split_features_target, validate_schema,
                  CATEGORICAL_FEATURES)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def metrics(actual_log, predicted_log):
    actual, predicted = np.exp(actual_log), np.exp(predicted_log)
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite predictions')
    residual = predicted - actual
    return {'mae': float(np.abs(residual).mean()),
            'rmse': float(np.sqrt(np.mean(residual ** 2))),
            'log_rmse': float(np.sqrt(np.mean((predicted_log - actual_log) ** 2)))}


def split_indices(X, strategy, seed=42):
    indices = np.arange(len(X))
    if 'PIN' not in X or X['PIN'].isna().any():
        raise ValueError('PIN is required and cannot be missing for validation.')
    if strategy == 'random':
        train, valid = train_test_split(indices, test_size=.2, random_state=seed)
    elif strategy == 'group':
        train, valid = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(X, groups=X['PIN']))
    elif strategy == 'time':
        # Latest calendar year is held out; prior transactions of the same PIN
        # are purged, so the experiment tests future, previously unseen properties.
        years = pd.to_numeric(X['Sale Year'], errors='raise')
        if years.isna().any() or years.nunique() < 2:
            raise ValueError('Time split requires at least two complete sale years.')
        valid = indices[years.eq(years.max())]
        held_pins = set(X.iloc[valid]['PIN'])
        train = indices[years.lt(years.max()) & ~X['PIN'].isin(held_pins)]
    else:
        raise ValueError(strategy)
    if not len(train) or not len(valid) or set(train) & set(valid):
        raise ValueError('Invalid/overlapping split')
    if strategy != 'random' and set(X.iloc[train]['PIN']) & set(X.iloc[valid]['PIN']):
        raise AssertionError('Property leakage')
    return train, valid


def quality(data):
    result = {'rows': len(data), 'columns': len(data.columns),
              'missing_fraction': data.isna().mean().to_dict(),
              'duplicate_rows': int(data.duplicated().sum()),
              'duplicate_pin_rows': int(data['PIN'].duplicated().sum()),
              'unique_pins': int(data['PIN'].nunique())}
    if 'Sale Price' in data:
        prices = pd.to_numeric(data['Sale Price'], errors='coerce')
        result['invalid_price_rows'] = int((~np.isfinite(prices) | prices.lt(500)).sum())
    return result


def unknown_categories(train, other):
    return {c: int((other[c].notna() & ~other[c].isin(train[c].dropna().unique())).sum())
            for c in CATEGORICAL_FEATURES}


def grouped_errors(X, y, predicted, tag):
    frame = X[['Property Class', 'Town Code']].copy()
    frame['price_band'] = pd.cut(np.exp(y), [0, 100000, 250000, 500000, 1000000, np.inf],
                                 labels=['<100k', '100k-250k', '250k-500k', '500k-1m', '>=1m'])
    frame['actual_log'], frame['predicted_log'] = y, predicted
    rows = []
    for dimension in ['price_band', 'Property Class', 'Town Code']:
        for group, subset in frame.groupby(dimension, observed=True, dropna=False):
            rows.append(dict(tag, dimension=dimension, group=str(group), rows=len(subset),
                             **metrics(subset.actual_log.to_numpy(), subset.predicted_log.to_numpy())))
    return rows


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=base/'cook_county_contest_data/cook_county_contest_train.csv')
    parser.add_argument('--test', type=Path, default=base/'cook_county_contest_test.csv')
    parser.add_argument('--output', type=Path, default=base/'artifacts')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--splits', nargs='+', choices=['random','group','time'], default=['random','group','time'])
    parser.add_argument('--importance-rows', type=int, default=1500)
    parser.add_argument('--without-estimates', action='store_true', help='Final exported model excludes estimates')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(args.output/'run.log', encoding='utf-8'), logging.StreamHandler()])
    started = time.time()
    raw, test = load_csv(args.train), load_csv(args.test)
    validate_schema(raw)
    validate_schema(test)
    X, y = split_features_target(raw)
    q = {'training': quality(raw), 'test': quality(test),
         'test_unknown_categories': unknown_categories(X, test),
         'test_pin_overlap_with_training': len(set(X.PIN) & set(test.PIN))}
    write_json(args.output/'data_quality.json', q)
    rows, splits, residuals = [], [], []
    for strategy in args.splits:
        for seed in (args.seeds[:1] if strategy == 'time' else args.seeds):
            a, b = split_indices(X, strategy, seed)
            splits.append({'split':strategy, 'seed':seed, 'train_rows':len(a), 'validation_rows':len(b),
                           'row_overlap':len(set(a)&set(b)),
                           'pin_overlap':len(set(X.iloc[a].PIN)&set(X.iloc[b].PIN)),
                           'train_year_max':int(X.iloc[a]['Sale Year'].max()),
                           'validation_year_min':int(X.iloc[b]['Sale Year'].min()),
                           'unknown_categories':unknown_categories(X.iloc[a],X.iloc[b])})
            np.savez_compressed(args.output/f'split_{strategy}_{seed}.npz', train=a, validation=b)
            for estimates in [True, False]:
                for kind in ['median', 'ridge', 'histgb']:
                    tag = {'split':strategy,'seed':seed,'model':kind,'estimates':estimates}
                    model = create_pipeline(seed, kind, estimates)
                    fit_start = time.time()
                    model.fit(X.iloc[a], y[a])
                    predicted = model.predict(X.iloc[b])
                    row = dict(tag, train_rows=len(a), validation_rows=len(b),
                               seconds=time.time()-fit_start, **metrics(y[b], predicted))
                    rows.append(row)
                    logging.info('%s', row)
                    pd.DataFrame(rows).to_csv(args.output/'metrics.csv', index=False)
                    if kind == 'histgb' and seed == args.seeds[0]:
                        residuals.extend(grouped_errors(X.iloc[b], y[b], predicted, tag))
                        if strategy == 'group' and args.importance_rows > 0:
                            sample = np.random.default_rng(seed).choice(b, min(len(b),args.importance_rows),replace=False)
                            importance = permutation_importance(model, X.iloc[sample], y[sample],
                                scoring='neg_root_mean_squared_error', n_repeats=3, random_state=seed, n_jobs=1)
                            pd.DataFrame({'feature':X.columns, 'log_rmse_increase':importance.importances_mean,
                                          'std':importance.importances_std}).sort_values('log_rmse_increase',ascending=False).to_csv(
                                          args.output/f'importance_estimates_{estimates}.csv',index=False)
    write_json(args.output/'splits.json', splits)
    write_json(args.output/'metrics.json', rows)
    pd.DataFrame(residuals).to_csv(args.output/'grouped_errors.csv', index=False)
    final = create_pipeline(args.seeds[0], include_estimates=not args.without_estimates)
    final.fit(X,y)
    predictions = final.predict(test)
    if len(predictions) != len(test) or not np.isfinite(predictions).all():
        raise ValueError('Invalid output predictions')
    joblib.dump(final,args.output/'pipeline.joblib.gz',compress=('gzip',3))
    pd.DataFrame({'Sale Price':predictions}).to_csv(args.output/'predictions.csv')
    np.testing.assert_allclose(joblib.load(args.output/'pipeline.joblib.gz').predict(test.iloc[:20]),predictions[:20])
    metadata = {'python':sys.version,'platform':platform.platform(),'sklearn':sklearn.__version__,
                'numpy':np.__version__,'pandas':pd.__version__,'seeds':args.seeds,'splits':args.splits,
                'training_rows':len(X),'prediction_rows':len(test),'seconds':time.time()-started,
                'final_model':'histgb','final_includes_estimates':not args.without_estimates,
                'test_source':'https://raw.githubusercontent.com/DS-100/fa22/master/proj/proj1b/cook_county_contest_test.csv',
                'test_source_limitation':'Public Berkeley dataset; identity with the local ECE grading set is unverified.',
                'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.train,args.test,Path(__file__),base/'proj.py']}}
    write_json(args.output/'run_metadata.json',metadata)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    table = pd.DataFrame(rows)
    summary = table.groupby(['split','model','estimates']).agg(rmse_mean=('rmse','mean'),rmse_std=('rmse','std'),
                                                              mae_mean=('mae','mean'),log_rmse_mean=('log_rmse','mean')).reset_index()
    summary.to_csv(args.output/'metrics_summary.csv',index=False)
    fig, ax = plt.subplots(figsize=(12,6))
    labels = [f'{r.split}/{r.model}/{"est" if r.estimates else "no-est"}' for r in summary.itertuples()]
    ax.bar(labels,summary.rmse_mean,yerr=summary.rmse_std.fillna(0),capsize=3)
    ax.set_ylabel('Validation RMSE (USD)'); ax.tick_params(axis='x',rotation=80)
    fig.tight_layout(); fig.savefig(args.output/'model_comparison.png',dpi=150); plt.close(fig)
    logging.info('Completed; artifacts: %s',args.output)


if __name__ == '__main__':
    main()
