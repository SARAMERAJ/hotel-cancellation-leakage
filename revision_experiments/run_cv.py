# 5-fold CV runner for the revision experiments.
# usage: python run_cv.py NAME [--raw] [--group] [--global-enc] [--resample smote|smotenc|none]
#                         [--class-weight] [--drop col1,col2] [--models m1,m2]
import sys, argparse, json, time
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from common import *
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--raw', action='store_true')
ap.add_argument('--group', action='store_true')
ap.add_argument('--global-enc', action='store_true')
ap.add_argument('--resample', default='smote')
ap.add_argument('--class-weight', action='store_true')
ap.add_argument('--drop', default='')
ap.add_argument('--models', default=','.join(MODELS))
a = ap.parse_args()
OUT = 'out'
os.makedirs(OUT, exist_ok=True)
LOG = f'{OUT}/{a.name}.log'

df, arrival, booking = load(dedup=not a.raw)
groups = exact_groups()
fgroups = group_ids(df)
if a.global_enc:
    df = global_country(df)
y = df['is_canceled'].values
X = df.drop(columns=['is_canceled'])
drop = [c for c in a.drop.split(',') if c]
if a.group:
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, groups))
else:
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
log(f'{a.name}: n={len(y)} rate={y.mean():.3f} args={vars(a)}', LOG)

# share of test rows with an exact copy (all columns) in the training part
ov, ovf = [], []
for tr, te in folds:
    ov.append(np.isin(groups[te], np.unique(groups[tr])).mean())
    ovf.append(np.isin(fgroups[te], np.unique(fgroups[tr])).mean())
log(f'test rows with exact 32-column copy in training: {np.mean(ov):.3%}; identical model-input vector: {np.mean(ovf):.3%}', LOG)

splits = []
for k, (tr, te) in enumerate(folds):
    splits.append(prepare_split(X, y, tr, te, fold_encode=not a.global_enc, resample=a.resample, drop_cols=drop))
    log(f'fold {k+1}: train {len(splits[-1]["ytr"])} feat {splits[-1]["Xtr"].shape[1]}', LOG)

prev = pd.read_csv(f'{OUT}/{a.name}_folds.csv') if os.path.exists(f'{OUT}/{a.name}_folds.csv') else pd.DataFrame()
rows = prev.to_dict('records')
have = {(r['Model'], r['Fold']) for r in rows}
oof = {}
for m in a.models.split(','):
    oof[m] = np.full(len(y), np.nan)
    for k, S in enumerate(splits):
        if (m, k + 1) in have:
            continue
        t0 = time.time()
        pv, pt = fit_predict(m, S, seed_offset=k, class_weight=a.class_weight)
        oof[m][S['te_idx']] = pt
        r = {'Experiment': a.name, 'Model': m, 'Fold': k + 1, **metrics(S['yte'], pt),
             'Val ROC-AUC': roc_auc_score(S['yval'], pv), 'Seconds': time.time() - t0}
        rows.append(r)
        log(f"{m} fold {k+1}: F1 {r['F1']:.4f} AUC {r['ROC-AUC']:.4f} ({r['Seconds']:.0f}s)", LOG)
        pd.DataFrame(rows).to_csv(f'{OUT}/{a.name}_folds.csv', index=False)
        np.savez(f'{OUT}/{a.name}_oof.npz', y=y, **{k_.replace(' ', '_').replace('+', 'p'): v for k_, v in oof.items()})
log('DONE', LOG)
