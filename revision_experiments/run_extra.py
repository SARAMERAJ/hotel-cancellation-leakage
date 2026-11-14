# Equal-budget boosting (first 6 of the original 15 random-search configurations, chosen per fold on the
# inner validation set) and TabNet, under the base pipeline (dedup, fold-fitted encoding, SMOTE).
import sys, time, argparse
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from common import *
from sklearn.model_selection import StratifiedKFold
ap = argparse.ArgumentParser(); ap.add_argument('what'); a = ap.parse_args()
OUT = 'out'; LOG = f'{OUT}/{a.what}.log'
df, arrival, booking = load(True); y = df['is_canceled'].values; X = df.drop(columns=['is_canceled'])
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
splits = [prepare_split(X, y, tr, te) for tr, te in folds]
rows = []
if a.what == 'eqbudget':
    space = {'XGBoost': pd.read_csv('../results_A/search_space_xgboost.csv'),
             'LightGBM': pd.read_csv('../results_A/search_space_lightgbm.csv')}
    for m, sp in space.items():
        for budget in [6]:
            cfgs = sp.drop(columns='config').head(budget).to_dict('records')
            for k, S in enumerate(splits):
                best = (-1, None)
                for c in cfgs:
                    c = {kk: (int(v) if kk in ('max_depth', 'min_child_weight', 'num_leaves', 'min_child_samples') else float(v)) for kk, v in c.items()}
                    pv, pt = fit_predict(m, S, cfg=c)
                    va = roc_auc_score(S['yval'], pv)
                    if va > best[0]: best = (va, pt)
                rows.append({'Model': m, 'Budget': budget, 'Fold': k + 1, **metrics(S['yte'], best[1])})
                log(f"{m} budget {budget} fold {k+1}: AUC {rows[-1]['ROC-AUC']:.4f} F1 {rows[-1]['F1']:.4f}", LOG)
                pd.DataFrame(rows).to_csv(f'{OUT}/{a.what}_folds.csv', index=False)
elif a.what == 'ftt':
    oof = np.full(len(y), np.nan)
    if os.path.exists(f'{OUT}/ftt_folds.csv'):
        rows = pd.read_csv(f'{OUT}/ftt_folds.csv').to_dict('records')
    done_f = {r['Fold'] for r in rows}
    for k, S in enumerate(splits):
        if k + 1 in done_f:
            continue
        t0 = time.time()
        pv, pt = fit_ft(S, seed_offset=k)  # small FT-Transformer, see common.fit_ft
        oof[S['te_idx']] = pt
        rows.append({'Model': 'FT-Transformer', 'Fold': k + 1, **metrics(S['yte'], pt), 'Val ROC-AUC': roc_auc_score(S['yval'], pv), 'Seconds': time.time() - t0})
        log(f"FT fold {k+1}: AUC {rows[-1]['ROC-AUC']:.4f} F1 {rows[-1]['F1']:.4f} ({rows[-1]['Seconds']:.0f}s)", LOG)
        pd.DataFrame(rows).to_csv(f'{OUT}/{a.what}_folds.csv', index=False)
        np.savez(f'{OUT}/ftt_oof.npz', y=y, FTT=oof)
else:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.set_num_threads(1)
    cfgs = [dict(n_d=16, n_a=16, n_steps=3, gamma=1.3, lr=0.02), dict(n_d=32, n_a=32, n_steps=5, gamma=1.5, lr=0.02)]
    oof = np.full(len(y), np.nan)
    for k, S in enumerate(splits):
        best = (-1, None, None)
        for ci, c in enumerate(cfgs):
            t0 = time.time()
            clf = TabNetClassifier(n_d=c['n_d'], n_a=c['n_a'], n_steps=c['n_steps'], gamma=c['gamma'],
                                   optimizer_params=dict(lr=c['lr']), seed=SEED + k, verbose=0,
                                   scheduler_fn=torch.optim.lr_scheduler.StepLR, scheduler_params=dict(step_size=10, gamma=0.9))
            clf.fit(S['Xtr'], S['ytr'], eval_set=[(S['Xval'], S['yval'])], eval_metric=['auc'], max_epochs=50,
                    patience=7, batch_size=1024, virtual_batch_size=128)
            va = roc_auc_score(S['yval'], clf.predict_proba(S['Xval'])[:, 1])
            log(f'TabNet fold {k+1} cfg {ci}: val AUC {va:.4f} ({time.time()-t0:.0f}s)', LOG)
            if va > best[0]: best = (va, clf.predict_proba(S['Xte'])[:, 1], ci)
        oof[S['te_idx']] = best[1]
        rows.append({'Model': 'TabNet', 'Fold': k + 1, 'Chosen config': best[2], **metrics(S['yte'], best[1])})
        log(f"TabNet fold {k+1}: AUC {rows[-1]['ROC-AUC']:.4f} F1 {rows[-1]['F1']:.4f}", LOG)
        pd.DataFrame(rows).to_csv(f'{OUT}/{a.what}_folds.csv', index=False)
        np.savez(f'{OUT}/tabnet_oof.npz', y=y, TabNet=oof)
log('DONE', LOG)
