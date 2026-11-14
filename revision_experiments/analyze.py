# Compile revision tables from the experiment outputs.
import sys, os, glob, json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, spearmanr, rankdata
from sklearn.metrics import roc_auc_score, f1_score
OUT = 'out'
R = {}
M6 = ['Logistic Regression', '1D CNN', 'DNN', 'CNN+DNN Hybrid', 'XGBoost', 'LightGBM']
KEY = lambda m: m.replace(' ', '_').replace('+', 'p')


def folds(*names):
    ds = [pd.read_csv(f'{OUT}/{n}_folds.csv') for n in names if os.path.exists(f'{OUT}/{n}_folds.csv')]
    return pd.concat(ds) if ds else pd.DataFrame()


def summ(d, cols=('Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC')):
    return d.groupby('Model')[list(cols)].agg(['mean', 'std'])


def oof(*names):
    out = {}
    for n in names:
        f = f'{OUT}/{n}_oof.npz'
        if os.path.exists(f):
            z = np.load(f)
            y = z['y']
            for m in M6:
                if KEY(m) in z.files and not np.isnan(z[KEY(m)]).any():
                    out[m] = z[KEY(m)]
    return y, out


def boot_auc(y, p, B=1000, seed=42):
    rs = np.random.RandomState(seed)
    n = len(y)
    v = []
    for _ in range(B):
        i = rs.randint(0, n, n)
        if y[i].min() == y[i].max(): continue
        v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def cost_eval(y, p, r, pi_train, policy):
    """expected cost per booking in units of one false alarm; missed cancellation costs r.
    policy: 'fixed' (0.5 on the SMOTE-balanced scale), 'bayes' (prior-corrected, threshold 1/(1+r)),
            'oracle' (best threshold on the test data itself; not attainable)"""
    if policy == 'fixed':
        flag = p >= 0.5
    elif policy == 'bayes':
        odds = p / np.clip(1 - p, 1e-9, None) * pi_train / (1 - pi_train)   # model trained at 50/50 after SMOTE
        q = odds / (1 + odds)
        flag = q >= 1 / (1 + r)
    else:
        best = None
        for t in np.unique(np.quantile(p, np.linspace(0, 1, 401))):
            c = cost_eval_flag(y, p >= t, r)
            best = c if best is None or c < best else best
        return best
    return cost_eval_flag(y, flag, r)


def cost_eval_flag(y, flag, r):
    fn = np.sum((y == 1) & ~flag)
    fp = np.sum((y == 0) & flag)
    return (r * fn + fp) / len(y)


def mean_table(d, models, cols=('ROC-AUC', 'F1', 'PR-AUC')):
    if not len(d): return {}
    if 'Fold' in d.columns:
        n = d.groupby(['Experiment', 'Model'])['Fold'].transform('count') if 'Experiment' in d.columns else d.groupby('Model')['Fold'].transform('count')
        d = d[n == 5]
        if not len(d): return {}
    g = d.groupby('Model')[list(cols)]
    m, s = g.mean(), g.std()
    return {mm: {c: (float(m.loc[mm, c]), float(s.loc[mm, c])) for c in cols} for mm in models if mm in m.index}


if __name__ == '__main__':
    R['base'] = mean_table(folds('base_gb', 'base_dnn', 'base_cnnh'), M6, ('Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC'))
    R['globalenc'] = mean_table(folds('globalenc_gb', 'globalenc_dnn'), M6)
    R['smotenc'] = mean_table(folds('smotenc_gb', 'smotenc_dnn'), M6, ('ROC-AUC', 'F1', 'PR-AUC', 'Recall', 'Precision'))
    R['cw'] = mean_table(folds('cw_gb', 'cw_dnn'), M6, ('ROC-AUC', 'F1', 'PR-AUC', 'Recall', 'Precision'))
    R['noparking'] = mean_table(folds('noparking_gb', 'noparking_dnn', 'noparking_cnnh'), M6)
    R['postbook'] = mean_table(folds('postbook_g4'), M6)
    R['rawrandom'] = mean_table(folds('rawrandom_gb', 'rawrandom_dnn', 'rawrandom_cnnh'), M6, ('Accuracy', 'F1', 'ROC-AUC', 'PR-AUC'))
    R['rawgroup'] = mean_table(folds('rawgroup_gb', 'rawgroup_dnn', 'rawgroup_cnnh'), M6, ('Accuracy', 'F1', 'ROC-AUC', 'PR-AUC'))
    e = folds('eqbudget')
    if len(e):
        R['eqbudget'] = {f'{m}|{b}': (float(g['ROC-AUC'].mean()), float(g['F1'].mean())) for (m, b), g in e.groupby(['Model', 'Budget'])}
        R['eqbudget'].update({'XGBoost|15': (0.9041, 0.6977), 'LightGBM|15': (0.9057, 0.7004)})   # original Table V
    for n, lab in [('ftt', 'FT-Transformer'), ('tabnet', 'TabNet')]:
        d = folds(n)
        if len(d) == 5:
            R[n] = {c: (float(d[c].mean()), float(d[c].std())) for c in ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']}

    # ---------------- expanding windows (9 x 2 months, arrival) ----------------
    t = pd.concat([pd.read_csv(f) for f in glob.glob(f'{OUT}/timearr_*_results.csv')])
    probs = {}
    for f in glob.glob(f'{OUT}/timearr_*_probs.npz'):
        z = np.load(f)
        probs.update({k: z[k] for k in z.files})
    wins = list(pd.read_csv(f'{OUT}/timearr_gb_windows.csv')['Window'])
    models_t = [m for m in M6 if (t.Model == m).sum() == len(wins)]
    A = t.pivot(index='Window', columns='Model', values='ROC-AUC').loc[wins, models_t]
    F = t.pivot(index='Window', columns='Model', values='F1').loc[wins, models_t]
    PR = t.pivot(index='Window', columns='Model', values='PR-AUC').loc[wins, models_t]
    tw = {}
    for m in models_t:
        tw[m] = {'auc_mean': float(A[m].mean()), 'auc_sd': float(A[m].std()), 'auc_min': float(A[m].min()),
                 'auc_max': float(A[m].max()), 'f1_mean': float(F[m].mean()), 'f1_sd': float(F[m].std()),
                 'pr_mean': float(PR[m].mean()),
                 'rank_auc': float(A.rank(axis=1, ascending=False)[m].mean()),
                 'best_auc': int((A.idxmax(axis=1) == m).sum())}
    R['timearr'] = tw
    R['timearr_models'] = models_t
    if len(models_t) >= 3:
        for name, M in [('auc', A), ('f1', F)]:
            st, p = friedmanchisquare(*[M[m] for m in models_t])
            R[f'friedman_{name}'] = (float(st), float(p), float(st / (len(wins) * (len(models_t) - 1))))
        three = [m for m in ['Logistic Regression', 'XGBoost', 'LightGBM'] if m in models_t]
        st3, p3 = friedmanchisquare(*[A[m] for m in three])
        R['friedman_auc_3'] = (float(st3), float(p3), float(st3 / (len(wins) * (len(three) - 1))))
        # rank stability: Spearman between rankings of consecutive windows and between CV ranking and each window
        cv = pd.Series({m: R['base'][m]['ROC-AUC'][0] for m in models_t if m in R['base']})
        rho_cv = [spearmanr(cv[models_t], A.loc[w, models_t]).correlation for w in wins] if len(cv) == len(models_t) else []
        rho_seq = [spearmanr(A.loc[a, models_t], A.loc[b, models_t]).correlation for a, b in zip(wins[:-1], wins[1:])]
        R['rho_cv'] = rho_cv
        R['rho_seq'] = rho_seq
    # bootstrap CIs per window
    ci = {}
    for w in wins:
        y = probs[f'y|{w}'.replace(' ', '_')]
        for m in models_t:
            lo, hi = boot_auc(y, probs[f'{m}|{w}'.replace(' ', '_').replace('+', 'p')], B=500)
            ci[f'{m}|{w}'] = (float(A.loc[w, m]), float(lo), float(hi))
    R['ci'] = ci
    R['ci_halfwidth_mean'] = float(np.mean([(v[2] - v[1]) / 2 for v in ci.values()]))
    A.to_csv(f'{OUT}/T_timearr_auc.csv'); F.to_csv(f'{OUT}/T_timearr_f1.csv')
    win_tab = pd.read_csv(f'{OUT}/timearr_gb_windows.csv')
    R['windows'] = win_tab.to_dict('records')

    # ---------------- booking-date windows ----------------
    if os.path.exists(f'{OUT}/timebook_g4_results.csv'):
        b = pd.read_csv(f'{OUT}/timebook_g4_results.csv')
        R['timebook'] = {m: {'auc_mean': float(g['ROC-AUC'].mean()), 'auc_sd': float(g['ROC-AUC'].std()),
                             'f1_mean': float(g['F1'].mean()), 'n': len(g)} for m, g in b.groupby('Model')}
        R['timebook_windows'] = pd.read_csv(f'{OUT}/timebook_g4_windows.csv').to_dict('records')

    # ---------------- cost analysis ----------------
    rs_ = [0.5, 1, 2, 4]
    cost = {}
    y, P = oof('base_gb', 'base_dnn', 'base_cnnh')
    R['cv_prev'] = float(y.mean())
    pi = y.mean()
    for r in rs_:
        cost[f'cv|trivial|{r}'] = float(min(r * pi, 1 - pi))
        for m, p in P.items():
            for pol in ['fixed', 'bayes', 'oracle']:
                cost[f'cv|{m}|{pol}|{r}'] = float(cost_eval(y, p, r, pi, pol))
    # time windows: cost summed over the nine windows
    for r in rs_:
        tot_triv, n_tot = 0, 0
        acc = {}
        for w, row in zip(wins, R['windows']):
            yw = probs[f'y|{w}'.replace(' ', '_')]
            tot_triv += min(r * yw.sum(), len(yw) - yw.sum()); n_tot += len(yw)
            for m in models_t:
                pw = probs[f'{m}|{w}'.replace(' ', '_').replace('+', 'p')]
                for pol in ['fixed', 'bayes', 'oracle']:
                    acc[(m, pol)] = acc.get((m, pol), 0) + cost_eval(yw, pw, r, row['Train rate'], pol) * len(yw)
        cost[f'time|trivial|{r}'] = float(tot_triv / n_tot)
        for (m, pol), v in acc.items():
            cost[f'time|{m}|{pol}|{r}'] = float(v / n_tot)
    R['cost'] = cost
    json.dump(R, open(f'{OUT}/results.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in R.items() if k not in ('ci', 'cost')}, indent=1, default=float)[:6000])
