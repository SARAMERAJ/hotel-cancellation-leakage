# Drift diagnostics for the three original expanding windows (W1-W3) and the parking/time check.
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from common import *
from sklearn.model_selection import StratifiedKFold, cross_val_predict
OUT = 'out'

df, arrival, booking = load(dedup=True)
y = df['is_canceled'].values
W = {'W1': ('2016-07-01', '2017-01-01'), 'W2': ('2017-01-01', '2017-05-01'), 'W3': ('2017-05-01', '2017-09-01')}
date_cols = ['arrival_date_year', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month']
feat_num = ['lead_time', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies',
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'days_in_waiting_list',
            'adr', 'required_car_parking_spaces', 'total_of_special_requests', 'total_guests', 'total_nights']
feat_cat = [c for c in CAT if c != 'arrival_date_month']


def psi(ref, cur, numeric):
    if numeric:
        q = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
        if len(q) < 3:
            bins_r = pd.Series(ref).value_counts(normalize=True)
            bins_c = pd.Series(cur).value_counts(normalize=True)
        else:
            e = np.concatenate([[-np.inf], q[1:-1], [np.inf]])
            bins_r = pd.Series(pd.cut(ref, e)).value_counts(normalize=True)
            bins_c = pd.Series(pd.cut(cur, e)).value_counts(normalize=True)
    else:
        bins_r = pd.Series(ref).value_counts(normalize=True)
        bins_c = pd.Series(cur).value_counts(normalize=True)
    idx = bins_r.index.union(bins_c.index)
    r = bins_r.reindex(idx, fill_value=0).values + 1e-4
    c = bins_c.reindex(idx, fill_value=0).values + 1e-4
    return float(np.sum((c - r) * np.log(c / r)))


def uni_auc(x, yy, numeric):
    """univariate discrimination of one feature (target-encoded for categoricals), as AUC >= 0.5"""
    if numeric:
        s = pd.Series(x).astype(float).values
    else:
        s = pd.Series(x).map(pd.Series(yy).groupby(pd.Series(x).values).mean()).values
    a = roc_auc_score(yy, s)
    return max(a, 1 - a)


rows, rel_rows, adv_rows = [], [], []
for w, (s, e) in W.items():
    tr = (arrival < s).values
    te = ((arrival >= s) & (arrival < e)).values
    for f in feat_num + feat_cat:
        num = f in feat_num
        rows.append({'Window': w, 'Feature': f, 'PSI': psi(df.loc[tr, f].values, df.loc[te, f].values, num),
                     'Univariate AUC train': uni_auc(df.loc[tr, f].values, y[tr], num),
                     'Univariate AUC test (train encoding)': np.nan})
    # predictor-target relationship: cancellation rate by group, train vs test period
    groups = {
        'Lead time 0-7': df.lead_time <= 7, 'Lead time 8-30': df.lead_time.between(8, 30),
        'Lead time 31-90': df.lead_time.between(31, 90), 'Lead time 91-180': df.lead_time.between(91, 180),
        'Lead time >180': df.lead_time > 180,
        'Portugal': df.country == 'PRT', 'Other countries': df.country != 'PRT',
        'Online TA': df.market_segment == 'Online TA', 'Offline TA/TO': df.market_segment == 'Offline TA/TO',
        'Direct': df.market_segment == 'Direct', 'Groups': df.market_segment == 'Groups',
        '0 special requests': df.total_of_special_requests == 0, '1+ special requests': df.total_of_special_requests >= 1,
        'Non-refundable': df.deposit_type == 'Non Refund', 'City hotel': df.hotel == 'City Hotel',
        'Resort hotel': df.hotel == 'Resort Hotel', 'Transient': df.customer_type == 'Transient',
    }
    for g, mk in groups.items():
        mk = mk.values
        a_, b_ = mk & tr, mk & te
        rel_rows.append({'Window': w, 'Group': g, 'Share train': a_.sum() / tr.sum(), 'Share test': b_.sum() / te.sum(),
                         'Cancel rate train': y[a_].mean(), 'Cancel rate test': y[b_].mean(),
                         'n test': int(b_.sum())})
    # adversarial validation: can a classifier tell training-period from test-period bookings?
    Xa = df.drop(columns=['is_canceled'] + date_cols)
    sel = tr | te
    Xa = Xa[sel].reset_index(drop=True)
    ya = te[sel].astype(int)
    enc = Encoder(cat=[c for c in CAT if c in Xa.columns]).fit(Xa)
    E = enc.transform(Xa).values
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, random_state=SEED, n_jobs=2, verbose=-1)
    p = cross_val_predict(clf, E, ya, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), method='predict_proba')[:, 1]
    clf.fit(E, ya)
    imp = pd.Series(clf.booster_.feature_importance('gain'), index=enc.transform(Xa.head(2)).columns).sort_values(ascending=False)
    # same with the label included: shift in the joint distribution beyond the covariates
    adv_rows.append({'Window': w, 'Adversarial AUC': roc_auc_score(ya, p), 'Top features': '; '.join(imp.index[:5])})
    # concept-shift check: P(y|x) learned on the past vs. learned within the test period (5-fold CV inside it)
    Xm = df.drop(columns=['is_canceled', 'arrival_date_year']).reset_index(drop=True)
    tr_i, te_i = np.where(tr)[0], np.where(te)[0]
    S = prepare_split(Xm, y, tr_i, te_i, val_mode='time', val_order=arrival.values, resample='none')
    _, p_past = fit_predict('LightGBM', S)
    p_in = np.full(len(te_i), np.nan)
    for k, (a2, b2) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(te_i, y[te_i])):
        S2 = prepare_split(Xm, y, te_i[a2], te_i[b2], resample='none')
        _, p_in[b2] = fit_predict('LightGBM', S2)
    # past model applied with the intercept shifted to the test prevalence (prior-shift correction)
    adv_rows[-1].update({'LightGBM AUC trained on past': roc_auc_score(y[te_i], p_past),
                         'LightGBM AUC trained within test period (5-fold)': roc_auc_score(y[te_i], p_in),
                         'Mean predicted prob (past model)': p_past.mean(), 'Test rate': y[te_i].mean(),
                         'Train rate': y[tr_i].mean()})
    print(adv_rows[-1], flush=True)

pd.DataFrame(rows).to_csv(f'{OUT}/drift_psi.csv', index=False)
pd.DataFrame(rel_rows).to_csv(f'{OUT}/drift_relationships.csv', index=False)
pd.DataFrame(adv_rows).to_csv(f'{OUT}/drift_adversarial.csv', index=False)

# ---- time-dependence of request fields (uses reservation_status_date for diagnosis only) ----
raw = pd.read_csv(DATA).drop_duplicates().reset_index(drop=True)
raw = raw[(raw.adults + raw.children.fillna(0) + raw.babies) > 0].reset_index(drop=True)
arr = pd.to_datetime(raw.arrival_date_year.astype(str) + '-' + raw.arrival_date_month + '-' +
                     raw.arrival_date_day_of_month.astype(str), format='%Y-%B-%d')
bk = arr - pd.to_timedelta(raw.lead_time, unit='D')
status = pd.to_datetime(raw.reservation_status_date)
c = raw.is_canceled == 1
days_in_system = (status - bk).dt.days.where(c, raw.lead_time)   # cancelled: until cancellation; kept: until arrival
frac = (days_in_system / raw.lead_time.replace(0, np.nan)).clip(0, 1)
t_rows = []
for lab, mk in [('Kept (state as of day before arrival)', ~c)] + [
        (f'Cancelled, {lo}-{hi} days before arrival', c & ((arr - status).dt.days.between(lo, hi)))
        for lo, hi in [(0, 1), (2, 7), (8, 30), (31, 90), (91, 10000)]]:
    t_rows.append({'Group': lab, 'n': int(mk.sum()),
                   'Share with parking request': (raw.loc[mk, 'required_car_parking_spaces'] > 0).mean(),
                   'Share with 1+ special request': (raw.loc[mk, 'total_of_special_requests'] > 0).mean(),
                   'Mean special requests': raw.loc[mk, 'total_of_special_requests'].mean(),
                   'Mean ADR': raw.loc[mk, 'adr'].mean()})
pd.DataFrame(t_rows).to_csv(f'{OUT}/request_timing.csv', index=False)
print(pd.DataFrame(t_rows).round(3).to_string())
# cancelled bookings: requests vs time already spent in the system, same lead-time band
cc = raw[c].copy()
cc['days_in_system'] = days_in_system[c]
band = cc[cc.lead_time.between(91, 180)]
print('Cancelled, lead 91-180: special-request share by days in system quartile')
print(band.groupby(pd.qcut(band.days_in_system, 4))['total_of_special_requests'].apply(lambda s: (s > 0).mean()).round(3))
