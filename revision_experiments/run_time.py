# Expanding-window evaluation for the revision.
# usage: python run_time.py NAME --mode arrival|booking [--models ...]
#  arrival: train on arrivals before t, test on arrivals in [t, t+2 months); 9 windows Mar 2016 - Aug 2017
#  booking: train on arrivals before t (outcome known at t), test on bookings MADE in [t, t+2 months)
#           with lead time <= 120 days (so every test booking arrives by Aug 2017: no censoring); 7 windows
import sys, argparse, time
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from common import *

ap = argparse.ArgumentParser()
ap.add_argument('name')
ap.add_argument('--mode', default='arrival')
ap.add_argument('--models', default=','.join(MODELS))
a = ap.parse_args()
OUT = 'out'
LOG = f'{OUT}/{a.name}.log'

df, arrival, booking = load(dedup=True)
y = df['is_canceled'].values
X = df.drop(columns=['is_canceled', 'arrival_date_year'])
if a.mode == 'arrival':
    starts = pd.date_range('2016-03-01', '2017-07-01', freq='2MS')
else:
    starts = pd.date_range('2016-03-01', '2017-03-01', freq='2MS')
wins = []
for t in starts:
    e = t + pd.DateOffset(months=2)
    tr = np.where(arrival < t)[0]
    if a.mode == 'arrival':
        te = np.where((arrival >= t) & (arrival < e))[0]
    else:
        te = np.where((booking >= t) & (booking < e) & (df['lead_time'] <= 120))[0]
    wins.append((f'{t:%b %Y}', tr, te))
    log(f'{a.mode} window {t:%b %Y}: train {len(tr)} ({y[tr].mean():.3f}) test {len(te)} ({y[te].mean():.3f})', LOG)

pd.DataFrame([{'Window': w, 'Train': len(tr), 'Train rate': y[tr].mean(), 'Test': len(te), 'Test rate': y[te].mean()}
              for w, tr, te in wins]).to_csv(f'{OUT}/{a.name}_windows.csv', index=False)
rows, probs = [], {}
splits = [prepare_split(X, y, tr, te, val_mode='time', val_order=arrival.values) for _, tr, te in wins]
for m in a.models.split(','):
    for k, ((w, tr, te), S) in enumerate(zip(wins, splits)):
        t0 = time.time()
        pv, pt = fit_predict(m, S, seed_offset=k)
        probs[f'{m}|{w}'] = pt
        probs[f'y|{w}'] = S['yte']
        r = {'Model': m, 'Window': w, **metrics(S['yte'], pt), 'Val ROC-AUC': roc_auc_score(S['yval'], pv),
             'Seconds': time.time() - t0}
        rows.append(r)
        log(f"{m} {w}: F1 {r['F1']:.4f} AUC {r['ROC-AUC']:.4f} val {r['Val ROC-AUC']:.4f} ({r['Seconds']:.0f}s)", LOG)
        pd.DataFrame(rows).to_csv(f'{OUT}/{a.name}_results.csv', index=False)
        np.savez(f'{OUT}/{a.name}_probs.npz', **{k_.replace(' ', '_').replace('+', 'p'): v for k_, v in probs.items()})
log('DONE', LOG)
