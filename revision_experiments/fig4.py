import json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = json.load(open('out/results.json'))
wins = [w['Window'] for w in R['windows']]
models = R['timearr_models']
lab = {'Logistic Regression': 'Logistic regression', 'CNN+DNN Hybrid': 'CNN+DNN hybrid'}
style = {'Logistic Regression': ('#1f77b4', 'o', '-'), '1D CNN': ('#ff7f0e', 's', '--'), 'DNN': ('#2ca02c', '^', '--'),
         'CNN+DNN Hybrid': ('#d62728', 'v', '--'), 'XGBoost': ('#9467bd', 'D', '-'), 'LightGBM': ('#8c564b', 'P', '-')}
plt.rcParams.update({'font.size': 13, 'font.family': 'DejaVu Sans'})
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={'height_ratios': [3, 1.2]})
x = np.arange(len(wins))
off = np.linspace(-0.25, 0.25, len(models))
for k, m in enumerate(models):
    c, mk, ls = style[m]
    v = np.array([R['ci'][f'{m}|{w}'] for w in wins])
    ax.errorbar(x + off[k], v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color=c, marker=mk, ls=ls, lw=1.6,
                ms=6, capsize=2.5, label=lab.get(m, m))
ax.set_ylabel('ROC-AUC on test window\n(95% bootstrap CI)')
ax.grid(alpha=0.3)
ax.legend(ncol=4, fontsize=11, loc='lower center', bbox_to_anchor=(0.5, 1.0), frameon=False)
rates = [w['Test rate'] * 100 for w in R['windows']]
tr = [w['Train rate'] * 100 for w in R['windows']]
ax2.bar(x - 0.18, tr, 0.36, color='#bbbbbb', label='Training data')
ax2.bar(x + 0.18, rates, 0.36, color='#555555', label='Test window')
ax2.set_ylabel('Cancellation\nrate (%)')
ax2.set_ylim(0, 45)
ax2.legend(ncol=2, fontsize=10, loc='upper left')
ax2.grid(axis='y', alpha=0.3)
ax2.set_xticks(x)
ax2.set_xticklabels([w.replace(' 20', '\n20') for w in wins])
ax2.set_xlabel('First month of the two-month test window (arrival date)')
plt.tight_layout()
plt.savefig('fig4.png', dpi=300)
print('ok')
