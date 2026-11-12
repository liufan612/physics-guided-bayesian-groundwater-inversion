from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'R1_1_revision_20260926'
OUT = REV / 'guide_diagnosis'
OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(OUT / 'runs.csv')
df = df[df.step == df.step.max()].copy()
metrics = [
    ('sd_mean', 'Mean spatial SD of ln K', 'ln K'),
    ('width95_mean', 'Mean pointwise 95% interval width', 'ln K'),
    ('latent_cov_trace', 'Trace of latent covariance', 'latent z'),
    ('latent_abs_corr_offdiag_mean', 'Mean absolute off-diagonal latent correlation', 'correlation'),
    ('logk_rmse', 'Posterior-mean ln K RMSE', 'ln K'),
    ('spread_error_r', 'Spread–absolute-error Pearson r', 'correlation'),
]
colors = {'diagonal': '#3b6fb6', 'fullrank': '#d55e00'}
fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.8), constrained_layout=True)
for ax, (col, title, ylabel) in zip(axs.flat, metrics):
    for gi, guide in enumerate(['diagonal', 'fullrank']):
        for hi, head_n in enumerate([0, 300]):
            sub = df[(df.guide == guide) & (df.head_n == head_n)][col].to_numpy()
            x = gi * 2 + hi
            jitter = np.linspace(-.11, .11, len(sub))
            ax.scatter(np.full(len(sub), x) + jitter, sub, s=24, alpha=.78,
                       color=colors[guide], edgecolor='white', linewidth=.35, zorder=3)
            ax.plot([x - .16, x + .16], [np.mean(sub), np.mean(sub)],
                    color='black', lw=2, zorder=4)
    ax.set_xticks([0, 1, 2, 3], ['Diag\n0 head', 'Diag\n300 head',
                                'Full-rank\n0 head', 'Full-rank\n300 head'])
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis='y', color='#d9d9d9', lw=.7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    if col == 'sd_mean': ax.set_ylim(0.078, 0.094)
    if col == 'width95_mean': ax.set_ylim(0.30, 0.37)
    if col == 'latent_cov_trace': ax.set_ylim(18, 26)
    if col == 'latent_abs_corr_offdiag_mean': ax.set_ylim(-.005, .08)
    if col == 'logk_rmse': ax.set_ylim(.78, 1.25)
    if col == 'spread_error_r': ax.set_ylim(-.02, .31)
fig.suptitle('Guide-diagnostic matched comparison (three retained synthetic fields; 3 seeds)', fontsize=13)
fig.savefig(OUT / 'FigureS8_r1_5_guide_diagnosis.png', dpi=300, bbox_inches='tight')
fig.savefig(OUT / 'FigureS8_r1_5_guide_diagnosis.pdf', bbox_inches='tight')
fig.savefig(OUT / 'FigureS8_r1_5_guide_diagnosis.svg', bbox_inches='tight')
df[['case_id', 'scenario', 'seed', 'guide', 'head_n'] + [m[0] for m in metrics]].to_csv(
    OUT / 'FigureS8_r1_5_guide_diagnosis_source.csv', index=False)
(OUT / 'FigureS8_r1_5_guide_diagnosis_caption.md').write_text(
    """# Figure S8

Guide-diagnostic matched comparison under the common IDW prior. Each point is one optimization seed for one retained evaluation field (three fields and three seeds per condition); black bars are condition means. (a) Mean spatial posterior SD of ln K. (b) Mean pointwise 95% interval width. (c) Trace of the 64-dimensional latent covariance. (d) Mean absolute off-diagonal latent correlation. (e) Posterior-mean ln K RMSE. (f) Pearson correlation between posterior spread and absolute ln K error. The diagonal guide has zero off-diagonal covariance by construction. The full-rank guide retains nonzero latent dependence and a covariance trace closer to the analytic K-only Gaussian reference, but both guide families show a similar approximately 3.5–3.7% contraction in spatial SD after adding 300 head observations. The guide choice therefore affects posterior-structure representation, whereas the small head-driven covariance contraction is not unique to the diagonal guide.""",
    encoding='utf-8')
print(OUT / 'FigureS8_r1_5_guide_diagnosis.png')
