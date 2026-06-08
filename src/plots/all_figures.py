"""
Publication-quality figures, 300 dpi, serif font.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats

import config

# --- rcParams ---
plt.rcParams.update({
    'font.family':          'serif',
    'font.size':            11,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'savefig.dpi':          300,
    'savefig.bbox':         'tight',
})

FIGS = Path('results/figures')
FIGS.mkdir(parents=True, exist_ok=True)

OPT_COLORS = {
    'GD_fixed':     '#1f77b4',
    'Adam':         '#ff7f0e',
    'AdamW':        '#2ca02c',
    'RMSProp':      '#d62728',
    'AdaGrad':      '#9467bd',
    'SGD_momentum': '#8c564b',
}


def _spearman(a, b):
    if len(a) < 3:
        return np.nan, np.nan
    r, p = scipy.stats.spearmanr(a, b)
    return float(r), float(p)


# ---------- Fig 1: SEE_quality heatmap ----------
def fig1_heatmap(df1: pd.DataFrame):
    fn_names = list(config.FUNCTIONS_2D_NAMES if hasattr(config, 'FUNCTIONS_2D_NAMES')
                    else df1['function'].unique())
    opt_names = list(config.OPTIMIZERS.keys())
    lrs = sorted(df1['lr'].unique())

    n_fns = len(fn_names)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for ax, fname in zip(axes, fn_names):
        sub = df1[df1['function'] == fname].groupby(['optimizer', 'lr'])['SEE_quality'].mean().unstack(level='lr')
        mat = sub.reindex(index=opt_names, columns=lrs).fillna(0).values
        im = ax.imshow(mat, aspect='auto', cmap='viridis', vmin=0)
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([str(lr) for lr in lrs], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(opt_names)))
        ax.set_yticklabels(opt_names, fontsize=8)
        ax.set_title(fname, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle('SEE_quality by Optimizer and Learning Rate', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig1_see_quality_heatmap.png')
    plt.close()
    print("Fig 1 saved.")


# ---------- Fig 2: SEE vs dimension ----------
def fig2_see_vs_dim(df2: pd.DataFrame):
    opt_names = list(config.OPTIMIZERS.keys())
    fn_names = ['Rastrigin-nD', 'Styblinski-nD', 'Synthetic_k=1', 'Synthetic_k=n//2']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, fname in zip(axes, fn_names):
        sub = df2[df2['function'] == fname]
        for opt in opt_names:
            s = sub[sub['optimizer'] == opt].sort_values('dimension')
            if s.empty:
                continue
            ax.plot(s['dimension'], s['SEE_basic'], marker='o',
                    color=OPT_COLORS.get(opt, None), label=opt)
            if 'CI_lo' in s.columns and 'CI_hi' in s.columns:
                ax.fill_between(s['dimension'], s['CI_lo'], s['CI_hi'],
                                alpha=0.15, color=OPT_COLORS.get(opt, None))
        ax.set_xscale('log')
        ax.set_xlabel('Dimension')
        ax.set_ylabel('SEE_basic')
        ax.set_title(fname)

    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle('SEE_basic vs Dimension', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig2_see_vs_dim.png')
    plt.close()
    print("Fig 2 saved.")


# ---------- Fig 3: Saddle index experiment ----------
def fig3_saddle_index(df2: pd.DataFrame):
    opt_names = list(config.OPTIMIZERS.keys())
    d_fixed = 50
    sub = df2[(df2['dimension'] == d_fixed) & (df2['function'].str.startswith('Synthetic'))]

    fig, ax = plt.subplots(figsize=(7, 5))
    for opt in opt_names:
        s = sub[sub['optimizer'] == opt].sort_values('saddle_index_k')
        if s.empty:
            continue
        ax.plot(s['saddle_index_k'], s['SEE_basic'], marker='o',
                color=OPT_COLORS.get(opt, None), label=opt)
    ax.set_xlabel('Saddle index k')
    ax.set_ylabel('SEE_basic')
    ax.set_title(f'SEE_basic vs Saddle Index (d={d_fixed})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig3_saddle_index.png')
    plt.close()
    print("Fig 3 saved.")


# ---------- Fig 4: NN training dynamics ----------
def fig4_nn_dynamics(df3: pd.DataFrame):
    tasks = df3['task'].unique()
    n_tasks = len(tasks)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4 * n_tasks))
    if n_tasks == 1:
        axes = axes.reshape(1, 2)

    for row_i, task in enumerate(tasks):
        sub = df3[df3['task'] == task]
        ax_loss = axes[row_i, 0]
        ax_grad = axes[row_i, 1]

        for opt in config.OPTIMIZERS:
            s = sub[sub['optimizer'] == opt]
            if s.empty:
                continue
            grp = s.groupby('step')['loss'].agg(['mean', 'std'])
            steps = grp.index.values
            mean_l = grp['mean'].values
            std_l  = grp['std'].values
            ax_loss.plot(steps, mean_l, label=opt, color=OPT_COLORS.get(opt))
            ax_loss.fill_between(steps, mean_l - std_l, mean_l + std_l, alpha=0.15,
                                  color=OPT_COLORS.get(opt))

            grp_g = s[s['grad_norm'].notna()].groupby('step')['grad_norm'].agg(['mean', 'std'])
            ax_grad.plot(grp_g.index, grp_g['mean'], label=opt, color=OPT_COLORS.get(opt))

        # Saddle event bands
        saddle_steps = sub[sub['is_saddle']]['step'].unique()
        for st in saddle_steps:
            ax_loss.axvspan(st - 12, st + 12, color='red', alpha=0.15)
            ax_grad.axvspan(st - 12, st + 12, color='red', alpha=0.15)

        ax_loss.set_title(f'{task} — Loss')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_grad.set_title(f'{task} — Grad Norm')
        ax_grad.set_xlabel('Step')
        ax_grad.set_ylabel('||∇||')

        if row_i == 0:
            ax_loss.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig4_nn_dynamics.png')
    plt.close()
    print("Fig 4 saved.")


# ---------- Fig 5: SEE_basic vs SEE_quality scatter ----------
def fig5_scatter(df1: pd.DataFrame):
    grp = df1.groupby(['function', 'optimizer', 'lr'])[['SEE_basic', 'SEE_quality']].mean()
    x = grp['SEE_basic'].values
    y = grp['SEE_quality'].values

    r, p = _spearman(x, y)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(x, y, alpha=0.5, s=20)
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y=x')
    ax.set_xlabel('SEE_basic')
    ax.set_ylabel('SEE_quality')
    ax.set_title('SEE_basic vs SEE_quality')
    ax.legend(title=f'Spearman r={r:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig5_see_scatter.png')
    plt.close()
    print("Fig 5 saved.")


# ---------- Fig 6: Sensitivity analysis ----------
def fig6_sensitivity(df4b: pd.DataFrame):
    fn_names = df4b['function'].unique()
    n = len(fn_names)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).ravel()

    for ax, fname in zip(axes, fn_names):
        sub = df4b[df4b['function'] == fname]
        for opt in config.OPTIMIZERS:
            s = sub[sub['optimizer'] == opt].sort_values('curvature_c')
            if s.empty:
                continue
            ax.plot(s['curvature_c'], s['SEE_quality'], marker='o',
                    label=opt, color=OPT_COLORS.get(opt))
        ax.set_xlabel('Curvature constant c')
        ax.set_ylabel('SEE_quality')
        ax.set_title(fname, fontsize=9)

    axes[0].legend(fontsize=7)
    for ax in axes[len(fn_names):]:
        ax.set_visible(False)
    fig.suptitle('Sensitivity Analysis: SEE_quality vs Curvature Constant', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig6_sensitivity.png')
    plt.close()
    print("Fig 6 saved.")


# ---------- Fig 7: Escape type breakdown ----------
def fig7_escape_breakdown(df1: pd.DataFrame):
    opt_names = list(config.OPTIMIZERS.keys())
    grp = df1.groupby('optimizer')[['escape_min_pct', 'escape_diverge_pct', 'stuck_pct']].mean()
    grp = grp.reindex(opt_names)
    grp_sorted = grp.sort_values('SEE_quality' if 'SEE_quality' in grp.columns
                                  else 'escape_min_pct', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(grp_sorted))
    min_vals  = grp_sorted['escape_min_pct'].values
    div_vals  = grp_sorted['escape_diverge_pct'].values
    stuck_vals = grp_sorted['stuck_pct'].values

    ax.barh(y_pos, min_vals,  color='#2ca02c', label='LOCAL_MIN')
    ax.barh(y_pos, div_vals,  left=min_vals, color='#ff7f0e', label='DIVERGE')
    ax.barh(y_pos, stuck_vals, left=min_vals + div_vals, color='#d62728', label='STUCK')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(grp_sorted.index)
    ax.set_xlabel('Percentage of trials (%)')
    ax.set_title('Escape Type Breakdown by Optimizer')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(FIGS / 'fig7_escape_breakdown.png')
    plt.close()
    print("Fig 7 saved.")


# ---------- Fig 8: Optimizer ranking heatmap ----------
def fig8_ranking_heatmap(df4d: pd.DataFrame):
    fn_names = df4d['fn1'].unique()
    mat = np.zeros((len(fn_names), len(fn_names)))
    fn_idx = {f: i for i, f in enumerate(fn_names)}

    for _, row in df4d.iterrows():
        i = fn_idx.get(row['fn1'])
        j = fn_idx.get(row['fn2'])
        if i is not None and j is not None:
            mat[i, j] = row['spearman_r']

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(fn_names)))
    ax.set_yticks(range(len(fn_names)))
    ax.set_xticklabels(fn_names, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(fn_names, fontsize=8)
    plt.colorbar(im, ax=ax, label='Spearman r')
    ax.set_title('Optimizer Ranking Consistency (Spearman r)')
    for i in range(len(fn_names)):
        for j in range(len(fn_names)):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=7)
    plt.tight_layout()
    plt.savefig(FIGS / 'fig8_ranking_heatmap.png')
    plt.close()
    print("Fig 8 saved.")


def make_all_figures(df1, df2, df3, df4b, df4d):
    print("\n=== Generating figures ===")
    fig1_heatmap(df1)
    fig2_see_vs_dim(df2)
    fig3_saddle_index(df2)
    fig4_nn_dynamics(df3)
    fig5_scatter(df1)
    fig6_sensitivity(df4b)
    fig7_escape_breakdown(df1)
    fig8_ranking_heatmap(df4d)
    print(f"All figures saved to {FIGS}/")
