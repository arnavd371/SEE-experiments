"""Figure 5: SEE predicts training efficiency (MAIN FIGURE).

Left:  Spearman correlation — SEE_quality vs steps_to_ppl (Part 4)
Right: SEE_quality vs steps_to_convergence (Part 3), one subplot per task
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

import config
from src.plots.style import apply_style, save_figure, OPTIMIZER_COLORS


def plot_figure5(df5: pd.DataFrame = None, df3: pd.DataFrame = None,
                 df4: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    def _load(path, name):
        if not os.path.exists(path):
            print(f'{name} not found; skipping Figure 5.')
            return None
        return pd.read_csv(path)

    if df5 is None:
        df5 = _load(os.path.join(config.RESULTS_DIR, 'part5_results.csv'), 'part5_results.csv')
    if df3 is None:
        df3 = _load(os.path.join(config.RESULTS_DIR, 'part3_results.csv'), 'part3_results.csv')
    if df4 is None:
        df4 = _load(os.path.join(config.RESULTS_DIR, 'part4_results.csv'), 'part4_results.csv')
    if any(d is None for d in [df5, df3, df4]):
        return

    task_names = df3['task'].unique().tolist() if df3 is not None else []
    n_right = max(len(task_names), 1)
    fig, axes = plt.subplots(1, 1 + n_right, figsize=(5 * (1 + n_right), 4.5))
    if not hasattr(axes, '__len__'):
        axes = [axes]

    # ── Left panel: SEE_quality vs steps_to_ppl ───────────────────────────────
    ax_left = axes[0]
    see_vals, ppl_steps, opt_labels = [], [], []

    for _, row in df5.iterrows():
        opt = row['optimizer']
        sv = row['mean_SEE_quality']
        pv = row['steps_to_ppl_threshold']
        if math.isfinite(float(sv)) and math.isfinite(float(pv)):
            see_vals.append(float(sv))
            ppl_steps.append(float(pv))
            opt_labels.append(opt)
            color = OPTIMIZER_COLORS.get(opt, '#333333')
            ax_left.scatter(sv, pv, color=color, s=80, zorder=5)
            ax_left.annotate(opt, (sv, pv), textcoords='offset points',
                             xytext=(4, 4), fontsize=8)

    if len(see_vals) >= 3:
        # Regression line
        m, b = np.polyfit(see_vals, ppl_steps, 1)
        xs = np.linspace(min(see_vals), max(see_vals), 100)
        ax_left.plot(xs, m * xs + b, 'k--', linewidth=1, alpha=0.6)

        # Annotation
        r_sp = float(df5['spearman_r_SEE_ppl'].iloc[0])
        p_sp = float(df5['spearman_p_SEE_ppl'].iloc[0])
        ax_left.text(0.05, 0.95,
                     f'Spearman r={r_sp:.2f}\np={p_sp:.3f}',
                     transform=ax_left.transAxes, fontsize=9,
                     va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))

    ax_left.set_xlabel('SEE_quality (Part 1, best LR)')
    ax_left.set_ylabel('Steps to PPL < 100 (Part 4)')
    ax_left.set_title('SEE Predicts LLM Training Efficiency')

    # ── Right panels: SEE_quality vs steps_to_convergence ──────────────────────
    p3_last = df3.groupby(['task', 'optimizer', 'run_seed']).last().reset_index()
    p3_agg = p3_last.groupby(['task', 'optimizer'])[
        'steps_to_convergence'].mean().reset_index()

    see_by_opt = df5.set_index('optimizer')['mean_SEE_quality'].to_dict()

    for ax_idx, task_name in enumerate(task_names):
        ax = axes[1 + ax_idx]
        sub3 = p3_agg[p3_agg['task'] == task_name]

        x_vals, y_vals = [], []
        for _, row in sub3.iterrows():
            opt = row['optimizer']
            sv = float(see_by_opt.get(opt, float('nan')))
            cv = float(row['steps_to_convergence'])
            if math.isfinite(sv) and math.isfinite(cv):
                x_vals.append(sv)
                y_vals.append(cv)
                color = OPTIMIZER_COLORS.get(opt, '#333333')
                ax.scatter(sv, cv, color=color, s=80, zorder=5)
                ax.annotate(opt, (sv, cv), textcoords='offset points',
                            xytext=(4, 4), fontsize=8)

        if len(x_vals) >= 3:
            m, b = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(min(x_vals), max(x_vals), 100)
            ax.plot(xs, m * xs + b, 'k--', linewidth=1, alpha=0.6)
            try:
                r5b, p5b = scipy.stats.spearmanr(x_vals, y_vals)
                ax.text(0.05, 0.95,
                        f'Spearman r={r5b:.2f}\np={p5b:.3f}',
                        transform=ax.transAxes, fontsize=9,
                        va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
            except Exception:
                pass

        ax.set_xlabel('SEE_quality (Part 1)')
        ax.set_ylabel('Steps to Convergence')
        ax.set_title(f'SEE vs NN Convergence\n({task_name})')

    fig.suptitle('Figure 5 — SEE Predicts Training Efficiency', fontsize=13)
    save_figure(fig, os.path.join(out_dir, 'figure5_main_result.pdf'))


if __name__ == '__main__':
    plot_figure5()
