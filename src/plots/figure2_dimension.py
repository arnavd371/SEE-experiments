"""Figure 2: SEE_basic vs dimension (log scale), one line per optimizer."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.plots.style import apply_style, save_figure, OPTIMIZER_COLORS


def plot_figure2(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part2_results.csv')
        if not os.path.exists(path):
            print('part2_results.csv not found; skipping Figure 2.')
            return
        df = pd.read_csv(path)

    fn_names = [f for f in config.ND_FUNCTION_NAMES if f != 'Synthetic-Saddle']
    fig, axes = plt.subplots(1, len(fn_names), figsize=(5 * len(fn_names), 4))
    if len(fn_names) == 1:
        axes = [axes]

    for ax, fn_name in zip(axes, fn_names):
        sub = df[df['function'] == fn_name]
        for opt in config.OPTIMIZER_NAMES:
            opt_sub = sub[sub['optimizer'] == opt].sort_values('dimension')
            if opt_sub.empty:
                continue
            dims = opt_sub['dimension'].values
            see_vals = opt_sub['SEE_basic'].values
            ci_lo = opt_sub['SEE_basic_CI_lo'].values
            ci_hi = opt_sub['SEE_basic_CI_hi'].values

            color = OPTIMIZER_COLORS.get(opt, '#333333')
            ax.plot(dims, see_vals, marker='o', label=opt, color=color, linewidth=1.5)
            ax.fill_between(dims, ci_lo, ci_hi, alpha=0.15, color=color)

        ax.set_xscale('log')
        ax.set_xlabel('Dimension (log scale)')
        ax.set_ylabel('SEE_basic')
        ax.set_title(fn_name)
        ax.set_xticks([2, 10, 50, 100, 500])
        ax.set_xticklabels(['2', '10', '50', '100', '500'])
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('SEE_basic vs Dimension — High-Dimensional Scaling', fontsize=13)
    save_figure(fig, os.path.join(out_dir, 'figure2_dimension_scaling.pdf'))


if __name__ == '__main__':
    plot_figure2()
