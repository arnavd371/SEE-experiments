"""Figure 3: SEE_basic vs saddle index k at d=50 for synthetic saddle."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.plots.style import apply_style, save_figure, OPTIMIZER_COLORS


def plot_figure3(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part2_results.csv')
        if not os.path.exists(path):
            print('part2_results.csv not found; skipping Figure 3.')
            return
        df = pd.read_csv(path)

    target_d = 50
    sub = df[(df['function'] == 'Synthetic-Saddle') & (df['dimension'] == target_d)]

    if sub.empty:
        print(f'No Synthetic-Saddle data at d={target_d}; skipping Figure 3.')
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    for opt in config.OPTIMIZER_NAMES:
        opt_sub = sub[sub['optimizer'] == opt].copy()
        opt_sub['saddle_index_k'] = pd.to_numeric(
            opt_sub['saddle_index_k'], errors='coerce')
        opt_sub = opt_sub.dropna(subset=['saddle_index_k']).sort_values('saddle_index_k')
        if opt_sub.empty:
            continue
        color = OPTIMIZER_COLORS.get(opt, '#333333')
        ax.plot(opt_sub['saddle_index_k'], opt_sub['SEE_basic'],
                marker='o', label=opt, color=color, linewidth=1.5)

    ax.set_xlabel('Saddle Index k (# negative curvature directions)')
    ax.set_ylabel('SEE_basic')
    ax.set_title(f'SEE vs Saddle Index (Synthetic Saddle, d={target_d})')
    ax.legend(fontsize=9, ncol=2)

    save_figure(fig, os.path.join(out_dir, 'figure3_saddle_index.pdf'))


if __name__ == '__main__':
    plot_figure3()
