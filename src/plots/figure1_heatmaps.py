"""Figure 1: SEE_quality heatmaps — optimizer × learning rate per function."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import config
from src.plots.style import apply_style, save_figure


def plot_figure1(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part1_results.csv')
        if not os.path.exists(path):
            print('part1_results.csv not found; skipping Figure 1.')
            return
        df = pd.read_csv(path)

    fn_names = config.FUNCTION_NAMES_2D
    opt_names = config.OPTIMIZER_NAMES
    lrs = config.LEARNING_RATES

    n_fns = len(fn_names)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for ax_idx, fn_name in enumerate(fn_names):
        ax = axes[ax_idx]
        sub = df[df['function'] == fn_name]

        # Build matrix: rows=optimizers, cols=lrs
        matrix = np.zeros((len(opt_names), len(lrs)))
        for i, opt in enumerate(opt_names):
            for j, lr in enumerate(lrs):
                sel = sub[(sub['optimizer'] == opt) & (sub['lr'] == lr)]['SEE_quality']
                matrix[i, j] = float(sel.mean()) if not sel.empty else 0.0

        im = ax.imshow(matrix, aspect='auto', cmap='viridis',
                       vmin=0, vmax=max(matrix.max(), 1e-9))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f'{lr}' for lr in lrs], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(opt_names)))
        ax.set_yticklabels(opt_names, fontsize=9)
        ax.set_title(fn_name)
        ax.set_xlabel('Learning Rate')
        if ax_idx % 3 == 0:
            ax.set_ylabel('Optimizer')

    fig.suptitle('SEE_quality Heatmaps (Optimizer × Learning Rate)', fontsize=13, y=1.01)
    save_figure(fig, os.path.join(out_dir, 'figure1_see_heatmaps.pdf'))
    save_figure(plt.figure(), os.path.join(out_dir, 'figure1_see_heatmaps.png'))


if __name__ == '__main__':
    plot_figure1()
