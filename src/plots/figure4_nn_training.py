"""Figure 4: NN training loss curves + grad norm (3 tasks × 2 columns)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import config
from src.plots.style import apply_style, save_figure, OPTIMIZER_COLORS


def plot_figure4(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part3_results.csv')
        if not os.path.exists(path):
            print('part3_results.csv not found; skipping Figure 4.')
            return
        df = pd.read_csv(path)

    task_names = df['task'].unique().tolist()
    n_tasks = len(task_names)
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 4 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for row_idx, task_name in enumerate(task_names):
        ax_loss = axes[row_idx, 0]
        ax_grad = axes[row_idx, 1]

        task_df = df[df['task'] == task_name]
        saddle_steps = task_df[task_df['is_saddle_step'] == 1]['step'].unique()

        for opt in config.OPTIMIZER_NAMES:
            opt_df = task_df[task_df['optimizer'] == opt]
            if opt_df.empty:
                continue
            color = OPTIMIZER_COLORS.get(opt, '#333333')

            # Mean ± std over runs
            grp = opt_df.groupby('step')['loss'].agg(['mean', 'std'])
            steps = grp.index.values
            mean_loss = grp['mean'].values
            std_loss = grp['std'].fillna(0).values

            ax_loss.plot(steps, mean_loss, color=color, label=opt, linewidth=1.2)
            ax_loss.fill_between(steps, mean_loss - std_loss, mean_loss + std_loss,
                                 alpha=0.15, color=color)

            # Grad norm
            grp_g = opt_df.groupby('step')['grad_norm'].agg(['mean', 'std'])
            steps_g = grp_g.index.values
            mean_gn = grp_g['mean'].values
            std_gn = grp_g['std'].fillna(0).values
            ax_grad.plot(steps_g, mean_gn, color=color, label=opt, linewidth=1.2)
            ax_grad.fill_between(steps_g, mean_gn - std_gn, mean_gn + std_gn,
                                 alpha=0.15, color=color)

        # Red vertical bands at saddle regions
        for ss in saddle_steps:
            ax_loss.axvspan(ss - config.NN_SADDLE_CHECK_INTERVAL / 2,
                            ss + config.NN_SADDLE_CHECK_INTERVAL / 2,
                            alpha=0.12, color='red')

        ax_loss.set_title(f'{task_name} — Training Loss')
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend(fontsize=7, ncol=2)

        ax_grad.set_title(f'{task_name} — Gradient Norm')
        ax_grad.set_xlabel('Step')
        ax_grad.set_ylabel('||∇L||₂')
        ax_grad.legend(fontsize=7, ncol=2)

        # Add red band legend entry
        red_patch = mpatches.Patch(color='red', alpha=0.3, label='Saddle region')
        ax_loss.legend(handles=ax_loss.get_legend_handles_labels()[0] + [red_patch],
                       labels=ax_loss.get_legend_handles_labels()[1] + ['Saddle'],
                       fontsize=7, ncol=2)

    fig.suptitle('Neural Network Training — Real Datasets (5 runs, mean±std)',
                 fontsize=13, y=1.01)
    save_figure(fig, os.path.join(out_dir, 'figure4_nn_training.pdf'))


if __name__ == '__main__':
    plot_figure4()
