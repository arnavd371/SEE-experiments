"""Figure 6: Escape type breakdown — stacked horizontal bar chart per optimizer."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config
from src.plots.style import apply_style, save_figure, OUTCOME_COLORS


def plot_figure6(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part1_results.csv')
        if not os.path.exists(path):
            print('part1_results.csv not found; skipping Figure 6.')
            return
        df = pd.read_csv(path)

    # Average across all functions and saddles, per optimizer
    agg = df.groupby('optimizer')[
        ['escape_min_pct', 'escape_diverge_pct', 'stuck_pct', 'SEE_quality']
    ].mean().reset_index()

    # Sort by SEE_quality descending
    agg = agg.sort_values('SEE_quality', ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(agg))
    bar_h = 0.65

    ax.barh(y_pos, agg['escape_min_pct'], height=bar_h,
            color=OUTCOME_COLORS['local_min'], label='LOCAL_MIN')
    ax.barh(y_pos, agg['escape_diverge_pct'], height=bar_h,
            left=agg['escape_min_pct'],
            color=OUTCOME_COLORS['diverge'], label='DIVERGE')
    ax.barh(y_pos, agg['stuck_pct'], height=bar_h,
            left=agg['escape_min_pct'] + agg['escape_diverge_pct'],
            color=OUTCOME_COLORS['stuck'], label='STUCK')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(agg['optimizer'])
    ax.set_xlabel('Percentage of Trials (%)')
    ax.set_title('Escape Type Breakdown by Optimizer\n(averaged over all Part 1 functions)')
    ax.set_xlim(0, 100)
    ax.legend(loc='lower right', fontsize=9)

    # Add SEE_quality annotation on right
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f'SEE={v:.4f}' for v in agg['SEE_quality']], fontsize=8)
    ax2.spines['right'].set_visible(True)

    save_figure(fig, os.path.join(out_dir, 'figure6_escape_types.pdf'))


if __name__ == '__main__':
    plot_figure6()
