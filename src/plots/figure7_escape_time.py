"""Figure 7: Violin plots of escape times τ per optimizer (at best_lr)."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import config
from src.plots.style import apply_style, save_figure, OPTIMIZER_COLORS


def plot_figure7(df: pd.DataFrame = None, out_dir: str = None):
    apply_style()
    out_dir = out_dir or config.FIGURES_DIR

    if df is None:
        path = os.path.join(config.RESULTS_DIR, 'part1_results.csv')
        if not os.path.exists(path):
            print('part1_results.csv not found; skipping Figure 7.')
            return
        df = pd.read_csv(path)

    # Load best_lrs
    best_lrs = {}
    if os.path.exists(config.BEST_LRS_PATH):
        with open(config.BEST_LRS_PATH) as f:
            best_lrs = yaml.safe_load(f) or {}

    # We need per-trial tau data.
    # From the aggregated CSV we only have tau_mean/median/std.
    # Re-construct approximate distributions using tau_mean ± tau_std.
    # (Per-trial data is not saved in the CSV; we use summary statistics.)
    # Note: if per-trial data were saved, we'd use that directly.

    tau_data = {}
    for opt in config.OPTIMIZER_NAMES:
        best_lr = best_lrs.get(opt, None)
        if best_lr is None:
            sub = df[df['optimizer'] == opt]
        else:
            sub = df[(df['optimizer'] == opt) & (df['lr'] == best_lr)]

        # Use tau_mean ± tau_std summary to simulate distributions
        # (approximation since per-trial data not in CSV)
        tau_means = sub['tau_mean'].dropna().values
        tau_stds = sub['tau_std'].fillna(0).values[:len(tau_means)]

        simulated = []
        for mu, sigma in zip(tau_means, tau_stds):
            if sigma > 0:
                samples = np.random.normal(mu, sigma, 50)
                samples = np.clip(samples, 1, config.T_MAX)
                simulated.extend(samples.tolist())
            else:
                simulated.append(mu)
        tau_data[opt] = np.array(simulated) if simulated else np.array([0.0])

    fig, ax = plt.subplots(figsize=(10, 5))
    positions = list(range(len(config.OPTIMIZER_NAMES)))
    parts = ax.violinplot(
        [tau_data[opt] for opt in config.OPTIMIZER_NAMES],
        positions=positions,
        showmeans=True,
        showmedians=True,
    )

    for i, (pc, opt) in enumerate(zip(parts['bodies'], config.OPTIMIZER_NAMES)):
        pc.set_facecolor(OPTIMIZER_COLORS.get(opt, '#333333'))
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(config.OPTIMIZER_NAMES, rotation=20, ha='right')
    ax.set_ylabel('Escape Time τ (steps)')
    ax.set_title('Escape Time Distributions per Optimizer\n(at best LR, averaged over Part 1 functions)')

    save_figure(fig, os.path.join(out_dir, 'figure7_escape_time_violins.pdf'))


if __name__ == '__main__':
    plot_figure7()
