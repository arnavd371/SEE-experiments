"""
Part 4: Synthesis and ablations.

4A. SEE_basic vs SEE_quality comparison
4B. Sensitivity analysis on r
4C. SEE_NN correlation with benchmark SEE
4D. Optimizer ranking consistency heatmap
"""
from __future__ import annotations
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.stats
import torch

import config
from src.functions.classical_2d import FUNCTIONS_2D
from src.functions.saddle_finder import find_saddles_2d, compute_r
from src.metrics.see import compute_see
from src.utils.seeding import set_all_seeds
from src.experiments.part1_vectorized import run_trials_vectorized, make_optimizer


def _spearman(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if len(a) < 3:
        return np.nan, np.nan
    r, p = scipy.stats.spearmanr(a, b)
    return float(r), float(p)


def run_4a(df1: pd.DataFrame) -> pd.DataFrame:
    """SEE_basic vs SEE_quality comparison."""
    # Per (function, optimizer, lr): ratio and agreement
    df1 = df1.copy()
    df1['ratio_qual_basic'] = df1['SEE_quality'] / (df1['SEE_basic'] + 1e-10)

    rows = []
    for (fname, opt, lr), grp in df1.groupby(['function', 'optimizer', 'lr']):
        rows.append({
            'function': fname, 'optimizer': opt, 'lr': lr,
            'SEE_basic': grp['SEE_basic'].mean(),
            'SEE_quality': grp['SEE_quality'].mean(),
            'ratio': grp['ratio_qual_basic'].mean(),
        })
    df4a = pd.DataFrame(rows)

    # Spearman rank correlation between SEE_basic and SEE_quality rankings
    # Across all (optimizer, lr) combinations, for each function
    corr_rows = []
    for fname in df1['function'].unique():
        sub = df1[df1['function'] == fname].groupby(['optimizer', 'lr'])[
            ['SEE_basic', 'SEE_quality']].mean().reset_index()
        r, p = _spearman(sub['SEE_basic'].values, sub['SEE_quality'].values)
        corr_rows.append({'function': fname, 'spearman_r': r, 'p_value': p})

    df4a_corr = pd.DataFrame(corr_rows)
    print("\n4A: SEE_basic vs SEE_quality Spearman correlations:")
    print(df4a_corr.to_string(index=False))

    # Ranking agreement: how often do SEE_basic and SEE_quality agree on best optimizer?
    agreement_count = 0
    total_count = 0
    for (fname, lr), grp in df1.groupby(['function', 'lr']):
        grp2 = grp.groupby('optimizer')[['SEE_basic', 'SEE_quality']].mean()
        if len(grp2) > 1:
            best_basic = grp2['SEE_basic'].idxmax()
            best_qual  = grp2['SEE_quality'].idxmax()
            agreement_count += int(best_basic == best_qual)
            total_count += 1
    agreement_pct = 100.0 * agreement_count / max(total_count, 1)
    print(f"\n4A: Best-optimizer agreement: {agreement_pct:.1f}% ({agreement_count}/{total_count})")

    return df4a, df4a_corr


def run_4b(best_lrs: dict, fast: bool = False) -> pd.DataFrame:
    """Sensitivity analysis: vary curvature constant c."""
    set_all_seeds(config.GLOBAL_SEED)
    device = config.DEVICE
    N     = config.FAST_N_TRIALS if fast else config.N_TRIALS
    T_max = config.FAST_T_MAX    if fast else config.T_MAX

    rows = []
    for fname, (fn, domain) in FUNCTIONS_2D.items():
        x_lo, x_hi, y_lo, y_hi = domain
        import math
        domain_diameter = math.sqrt((x_hi - x_lo)**2 + (y_hi - y_lo)**2)
        r_max = 0.25 * domain_diameter

        saddles = find_saddles_2d(fn, domain, device,
                                  grid_n=config.SADDLE_GRID,
                                  max_saddles=config.MAX_SADDLES)
        if not saddles:
            continue

        sad = saddles[0]  # use first saddle for sensitivity analysis
        x_s = sad['x_s']
        lmin_abs = abs(sad['lambda_min'])

        for c in config.SENSITIVITY_CONSTANTS:
            r_c = min(r_max, c / math.sqrt(lmin_abs + config.CURVATURE_EPSILON))

            for opt_name in config.OPTIMIZERS:
                lr = best_lrs.get(opt_name, 0.01)
                esc_min, esc_div, esc_t = run_trials_vectorized(
                    fn, x_s, r_c, opt_name, lr, N, T_max, device
                )
                metrics = compute_see(esc_min, esc_div, esc_t, T_max,
                                      n_resamples=config.BOOTSTRAP_RESAMPLES)
                rows.append({
                    'function': fname, 'optimizer': opt_name,
                    'curvature_c': c, 'r': r_c,
                    'SEE_quality': metrics['SEE_quality'],
                    'SEE_basic':   metrics['SEE_basic'],
                })
            print(f"  4B: {fname} c={c} done")

    df4b = pd.DataFrame(rows)
    return df4b


def run_4c(df1: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    """SEE_NN correlation with benchmark SEE."""
    # x = mean SEE_quality from Part 1 at best LR per optimizer
    # y = mean SEE_NN from Part 3 per optimizer
    import yaml
    try:
        with open('results/best_lrs.yaml') as f:
            best_lrs = yaml.safe_load(f)
    except FileNotFoundError:
        best_lrs = {}

    rows = []
    for opt_name in config.OPTIMIZERS:
        lr_best = best_lrs.get(opt_name, None)
        if lr_best is not None:
            mask1 = (df1['optimizer'] == opt_name) & (df1['lr'] == lr_best)
        else:
            mask1 = df1['optimizer'] == opt_name
        see_bench = df1.loc[mask1, 'SEE_quality'].mean()

        mask3 = (df3['optimizer'] == opt_name) & df3['SEE_NN'].notna()
        see_nn = df3.loc[mask3, 'SEE_NN'].mean()

        rows.append({'optimizer': opt_name, 'SEE_quality_bench': see_bench, 'SEE_NN': see_nn})

    df4c = pd.DataFrame(rows).dropna()
    if len(df4c) >= 3:
        r, p = _spearman(df4c['SEE_quality_bench'].values, df4c['SEE_NN'].values)
        print(f"\n4C: Spearman r={r:.3f} p={p:.3f}")
        if r > 0.6:
            print("   => benchmark SEE predicts NN saddle escape")
        df4c['spearman_r'] = r
        df4c['spearman_p'] = p
    else:
        print("\n4C: insufficient saddle events detected in NN training for correlation")
        df4c['spearman_r'] = np.nan
        df4c['spearman_p'] = np.nan

    return df4c


def run_4d(df1: pd.DataFrame) -> pd.DataFrame:
    """Optimizer ranking consistency: 6×6 Spearman heatmap across function pairs."""
    opt_names = list(config.OPTIMIZERS.keys())
    fn_names  = df1['function'].unique().tolist()

    # For each function, optimizer ranking by mean SEE_quality
    rankings = {}
    for fname in fn_names:
        sub = df1[df1['function'] == fname].groupby('optimizer')['SEE_quality'].mean()
        ranks = sub.rank(ascending=False)
        rankings[fname] = {opt: float(ranks.get(opt, np.nan)) for opt in opt_names}

    # Pairwise Spearman
    results = {}
    for f1 in fn_names:
        for f2 in fn_names:
            r1 = [rankings[f1][o] for o in opt_names]
            r2 = [rankings[f2][o] for o in opt_names]
            r, p = _spearman(np.array(r1), np.array(r2))
            results[(f1, f2)] = r

    rows = [{'fn1': f1, 'fn2': f2, 'spearman_r': results[(f1, f2)]}
            for f1 in fn_names for f2 in fn_names]
    df4d = pd.DataFrame(rows)
    return df4d


def run_part4(df1: pd.DataFrame, df3: pd.DataFrame, best_lrs: dict, fast: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)

    print("\n=== Part 4A: SEE_basic vs SEE_quality ===")
    df4a, df4a_corr = run_4a(df1)

    print("\n=== Part 4B: Sensitivity analysis ===")
    df4b = run_4b(best_lrs, fast=fast)

    print("\n=== Part 4C: SEE_NN correlation ===")
    df4c = run_4c(df1, df3)

    print("\n=== Part 4D: Optimizer ranking consistency ===")
    df4d = run_4d(df1)

    out = Path('results/part4_synthesis.csv')
    out.parent.mkdir(exist_ok=True)

    # Write combined output
    with pd.ExcelWriter(str(out).replace('.csv', '.xlsx'), engine='openpyxl') as writer:
        df4a.to_excel(writer, sheet_name='4A_comparison', index=False)
        df4a_corr.to_excel(writer, sheet_name='4A_correlation', index=False)
        df4b.to_excel(writer, sheet_name='4B_sensitivity', index=False)
        df4c.to_excel(writer, sheet_name='4C_nn_corr', index=False)
        df4d.to_excel(writer, sheet_name='4D_ranking_heatmap', index=False)

    df4b.to_csv(out, index=False)
    print(f"\nPart 4 results saved to {out} and .xlsx")
    return df4a, df4b, df4c, df4d
