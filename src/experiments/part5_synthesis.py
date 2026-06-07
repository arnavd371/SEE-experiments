"""
Part 5: Predictive Validity — Does SEE predict training efficiency?

Experiments:
  5A: Spearman/Pearson correlation between SEE_quality (Part 1) and
      steps_to_ppl_threshold (Part 4)
  5B: SEE_quality vs steps_to_convergence for each NN task (Part 3)
  5C: Early plateau fraction (first 1000 steps of Part 4) vs final_val_ppl
  5D: Compute savings table between optimizer pairs

Results saved to results/part5_results.csv.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.logging_utils import PART5_FIELDS


def run_part5(fast: bool = False) -> pd.DataFrame:
    """Synthesize results from Parts 1, 3, 4 into predictive validity analyses."""

    p1_path = os.path.join(config.RESULTS_DIR, 'part1_results.csv')
    p3_path = os.path.join(config.RESULTS_DIR, 'part3_results.csv')
    p4_path = os.path.join(config.RESULTS_DIR, 'part4_results.csv')

    for path in [p1_path, p3_path, p4_path]:
        if not os.path.exists(path):
            print(f'WARNING: {path} not found. Run Parts 1-4 first.')
            return pd.DataFrame(columns=PART5_FIELDS)

    df1 = pd.read_csv(p1_path)
    df3 = pd.read_csv(p3_path)
    df4 = pd.read_csv(p4_path)

    # ── SEE_quality per optimizer from Part 1 ─────────────────────────────────
    # Mean across all functions and saddles at best_lr (highest mean SEE_quality)
    mean_see = {}
    for opt in config.OPTIMIZER_NAMES:
        sub = df1[df1['optimizer'] == opt]
        if sub.empty:
            mean_see[opt] = float('nan')
            continue
        # Best lr per optimizer = lr with highest mean SEE_quality
        lr_means = sub.groupby('lr')['SEE_quality'].mean()
        best_lr = lr_means.idxmax()
        mean_see[opt] = float(sub[sub['lr'] == best_lr]['SEE_quality'].mean())

    # ── Part 4 summary per optimizer ─────────────────────────────────────────
    # steps_to_ppl_threshold: use last step's value per (optimizer, run)
    p4_last = df4.groupby(['optimizer', 'run_seed']).last().reset_index()
    p4_agg = p4_last.groupby('optimizer').agg(
        steps_to_ppl_threshold=('steps_to_ppl_threshold', 'mean'),
        final_val_ppl=('final_val_ppl', 'mean'),
        plateau_fraction=('plateau_fraction', 'mean'),
    ).reset_index()
    p4_dict = p4_agg.set_index('optimizer').to_dict('index')

    # ── Part 3 summary per (optimizer, task) ──────────────────────────────────
    p3_last = df3.groupby(['task', 'optimizer', 'run_seed']).last().reset_index()
    p3_agg = p3_last.groupby(['task', 'optimizer']).agg(
        steps_to_convergence=('steps_to_convergence', 'mean'),
    ).reset_index()

    # ── Experiment 5C: early plateau fraction (first 1000 steps) ─────────────
    early_steps = 500 if fast else 1000
    df4_early = df4[df4['step'] <= early_steps]
    early_pf = {}
    for opt in config.OPTIMIZER_NAMES:
        sub = df4_early[df4_early['optimizer'] == opt]
        if sub.empty:
            early_pf[opt] = float('nan')
            continue
        early_pf[opt] = float(sub['is_plateau_step'].mean())

    # Correlation: early_pf vs final_val_ppl
    early_pf_vals = []
    final_ppls = []
    for opt in config.OPTIMIZER_NAMES:
        epf = early_pf.get(opt, float('nan'))
        fppl = p4_dict.get(opt, {}).get('final_val_ppl', float('nan'))
        if math.isfinite(epf) and math.isfinite(fppl):
            early_pf_vals.append(epf)
            final_ppls.append(fppl)

    early_vs_final_r = float('nan')
    if len(early_pf_vals) >= 3:
        try:
            r, _ = scipy.stats.pearsonr(early_pf_vals, final_ppls)
            early_vs_final_r = float(r)
        except Exception:
            pass

    # ── Experiment 5A: SEE vs steps_to_ppl ───────────────────────────────────
    see_vals = [mean_see.get(opt, float('nan')) for opt in config.OPTIMIZER_NAMES]
    ppl_steps = [p4_dict.get(opt, {}).get('steps_to_ppl_threshold', float('nan'))
                 for opt in config.OPTIMIZER_NAMES]

    valid_5a = [(s, p) for s, p in zip(see_vals, ppl_steps)
                if math.isfinite(s) and math.isfinite(p)]
    spearman_r_ppl = spearman_p_ppl = pearson_r_ppl = pearson_p_ppl = float('nan')
    if len(valid_5a) >= 3:
        sv, pv = zip(*valid_5a)
        try:
            sp_r, sp_p = scipy.stats.spearmanr(sv, pv)
            spearman_r_ppl, spearman_p_ppl = float(sp_r), float(sp_p)
            pe_r, pe_p = scipy.stats.pearsonr(sv, pv)
            pearson_r_ppl, pearson_p_ppl = float(pe_r), float(pe_p)
        except Exception:
            pass

    print(f'\n── Experiment 5A ──')
    print(f'  Spearman r(SEE_quality, steps_to_ppl) = {spearman_r_ppl:.3f}, '
          f'p = {spearman_p_ppl:.4f}')
    print(f'  Pearson  r(SEE_quality, steps_to_ppl) = {pearson_r_ppl:.3f}, '
          f'p = {pearson_p_ppl:.4f}')
    if not math.isnan(spearman_r_ppl) and abs(spearman_r_ppl) > 0.6 and spearman_p_ppl < 0.05:
        print('  → Strong evidence: SEE predicts training efficiency')

    # ── Experiment 5B: SEE vs steps_to_convergence ───────────────────────────
    print('\n── Experiment 5B ──')
    for task_name in ['Moons', 'MNIST-3v8', 'CalHousing']:
        sub3 = p3_agg[p3_agg['task'] == task_name]
        conv_steps = {row['optimizer']: row['steps_to_convergence']
                      for _, row in sub3.iterrows()}
        valid_5b = [(mean_see.get(opt, float('nan')), conv_steps.get(opt, float('nan')))
                    for opt in config.OPTIMIZER_NAMES]
        valid_5b = [(s, c) for s, c in valid_5b
                    if math.isfinite(s) and math.isfinite(c)]
        if len(valid_5b) >= 3:
            sv, cv = zip(*valid_5b)
            try:
                r5b, p5b = scipy.stats.spearmanr(sv, cv)
                print(f'  {task_name}: Spearman r = {r5b:.3f}, p = {p5b:.4f}')
            except Exception:
                print(f'  {task_name}: Correlation failed')

    # ── Experiment 5D: Compute savings ────────────────────────────────────────
    print('\n── Experiment 5D: Compute Savings ──')
    print(f'{"Optimizer A (better)":20s} {"Optimizer B (worse)":20s} '
          f'{"SEE_A":>8s} {"SEE_B":>8s} {"Plateau saved":>14s} {"Savings/$1M":>12s}')
    pf_dict = {opt: p4_dict.get(opt, {}).get('plateau_fraction', float('nan'))
               for opt in config.OPTIMIZER_NAMES}
    opts_ranked = sorted(config.OPTIMIZER_NAMES,
                         key=lambda o: mean_see.get(o, 0.0), reverse=True)
    for i, opt_a in enumerate(opts_ranked):
        for opt_b in opts_ranked[i + 1:]:
            see_a = mean_see.get(opt_a, float('nan'))
            see_b = mean_see.get(opt_b, float('nan'))
            pf_a = pf_dict.get(opt_a, float('nan'))
            pf_b = pf_dict.get(opt_b, float('nan'))
            if any(math.isnan(v) for v in [see_a, see_b, pf_a, pf_b]):
                continue
            savings_frac = pf_b - pf_a
            savings_1m = savings_frac * 1_000_000
            print(f'  {opt_a:20s} {opt_b:20s} '
                  f'{see_a:8.4f} {see_b:8.4f} '
                  f'{savings_frac:14.4f} ${savings_1m:>10,.0f}')

    # ── Build output rows ─────────────────────────────────────────────────────
    ranks_see = _rank_ascending_finite(
        [mean_see.get(opt, float('nan')) for opt in config.OPTIMIZER_NAMES],
        reverse=True  # higher SEE = better rank = lower rank number
    )
    ranks_ppl_step = _rank_ascending_finite(
        [pf_dict.get(opt, float('nan')) for opt in config.OPTIMIZER_NAMES],
        reverse=False
    )
    ranks_pf = _rank_ascending_finite(
        [pf_dict.get(opt, float('nan')) for opt in config.OPTIMIZER_NAMES],
        reverse=False
    )

    rows = []
    for i, opt in enumerate(config.OPTIMIZER_NAMES):
        rows.append({
            'optimizer': opt,
            'mean_SEE_quality': mean_see.get(opt, float('nan')),
            'SEE_quality_rank': ranks_see[i],
            'plateau_fraction': pf_dict.get(opt, float('nan')),
            'plateau_fraction_rank': ranks_pf[i],
            'steps_to_ppl_threshold': p4_dict.get(opt, {}).get('steps_to_ppl_threshold', float('nan')),
            'ppl_rank': ranks_ppl_step[i],
            'spearman_r_SEE_ppl': spearman_r_ppl,
            'spearman_p_SEE_ppl': spearman_p_ppl,
            'pearson_r_SEE_ppl': pearson_r_ppl,
            'pearson_p_SEE_ppl': pearson_p_ppl,
            'early_plateau_fraction': early_pf.get(opt, float('nan')),
            'early_vs_final_ppl_r': early_vs_final_r,
        })

    df5 = pd.DataFrame(rows, columns=PART5_FIELDS)
    out_csv = os.path.join(config.RESULTS_DIR, 'part5_results.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df5.to_csv(out_csv, index=False)
    print(f'\nPart 5 results saved to {out_csv}')
    return df5


def _rank_ascending_finite(values: list, reverse: bool = False) -> list:
    """Return 1-based ranks; NaN → NaN."""
    import math
    indexed = [(v, i) for i, v in enumerate(values) if math.isfinite(v)]
    indexed.sort(key=lambda t: t[0], reverse=reverse)
    rank_map = {i: r + 1 for r, (_, i) in enumerate(indexed)}
    return [rank_map.get(i, float('nan')) for i in range(len(values))]
