"""
Part 2: High-Dimensional Scaling Experiments.

Tests SEE vs dimension for 4 nD functions (Rastrigin, Styblinski, Ackley, Synthetic saddle).
For synthetic saddle: tests multiple saddle indices k.
Uses best_lrs from Part 1.
Results saved to results/part2_results.csv.
"""

import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.seeding import set_all_seeds, trial_seed
from src.utils.logging_utils import PART2_FIELDS
from src.utils.parallel import run_parallel
from src.functions.nd_functions import get_nd_function, saddle_location_nd, saddle_indices_for_dim
from src.optimizers.wrapper import make_optimizer
from src.metrics.hessian import compute_min_eigenvalue
from src.metrics.see import compute_see


def _run_single_trial_nd(args: tuple) -> dict:
    """
    Top-level picklable trial runner for nD functions.
    args = (fn_name, k, d, opt_name, lr, seed, T_max)
    """
    fn_name, k, d, opt_name, lr, seed, T_max = args

    import torch, numpy as np, math, sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    import config
    from src.functions.nd_functions import get_nd_function, saddle_location_nd
    from src.optimizers.wrapper import make_optimizer
    from src.metrics.hessian import compute_min_eigenvalue

    torch.manual_seed(seed)
    np.random.seed(seed)

    fn = get_nd_function(fn_name, k=k)
    saddle = saddle_location_nd(fn_name, d, k=k).to(torch.float64)
    diverge_radius = 10.0 * float(saddle.norm()) + 50.0

    noise = torch.randn(d, dtype=torch.float64) * config.PERTURBATION_STD
    x = torch.nn.Parameter((saddle + noise).clone())
    opt = make_optimizer(opt_name, [x], lr)

    final_loss = float('nan')
    for t in range(1, T_max + 1):
        opt.zero_grad()
        try:
            loss = fn(x)
            if not torch.isfinite(loss):
                return {'outcome': 'diverge', 'tau': t}
            loss.backward()
        except Exception:
            return {'outcome': 'diverge', 'tau': t}

        final_loss = loss.item()
        with torch.no_grad():
            grad_norm = float(x.grad.norm()) if x.grad is not None else 0.0
            x_norm = float(x.norm())

        if not math.isfinite(grad_norm) or not math.isfinite(x_norm):
            return {'outcome': 'diverge', 'tau': t}

        if x_norm > diverge_radius:
            return {'outcome': 'diverge', 'tau': t}

        if grad_norm < config.GRAD_TOL:
            try:
                min_eig = compute_min_eigenvalue(fn, x.detach(), d=d)
                if math.isfinite(min_eig) and min_eig > -config.EIGEN_TOL_POS:
                    return {'outcome': 'local_min', 'tau': t}
            except Exception:
                pass

        opt.step()

    return {'outcome': 'stuck', 'tau': T_max}


def run_config_nd(task: tuple) -> dict:
    fn_name, k, d, opt_name, lr, N_trials, T_max, base_seed = task

    trial_args = [
        (fn_name, k, d, opt_name, lr, trial_seed(base_seed, i), T_max)
        for i in range(N_trials)
    ]
    results = [_run_single_trial_nd(a) for a in trial_args]

    outcomes = np.array([r['outcome'] for r in results], dtype=object)
    taus = np.array([r['tau'] for r in results], dtype=float)
    see_metrics = compute_see(outcomes, taus)

    return {
        'fn_name': fn_name, 'k': k, 'd': d,
        'opt_name': opt_name, 'lr': lr,
        'outcomes': outcomes, 'taus': taus,
        'see_metrics': see_metrics,
    }


def _fit_power_law(dims: list, tau_means: list) -> float:
    """Fit τ_avg ~ d^α via linear regression in log-log space. Returns α."""
    valid = [(d, t) for d, t in zip(dims, tau_means)
             if t is not None and math.isfinite(t) and t > 0 and d > 1]
    if len(valid) < 2:
        return float('nan')
    log_d = np.array([math.log(d) for d, _ in valid])
    log_t = np.array([math.log(t) for _, t in valid])
    alpha = float(np.polyfit(log_d, log_t, 1)[0])
    return alpha


def load_best_lrs() -> dict:
    if not os.path.exists(config.BEST_LRS_PATH):
        # Default fallback
        return {opt: 0.01 for opt in config.OPTIMIZER_NAMES}
    with open(config.BEST_LRS_PATH) as f:
        return yaml.safe_load(f)


def run_part2(fast: bool = False, resume: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)
    best_lrs = load_best_lrs()

    dimensions = [d for d in config.DIMENSIONS
                  if not fast or d <= config.FAST_D_MAX]

    rows = []
    out_csv = os.path.join(config.RESULTS_DIR, 'part2_results.csv')

    for fn_name in config.ND_FUNCTION_NAMES:
        print(f'\n── {fn_name} ──')

        # Determine saddle indices (only relevant for Synthetic-Saddle)
        for d in dimensions:
            N_trials = (config.FAST_N_TRIALS if fast else
                        (config.N_TRIALS_PART2_HIGHD if d > 50 else config.N_TRIALS_PART2))
            T_max = (config.FAST_T_MAX if fast else
                     (config.T_MAX_HIGHD if d > 50 else config.T_MAX))

            if fn_name == 'Synthetic-Saddle':
                k_values = saddle_indices_for_dim(d)
            else:
                k_values = [None]  # single saddle at origin

            for k in k_values:
                print(f'  d={d}, k={k}...')
                tasks = [
                    (fn_name, k, d, opt_name, best_lrs.get(opt_name, 0.01),
                     N_trials, T_max, config.GLOBAL_SEED)
                    for opt_name in config.OPTIMIZER_NAMES
                ]
                opt_results = run_parallel(run_config_nd, tasks)

                for r in opt_results:
                    sm = r['see_metrics']
                    row = {
                        'function': fn_name,
                        'optimizer': r['opt_name'],
                        'best_lr': r['lr'],
                        'dimension': d,
                        'saddle_index_k': k if k is not None else '',
                        'SEE_basic': sm['SEE_basic'],
                        'SEE_quality': sm['SEE_quality'],
                        'SEE_diverge': sm['SEE_diverge'],
                        'SEE_basic_CI_lo': sm['SEE_basic_CI_lo'],
                        'SEE_basic_CI_hi': sm['SEE_basic_CI_hi'],
                        'tau_mean': sm['tau_mean'],
                        'tau_std': sm['tau_std'],
                        'escape_min_pct': sm['escape_min_pct'],
                        'stuck_pct': sm['stuck_pct'],
                        'power_law_alpha': float('nan'),  # filled below
                    }
                    rows.append(row)

    df = pd.DataFrame(rows, columns=PART2_FIELDS)

    # ── Power law fitting: τ_avg ~ d^α per (function, optimizer) ─────────────
    print('\nFitting power laws...')
    for fn_name in config.ND_FUNCTION_NAMES:
        for opt_name in config.OPTIMIZER_NAMES:
            mask = (df['function'] == fn_name) & (df['optimizer'] == opt_name)
            sub = df[mask].sort_values('dimension')
            alpha = _fit_power_law(sub['dimension'].tolist(), sub['tau_mean'].tolist())
            df.loc[mask, 'power_law_alpha'] = alpha

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Part 2 results saved to {out_csv}')
    return df
