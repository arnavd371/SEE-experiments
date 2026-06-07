"""
Part 1: Classical 2D Benchmark Experiments.

For each (function, saddle, optimizer, lr): run N_trials escape trials,
compute SEE metrics, bootstrap CIs, and Wilcoxon/Cohen's d statistics.
Results saved to results/part1_results.csv.
best_lrs.yaml populated at the end.
"""

import os
import sys
import warnings
import math
import numpy as np
import pandas as pd
import torch
import yaml

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.seeding import set_all_seeds, trial_seed
from src.utils.checkpointing import save_checkpoint, load_checkpoint, is_done
from src.utils.logging_utils import PART1_FIELDS
from src.utils.parallel import run_parallel
from src.functions.classical_2d import get_function_2d
from src.functions.saddle_finder import find_saddles_2d
from src.optimizers.wrapper import make_optimizer
from src.metrics.hessian import compute_min_eigenvalue
from src.metrics.see import compute_see
from src.metrics.statistics import pairwise_wilcoxon, p_vs_best


# ── Core trial runner (top-level for joblib pickling) ─────────────────────────

def _run_single_trial(args: tuple) -> dict:
    """
    Run one escape trial on CPU.
    args = (fn_name, saddle_loc, opt_name, lr, seed, T_max)
    Returns: {'outcome': str, 'tau': int, 'final_loss': float}
    """
    fn_name, saddle_loc, opt_name, lr, seed, T_max = args

    import torch
    import numpy as np
    # Re-import config and modules (each subprocess needs fresh imports)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    import config
    from src.functions.classical_2d import get_function_2d
    from src.optimizers.wrapper import make_optimizer
    from src.metrics.hessian import compute_min_eigenvalue

    torch.manual_seed(seed)
    np.random.seed(seed)

    fn = get_function_2d(fn_name)
    saddle = torch.tensor(saddle_loc, dtype=torch.float64)
    d = len(saddle_loc)
    diverge_radius = 10.0 * float(saddle.norm()) + 50.0

    # Perturb from saddle
    noise = torch.randn(d, dtype=torch.float64) * config.PERTURBATION_STD
    x_init = (saddle + noise).detach()
    x = torch.nn.Parameter(x_init.clone())
    opt = make_optimizer(opt_name, [x], lr)

    final_loss = float('nan')
    for t in range(1, T_max + 1):
        opt.zero_grad()
        try:
            loss = fn(x)
            if not torch.isfinite(loss):
                return {'outcome': 'diverge', 'tau': t, 'final_loss': float('nan')}
            loss.backward()
        except Exception:
            return {'outcome': 'diverge', 'tau': t, 'final_loss': float('nan')}

        final_loss = loss.item()

        with torch.no_grad():
            if x.grad is None:
                grad_norm = 0.0
            else:
                grad_norm = float(x.grad.norm())
            x_norm = float(x.norm())

        if not math.isfinite(grad_norm) or not math.isfinite(x_norm):
            return {'outcome': 'diverge', 'tau': t, 'final_loss': final_loss}

        # DIVERGE check
        if x_norm > diverge_radius:
            return {'outcome': 'diverge', 'tau': t, 'final_loss': final_loss}

        # LOCAL_MIN check (only when gradient is very small)
        if grad_norm < config.GRAD_TOL:
            try:
                min_eig = compute_min_eigenvalue(fn, x.detach(), d=d)
                if math.isfinite(min_eig) and min_eig > -config.EIGEN_TOL_POS:
                    return {'outcome': 'local_min', 'tau': t, 'final_loss': final_loss}
            except Exception:
                pass  # if Hessian fails, continue running

        opt.step()

    return {'outcome': 'stuck', 'tau': T_max, 'final_loss': final_loss}


def run_config(task: tuple) -> dict:
    """
    Run all N_trials for one (fn_name, saddle_id, saddle_loc, opt_name, lr) config.
    Returns a dict of trial results.
    """
    fn_name, saddle_id, saddle_loc, opt_name, lr, N_trials, T_max, base_seed = task

    trial_args = [
        (fn_name, saddle_loc, opt_name, lr,
         trial_seed(base_seed, i), T_max)
        for i in range(N_trials)
    ]

    results = [_run_single_trial(a) for a in trial_args]

    outcomes = np.array([r['outcome'] for r in results], dtype=object)
    taus = np.array([r['tau'] for r in results], dtype=float)
    final_losses = np.array([r['final_loss'] for r in results], dtype=float)

    see_metrics = compute_see(outcomes, taus)

    # best_loss_at_escape_mean: mean final loss over local_min trials only
    min_mask = (outcomes == 'local_min')
    best_loss = float(final_losses[min_mask].mean()) if min_mask.any() else float('nan')

    return {
        'fn_name': fn_name,
        'saddle_id': saddle_id,
        'saddle_loc': saddle_loc,
        'opt_name': opt_name,
        'lr': lr,
        'outcomes': outcomes,
        'taus': taus,
        'see_metrics': see_metrics,
        'best_loss_at_escape_mean': best_loss,
    }


def run_part1(fast: bool = False, resume: bool = False) -> pd.DataFrame:
    """Run Part 1 experiments. Returns DataFrame of results."""
    set_all_seeds(config.GLOBAL_SEED)

    N_trials = config.FAST_N_TRIALS if fast else config.N_TRIALS_PART1
    T_max = config.FAST_T_MAX if fast else config.T_MAX
    grid_size = config.FAST_SADDLE_GRID if fast else config.GRID_SIZE
    n_bootstrap = config.FAST_BOOTSTRAP if fast else config.BOOTSTRAP_RESAMPLES
    lrs_to_test = config.FAST_LRS if fast else config.LEARNING_RATES
    max_saddles = config.FAST_MAX_SADDLES if fast else None

    out_csv = os.path.join(config.RESULTS_DIR, 'part1_results.csv')
    ckpt = load_checkpoint(1) if resume else {}
    completed = set(ckpt.get('completed_keys', []))

    # ── Find saddles for each function ────────────────────────────────────────
    print('Finding saddles...')
    saddles_by_fn = {}
    for fn_name in config.FUNCTION_NAMES_2D:
        saddles = find_saddles_2d(fn_name, grid_size=grid_size)
        from src.functions.saddle_finder import cap_saddles
        saddles = cap_saddles(saddles, config.MAX_SADDLES_PER_FUNCTION)
        if not saddles:
            # Fallback: use domain center as a nominal saddle
            domain = config.FUNCTION_DOMAINS[fn_name]
            saddles = [((domain[0][0] + domain[0][1]) / 2,
                        (domain[1][0] + domain[1][1]) / 2)]
            warnings.warn(f'No saddles found for {fn_name}; using domain center.')
        if max_saddles is not None:
            saddles = saddles[:max_saddles]
        saddles_by_fn[fn_name] = saddles
        print(f'  {fn_name}: {len(saddles)} saddle(s)')

    # ── Build task list ────────────────────────────────────────────────────────
    tasks = []
    for fn_name, saddles in saddles_by_fn.items():
        for saddle_id, saddle_loc in enumerate(saddles):
            for opt_name in config.OPTIMIZER_NAMES:
                for lr in lrs_to_test:
                    key = (fn_name, saddle_id, opt_name, lr)
                    if resume and is_done(completed, key):
                        continue
                    tasks.append(
                        (fn_name, saddle_id, saddle_loc, opt_name, lr,
                         N_trials, T_max, config.GLOBAL_SEED)
                    )

    print(f'Running {len(tasks)} configurations × {N_trials} trials each...')

    # Run in parallel
    raw_results = run_parallel(run_config, tasks)

    # ── Aggregate into per-config rows ────────────────────────────────────────
    # Store per-config outcomes/taus for Wilcoxon tests
    config_data = {}
    for r in raw_results:
        key = (r['fn_name'], r['saddle_id'], r['opt_name'], r['lr'])
        config_data[key] = r

    # Also load previously completed configs from checkpoint
    for key, r in ckpt.get('config_data', {}).items():
        if key not in config_data:
            config_data[key] = r

    # ── Wilcoxon tests per (function, lr) ─────────────────────────────────────
    print('Computing Wilcoxon tests...')
    wilcoxon_results = {}
    for fn_name, saddles in saddles_by_fn.items():
        for saddle_id in range(len(saddles)):
            for lr in lrs_to_test:
                outcomes_dict = {}
                taus_dict = {}
                for opt_name in config.OPTIMIZER_NAMES:
                    key = (fn_name, saddle_id, opt_name, lr)
                    if key in config_data:
                        outcomes_dict[opt_name] = config_data[key]['outcomes']
                        taus_dict[opt_name] = config_data[key]['taus']
                if len(outcomes_dict) < 2:
                    continue
                pairs = pairwise_wilcoxon(
                    list(outcomes_dict.keys()), outcomes_dict, taus_dict
                )
                wilcoxon_results[(fn_name, saddle_id, lr)] = pairs

    # Find best optimizer per (function, lr) by SEE_quality
    best_opt_map = {}
    for fn_name, saddles in saddles_by_fn.items():
        for saddle_id in range(len(saddles)):
            for lr in lrs_to_test:
                best_see = -1.0
                best_opt = config.OPTIMIZER_NAMES[0]
                for opt_name in config.OPTIMIZER_NAMES:
                    key = (fn_name, saddle_id, opt_name, lr)
                    if key in config_data:
                        sq = config_data[key]['see_metrics']['SEE_quality']
                        if sq > best_see:
                            best_see = sq
                            best_opt = opt_name
                best_opt_map[(fn_name, saddle_id, lr)] = best_opt

    # ── Build output rows ──────────────────────────────────────────────────────
    rows = []
    for (fn_name, saddle_id, opt_name, lr), r in config_data.items():
        saddle_loc = r['saddle_loc']
        sm = r['see_metrics']

        # Wilcoxon vs best
        best_opt = best_opt_map.get((fn_name, saddle_id, lr), opt_name)
        wilc_p, cohen_d_val = float('nan'), float('nan')
        if best_opt != opt_name:
            wkey = (fn_name, saddle_id, lr)
            if wkey in wilcoxon_results:
                pairs = wilcoxon_results[wkey]
                pair_key = (opt_name, best_opt) if (opt_name, best_opt) in pairs else (best_opt, opt_name)
                if pair_key in pairs:
                    wilc_p = pairs[pair_key]['p_raw']
                    cohen_d_val = pairs[pair_key]['cohens_d']

        row = {
            'function': fn_name,
            'optimizer': opt_name,
            'lr': lr,
            'saddle_id': saddle_id,
            'saddle_x': saddle_loc[0],
            'saddle_y': saddle_loc[1],
            'SEE_basic': sm['SEE_basic'],
            'SEE_quality': sm['SEE_quality'],
            'SEE_diverge': sm['SEE_diverge'],
            'SEE_basic_CI_lo': sm['SEE_basic_CI_lo'],
            'SEE_basic_CI_hi': sm['SEE_basic_CI_hi'],
            'SEE_quality_CI_lo': sm['SEE_quality_CI_lo'],
            'SEE_quality_CI_hi': sm['SEE_quality_CI_hi'],
            'escape_min_pct': sm['escape_min_pct'],
            'escape_diverge_pct': sm['escape_diverge_pct'],
            'stuck_pct': sm['stuck_pct'],
            'tau_mean': sm['tau_mean'],
            'tau_median': sm['tau_median'],
            'tau_std': sm['tau_std'],
            'n_trials': sm['n_trials'],
            'best_loss_at_escape_mean': r['best_loss_at_escape_mean'],
            'wilcoxon_p_vs_best': wilc_p,
            'cohens_d_vs_best': cohen_d_val,
            'reliable': sm['reliable'],
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=PART1_FIELDS)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'Part 1 results saved to {out_csv}')

    # ── Compute and save best_lrs.yaml ────────────────────────────────────────
    best_lrs = {}
    for opt_name in config.OPTIMIZER_NAMES:
        opt_rows = df[df['optimizer'] == opt_name]
        if opt_rows.empty:
            best_lrs[opt_name] = config.LEARNING_RATES[3]  # default 0.005
            continue
        lr_means = opt_rows.groupby('lr')['SEE_quality'].mean()
        best_lrs[opt_name] = float(lr_means.idxmax())

    os.makedirs(os.path.dirname(config.BEST_LRS_PATH), exist_ok=True)
    with open(config.BEST_LRS_PATH, 'w') as f:
        yaml.dump(best_lrs, f, default_flow_style=False)
    print(f'Best LRs saved to {config.BEST_LRS_PATH}')
    print('Best LRs:', best_lrs)

    # ── Save checkpoint ────────────────────────────────────────────────────────
    all_keys = list(config_data.keys())
    save_checkpoint(1, {'completed_keys': all_keys, 'config_data': config_data})

    return df
