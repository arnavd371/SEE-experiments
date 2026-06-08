"""
Part 2: High-dimensional scaling experiments.

Uses best LR per optimizer from Part 1.
Vectorizes all trials as (N, d) tensors.
Uses Lanczos for d > 20 eigenvalue computation.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse.linalg
import torch
import yaml

import config
from src.functions.nd_functions import rastrigin_nd, styblinski_nd, synthetic_saddle
from src.metrics.see import compute_see
from src.utils.seeding import set_all_seeds
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.experiments.part1_vectorized import make_optimizer


def _find_nd_saddle_synthetic(k: int, d: int, device):
    """Synthetic saddle is at origin by construction. r = 0.5/sqrt(2)."""
    x_s = np.zeros(d)
    lambda_min = -2.0
    r = 0.5 / math.sqrt(abs(lambda_min) + 1e-6)
    return x_s, r, lambda_min


def _lanczos_lambda_min(fn: Callable, x_np: np.ndarray, device, k: int = 6) -> float:
    """Estimate smallest Hessian eigenvalue via Lanczos (matrix-free)."""
    d = len(x_np)
    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
    v = fn(x)
    g = torch.autograd.grad(v, x, create_graph=True)[0]

    def hess_vec(vec_np):
        vec = torch.tensor(vec_np, dtype=torch.float32, device=device)
        hv = torch.autograd.grad(g, x, grad_outputs=vec, retain_graph=True)[0]
        return hv.detach().cpu().numpy().astype(np.float64)

    lo_op = scipy.sparse.linalg.LinearOperator((d, d), matvec=hess_vec)
    k_eff = min(k, d - 1)
    if k_eff < 1:
        return 0.0
    try:
        eigs = scipy.sparse.linalg.eigsh(lo_op, k=k_eff, which='SA',
                                          return_eigenvectors=False, tol=1e-3,
                                          maxiter=d * 10)
        return float(eigs.min())
    except Exception:
        return 0.0


def _full_lambda_min(fn: Callable, x_np: np.ndarray, device) -> float:
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    H = torch.autograd.functional.hessian(fn, x)
    H_np = H.detach().cpu().numpy().reshape(len(x_np), len(x_np))
    return float(np.linalg.eigvalsh(H_np).min())


def compute_lambda_min(fn, x_np, device, d):
    if d > 20:
        return _lanczos_lambda_min(fn, x_np, device)
    return _full_lambda_min(fn, x_np, device)


def run_trials_nd_vectorized(
    fn: Callable,
    x_s: np.ndarray,
    r: float,
    optimizer_name: str,
    lr: float,
    N: int,
    T_max: int,
    device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = len(x_s)
    x = torch.tensor(x_s, device=device).float()
    x = x + torch.randn(N, d, device=device) * config.PERTURBATION_STD
    x = x.detach().requires_grad_(True)
    x_s_t = torch.tensor(x_s, device=device).float()

    opt = make_optimizer(optimizer_name, [x], lr)

    escaped_min = torch.zeros(N, dtype=torch.bool, device=device)
    escaped_div = torch.zeros(N, dtype=torch.bool, device=device)
    escape_time = torch.full((N,), T_max, dtype=torch.float, device=device)
    active      = torch.ones(N, dtype=torch.bool, device=device)

    for t in range(T_max):
        if not active.any():
            break

        opt.zero_grad()
        vals = torch.stack([fn(x[i]) for i in range(N)])
        vals.sum().backward()

        with torch.no_grad():
            dist = torch.norm(x - x_s_t, dim=1)
            new_div = active & (dist > r)
            escaped_div[new_div] = True
            escape_time[new_div] = t
            active[new_div] = False

            grad_norm = x.grad.norm(dim=1)
            candidates = active & (grad_norm < config.GRAD_TOL)

        for idx in candidates.nonzero(as_tuple=True)[0]:
            xi_np = x[idx].detach().cpu().numpy()
            lmin = compute_lambda_min(fn, xi_np, device, d)
            if lmin > -config.EIGEN_TOL_MIN:
                escaped_min[idx] = True
                escape_time[idx] = t
                active[idx] = False

        opt.step()

    return escaped_min.cpu().numpy(), escaped_div.cpu().numpy(), escape_time.cpu().numpy()


def power_law_fit(dims: list[int], tau_vals: list[float]) -> float:
    """Fit tau ~ d^alpha; return alpha."""
    valid = [(d, t) for d, t in zip(dims, tau_vals)
             if t > 0 and not np.isnan(t) and d > 0]
    if len(valid) < 2:
        return np.nan
    log_d = np.log([v[0] for v in valid])
    log_t = np.log([v[1] for v in valid])
    alpha = np.polyfit(log_d, log_t, 1)[0]
    return float(alpha)


def run_part2(best_lrs: dict, fast: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)
    device = config.DEVICE

    dimensions = config.FAST_DIMENSIONS if fast else config.DIMENSIONS
    T_max = config.FAST_T_MAX if fast else config.T_MAX
    opt_names = list(config.OPTIMIZERS.keys())

    ckpt_path = Path('results/part2_checkpoint.pkl')
    completed = load_checkpoint(str(ckpt_path)) or {}
    rows = []

    # Functions: Rastrigin-nD, Styblinski-nD, Synthetic (k=1, k=n//4, k=n//2)
    def get_functions(d):
        fns = []
        fns.append(('Rastrigin-nD', rastrigin_nd, None))
        fns.append(('Styblinski-nD', styblinski_nd, None))
        for k_desc, k_fn in [('k=1', lambda d: 1),
                              ('k=n//4', lambda d: max(1, d // 4)),
                              ('k=n//2', lambda d: max(1, d // 2))]:
            fns.append((f'Synthetic_{k_desc}', None, k_fn))
        return fns

    # Collect all (function, d, optimizer) for power-law fit
    tau_by_opt_fn: dict[tuple[str, str], dict[int, float]] = {}

    for d in dimensions:
        N = config.FAST_N_TRIALS if fast else (200 if d <= 50 else 100)
        print(f"\n=== Dimension {d} (N={N}) ===")

        fn_list = get_functions(d)
        for fname, fn_base, k_fn in fn_list:
            # Determine function and saddle
            if k_fn is not None:
                k = k_fn(d)
                fn = synthetic_saddle(k)
                x_s, r, lmin = _find_nd_saddle_synthetic(k, d, device)
                saddle_index_k = k
            else:
                fn = fn_base
                # For nD Rastrigin/Styblinski the origin is a saddle (gradient=0,
                # mixed curvature at origin for Styblinski; for Rastrigin it's a
                # local min at origin but other points are saddles — we place
                # trials at a small perturbation of origin and use the origin).
                # Use origin as approximate saddle; compute lambda_min.
                x_s = np.zeros(d)
                lmin = compute_lambda_min(fn, x_s, device, d)
                r_max = 0.25 * math.sqrt(d * (2 * 5.12) ** 2) if 'Rastrigin' in fname \
                    else 0.25 * math.sqrt(d * 100)
                r = min(r_max, 0.5 / math.sqrt(abs(lmin) + 1e-6))
                saddle_index_k = -1

                # Skip if not a saddle
                if lmin >= 0:
                    print(f"  {fname} d={d}: origin is not a saddle (lmin={lmin:.3f}), skip")
                    continue

            for opt_name in opt_names:
                lr = best_lrs.get(opt_name, 0.01)
                key = (fname, d, opt_name, saddle_index_k)

                if key in completed:
                    rows.append(completed[key])
                    tau_by_opt_fn.setdefault((opt_name, fname), {})[d] = completed[key]['tau_mean']
                    continue

                print(f"  {fname} d={d} k={saddle_index_k} {opt_name} lr={lr}", end='', flush=True)
                esc_min, esc_div, esc_t = run_trials_nd_vectorized(
                    fn, x_s, r, opt_name, lr, N, T_max, device
                )
                metrics = compute_see(esc_min, esc_div, esc_t, T_max,
                                      n_resamples=config.BOOTSTRAP_RESAMPLES)
                print(f"  SEE_basic={metrics['SEE_basic']:.4f}")

                row = {
                    'function':         fname,
                    'optimizer':        opt_name,
                    'lr_best':          lr,
                    'dimension':        d,
                    'saddle_index_k':   saddle_index_k,
                    'SEE_basic':        metrics['SEE_basic'],
                    'SEE_quality':      metrics['SEE_quality'],
                    'CI_lo':            metrics['SEE_basic_CI_lo'],
                    'CI_hi':            metrics['SEE_basic_CI_hi'],
                    'tau_mean':         metrics['tau_mean'],
                    'tau_std':          metrics['tau_std'],
                    'escape_min_pct':   metrics['escape_min_pct'],
                    'stuck_pct':        metrics['stuck_pct'],
                    'power_law_alpha':  np.nan,  # filled below
                }
                completed[key] = row
                rows.append(row)
                tau_by_opt_fn.setdefault((opt_name, fname), {})[d] = metrics['tau_mean']

        save_checkpoint(completed, str(ckpt_path))

    df = pd.DataFrame(rows)

    # --- Power law fit ---
    for (opt_name, fname), dim_tau in tau_by_opt_fn.items():
        sorted_items = sorted(dim_tau.items())
        dims_used = [x[0] for x in sorted_items]
        taus_used = [x[1] for x in sorted_items]
        alpha = power_law_fit(dims_used, taus_used)
        mask = (df['optimizer'] == opt_name) & (df['function'] == fname)
        df.loc[mask, 'power_law_alpha'] = alpha

    out = Path('results/part2.csv')
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPart 2 results saved to {out}")
    return df
