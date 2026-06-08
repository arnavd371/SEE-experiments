"""
Part 3: Neural network experiments on real datasets.
Detects saddle points during training via Lanczos eigenvalue computation.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config
from src.data.loaders import load_moons, load_mnist_binary, load_housing
from src.models.mlp import MLP
from src.metrics.see import compute_see
from src.metrics.bootstrap import bootstrap_ci
from src.utils.seeding import set_all_seeds
from src.utils.checkpointing import save_checkpoint, load_checkpoint
from src.experiments.part1_vectorized import make_optimizer

try:
    import scipy.sparse.linalg as spla
    HAS_SCIPY_SPARSE = True
except ImportError:
    HAS_SCIPY_SPARSE = False


def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.view(-1) for p in model.parameters()])


def _set_flat_params(model: nn.Module, flat: torch.Tensor):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].view(p.shape))
        offset += n


def _grad_flat(loss: torch.Tensor, model: nn.Module) -> torch.Tensor:
    grads = torch.autograd.grad(loss, model.parameters(),
                                create_graph=False, retain_graph=True)
    return torch.cat([g.view(-1) for g in grads])


def _lanczos_lambda_min_nn(
    model: nn.Module,
    loss_fn: callable,
    X: torch.Tensor,
    y: torch.Tensor,
    k: int = 6,
) -> tuple[float, float]:
    """Hessian-vector product via autograd; returns (lambda_min, lambda_max)."""
    params = list(model.parameters())
    d = sum(p.numel() for p in params)

    # Compute gradient with create_graph for HVP
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.view(-1) for g in grads])

    def hvp(vec_np: np.ndarray) -> np.ndarray:
        vec = torch.tensor(vec_np, dtype=torch.float32, device=X.device)
        vTg = (grad_flat * vec).sum()
        hv = torch.autograd.grad(vTg, params, retain_graph=True, allow_unused=True)
        return torch.cat([
            h.view(-1) if h is not None else torch.zeros(p.numel(), device=X.device)
            for h, p in zip(hv, params)
        ]).detach().cpu().numpy().astype(np.float64)

    op = spla.LinearOperator((d, d), matvec=hvp)
    k_eff = min(k, d - 1)
    try:
        eigs_lo = spla.eigsh(op, k=k_eff, which='SA', return_eigenvectors=False,
                              tol=1e-2, maxiter=d * 5)
        eigs_hi = spla.eigsh(op, k=k_eff, which='LA', return_eigenvectors=False,
                              tol=1e-2, maxiter=d * 5)
        return float(eigs_lo.min()), float(eigs_hi.max())
    except Exception:
        return np.nan, np.nan


def _run_sub_trials(
    model: nn.Module,
    theta_flat: torch.Tensor,
    loss_fn: callable,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer_name: str,
    lr: float,
    r: float,
    N: int,
    T_max: int,
    device,
) -> dict:
    """
    Launch N sub-trials from N(theta_current, 0.01²I).
    Each trial is a fresh model copy; we vectorize over N copies
    by running them sequentially (N=50 is small enough).
    """
    d = theta_flat.numel()
    x_s_np = theta_flat.cpu().numpy()

    escaped_min = np.zeros(N, dtype=bool)
    escaped_div = np.zeros(N, dtype=bool)
    escape_time = np.full(N, T_max, dtype=float)

    for trial_i in range(N):
        # Perturb
        perturb = torch.randn(d, device=device) * 0.01
        theta_trial = theta_flat + perturb

        m = MLP(X.shape[1]).to(device)
        _set_flat_params(m, theta_trial)

        opt = make_optimizer(optimizer_name, m.parameters(), lr)
        x_s_t = torch.tensor(x_s_np, dtype=torch.float32, device=device)

        for t in range(T_max):
            opt.zero_grad()
            out = m(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            with torch.no_grad():
                theta_now = _flat_params(m)
                dist = torch.norm(theta_now - x_s_t).item()
                if dist > r:
                    escaped_div[trial_i] = True
                    escape_time[trial_i] = t
                    break

                grad_norms = [p.grad.norm().item() for p in m.parameters() if p.grad is not None]
                gnorm = math.sqrt(sum(g ** 2 for g in grad_norms))
                if gnorm < config.GRAD_TOL:
                    escaped_min[trial_i] = True
                    escape_time[trial_i] = t
                    break

    return compute_see(escaped_min, escaped_div, escape_time, T_max,
                       n_resamples=min(200, config.BOOTSTRAP_RESAMPLES))


TASKS = {
    'Moons':   {'loader': load_moons,        'in_dim': 2,   'loss': nn.BCEWithLogitsLoss(), 'conv_thresh': 0.05},
    'MNIST':   {'loader': load_mnist_binary, 'in_dim': 784, 'loss': nn.BCEWithLogitsLoss(), 'conv_thresh': 0.10},
    'Housing': {'loader': load_housing,      'in_dim': 8,   'loss': nn.MSELoss(),           'conv_thresh': 0.02},
}


def run_part3(best_lrs: dict, fast: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)
    device = config.DEVICE

    n_steps     = config.FAST_NN_STEPS if fast else config.NN_STEPS
    n_runs      = config.FAST_NN_RUNS  if fast else config.NN_RUNS
    batch_size  = config.NN_BATCH_SIZE
    check_every = config.NN_CHECK_INTERVAL

    ckpt_path = Path('results/part3_checkpoint.pkl')
    completed = load_checkpoint(str(ckpt_path)) or {}
    all_rows = []

    for task_name, task_cfg in TASKS.items():
        loader   = task_cfg['loader']
        in_dim   = task_cfg['in_dim']
        loss_fn  = task_cfg['loss'].to(device)
        conv_thr = task_cfg['conv_thresh']

        print(f"\n=== Task: {task_name} ===")
        X, y = loader(device)

        for opt_name in config.OPTIMIZERS:
            lr = best_lrs.get(opt_name, 0.01)

            for seed in range(n_runs):
                key = (task_name, opt_name, seed)
                if key in completed:
                    all_rows.extend(completed[key])
                    continue

                print(f"  {task_name} {opt_name} seed={seed}", flush=True)
                set_all_seeds(config.GLOBAL_SEED + seed * 1000)

                model = MLP(in_dim).to(device)
                opt = make_optimizer(opt_name, model.parameters(), lr)

                run_rows = []
                steps_to_conv = n_steps
                final_loss = np.nan
                final_acc  = np.nan

                # Sub-trial radius: use parameter-space norm heuristic
                d_params = sum(p.numel() for p in model.parameters())
                r_nn = 0.5 / math.sqrt(2.0 + 1e-6)  # same formula as synthetic saddle

                for step in range(n_steps):
                    # Mini-batch
                    idx = torch.randint(0, X.shape[0], (batch_size,), device=device)
                    X_b, y_b = X[idx], y[idx]

                    opt.zero_grad()
                    out = model(X_b)
                    loss = loss_fn(out, y_b)
                    loss.backward()
                    opt.step()

                    loss_val = loss.item()

                    # Convergence check
                    if loss_val < conv_thr and steps_to_conv == n_steps:
                        steps_to_conv = step

                    row_base = {
                        'task': task_name, 'optimizer': opt_name,
                        'run_seed': seed, 'step': step,
                        'loss': loss_val,
                        'grad_norm': np.nan, 'is_saddle': False,
                        'SEE_NN': np.nan, 'SEE_NN_CI_lo': np.nan, 'SEE_NN_CI_hi': np.nan,
                        'lambda_min': np.nan, 'lambda_max': np.nan,
                        'steps_to_convergence': steps_to_conv,
                        'final_loss': np.nan, 'final_accuracy': np.nan,
                    }

                    # Saddle detection every check_every steps
                    if step % check_every == 0:
                        with torch.no_grad():
                            out_full = model(X)
                            loss_full = loss_fn(out_full, y).item()

                        # Compute gradient over full batch
                        model.zero_grad()
                        out_full = model(X)
                        loss_full_t = loss_fn(out_full, y)
                        loss_full_t.backward()
                        g_norms = [p.grad.norm().item() for p in model.parameters()
                                   if p.grad is not None]
                        gnorm = math.sqrt(sum(g ** 2 for g in g_norms))
                        row_base['grad_norm'] = gnorm

                        if gnorm < config.NN_GRAD_TOL:
                            lmin, lmax = _lanczos_lambda_min_nn(model, loss_fn, X, y)
                            row_base['lambda_min'] = lmin
                            row_base['lambda_max'] = lmax

                            if not np.isnan(lmin) and lmin < -config.NN_LAMBDA_TOL:
                                row_base['is_saddle'] = True
                                # Launch sub-trials
                                theta_flat = _flat_params(model)
                                N_sub = config.NN_SUBTRIAL_N
                                T_sub = config.NN_SUBTRIAL_T_MAX
                                see_metrics = _run_sub_trials(
                                    model, theta_flat, loss_fn, X, y,
                                    opt_name, lr, r_nn, N_sub, T_sub, device
                                )
                                row_base['SEE_NN']      = see_metrics['SEE_basic']
                                row_base['SEE_NN_CI_lo'] = see_metrics['SEE_basic_CI_lo']
                                row_base['SEE_NN_CI_hi'] = see_metrics['SEE_basic_CI_hi']

                    run_rows.append(row_base)

                # Final metrics
                with torch.no_grad():
                    out_all = model(X)
                    fl = loss_fn(out_all, y).item()
                if task_name in ('Moons', 'MNIST'):
                    preds = (out_all > 0).float()
                    fa = (preds == y).float().mean().item()
                else:
                    fa = np.nan

                for r in run_rows:
                    r['final_loss'] = fl
                    r['final_accuracy'] = fa
                    r['steps_to_convergence'] = steps_to_conv

                completed[key] = run_rows
                all_rows.extend(run_rows)
                save_checkpoint(completed, str(ckpt_path))
                print(f"    final_loss={fl:.4f}  steps_to_conv={steps_to_conv}")

    df = pd.DataFrame(all_rows)
    out = Path('results/part3.csv')
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nPart 3 results saved to {out}")
    return df
