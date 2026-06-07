"""
Part 3: Neural Network Training on Real Datasets.

Three tasks: Moons, MNIST-3v8, CalHousing.
Saddle detection every 25 steps using gradient norm + Lanczos.
Sub-trials at detected saddles compute SEE_NN.
Uses best_lrs from Part 1.
Results saved to results/part3_results.csv.
"""

import os
import sys
import copy
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.seeding import set_all_seeds
from src.utils.logging_utils import PART3_FIELDS
from src.data.loaders import load_all_nn_tasks
from src.models.mlp import MLP, make_loss_fn
from src.optimizers.wrapper import make_optimizer
from src.metrics.hessian import model_min_eigenvalue
from src.metrics.see import compute_see
from src.metrics.bootstrap import bootstrap_ci


def load_best_lrs() -> dict:
    if not os.path.exists(config.BEST_LRS_PATH):
        return {opt: 0.01 for opt in config.OPTIMIZER_NAMES}
    with open(config.BEST_LRS_PATH) as f:
        return yaml.safe_load(f)


def _compute_full_grad_norm(model: MLP, loss_fn, data, device: torch.device) -> float:
    """Gradient norm over full training set."""
    X, y = data
    model.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    total = sum(p.grad.pow(2).sum().item()
                for p in model.parameters() if p.grad is not None)
    model.zero_grad()
    return math.sqrt(total)


def _run_sub_trials(model: MLP, loss_fn, data, device: torch.device,
                    opt_name: str, lr: float,
                    task_info: dict) -> dict:
    """
    Launch NN_SUBTRIAL_N sub-trials from N(θ_current, NN_PERTURBATION_STD²·I).
    Returns SEE_NN metrics dict.
    """
    params = model.state_dict()
    conv_loss = task_info['conv_loss']
    task_type = task_info['task_type']
    X, y = data

    outcomes = []
    taus = []

    for i in range(config.NN_SUBTRIAL_N):
        # Perturbed copy
        sub_model = copy.deepcopy(model)
        with torch.no_grad():
            for p in sub_model.parameters():
                p.add_(torch.randn_like(p) * config.NN_PERTURBATION_STD)
        sub_model.to(device)

        sub_opt = make_optimizer(opt_name, list(sub_model.parameters()), lr)

        # Diverge radius: 10 * ||flat_params||_2 + 50
        flat_ref = torch.cat([p.detach().reshape(-1)
                               for p in model.parameters()])
        diverge_radius = 10.0 * float(flat_ref.norm()) + 50.0

        outcome = 'stuck'
        tau = config.NN_SUBTRIAL_T_MAX

        for t in range(1, config.NN_SUBTRIAL_T_MAX + 1):
            sub_opt.zero_grad()
            out = sub_model(X)
            loss = loss_fn(out, y)
            if not torch.isfinite(loss):
                outcome = 'diverge'
                tau = t
                break
            loss.backward()
            sub_opt.step()

            with torch.no_grad():
                flat = torch.cat([p.reshape(-1) for p in sub_model.parameters()])
                if float(flat.norm()) > diverge_radius:
                    outcome = 'diverge'
                    tau = t
                    break
                if loss.item() < conv_loss:
                    outcome = 'local_min'
                    tau = t
                    break

        outcomes.append(outcome)
        taus.append(tau)

    outcomes_arr = np.array(outcomes, dtype=object)
    taus_arr = np.array(taus, dtype=float)
    see_metrics = compute_see(outcomes_arr, taus_arr,
                              n_resamples=config.FAST_BOOTSTRAP)  # faster for sub-trials
    return see_metrics


def run_one_task_optimizer(task_name: str, task_info: dict,  # noqa: C901
                           opt_name: str, lr: float,
                           run_seed: int, device: torch.device,
                           fast: bool = False) -> list:
    """
    Run one (task, optimizer, seed) training run.
    Returns list of row dicts for part3_results.csv.
    """
    set_all_seeds(run_seed)

    input_dim = task_info['input_dim']
    task_type = task_info['task_type']
    conv_loss = task_info['conv_loss']
    X, y = task_info['full_data']
    loader = task_info['loader']

    model = MLP(input_dim=input_dim).to(device)
    loss_fn = make_loss_fn(task_type)
    opt = make_optimizer(opt_name, list(model.parameters()), lr)

    n_steps = config.FAST_NN_STEPS if fast else config.NN_STEPS
    steps_to_convergence = n_steps  # updated when convergence detected
    final_loss = float('nan')
    final_accuracy = float('nan')

    # Saddle tracking
    saddle_check_interval = config.NN_SADDLE_CHECK_INTERVAL
    saddle_events = []  # list of step indices where saddle was detected

    rows = []
    data_iter = iter(loader)

    for step in range(1, n_steps + 1):
        # Get batch
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            xb, yb = next(data_iter)
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        out = model(xb)
        loss = loss_fn(out, yb)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()

        current_loss = loss.item()
        final_loss = current_loss

        # Check convergence
        if current_loss < conv_loss and steps_to_convergence == n_steps:
            steps_to_convergence = step

        # Compute grad_norm over full training set every check_interval steps
        is_saddle_step = False
        see_nn = see_ci_lo = see_ci_hi = float('nan')
        min_eig = max_eig = float('nan')

        if step % saddle_check_interval == 0:
            with torch.no_grad():
                out_full = model(X)
                full_loss = loss_fn(out_full, y)
            # Grad norm requires grad
            model.zero_grad()
            out_full2 = model(X)
            full_loss2 = loss_fn(out_full2, y)
            full_loss2.backward()
            grad_norm = math.sqrt(sum(
                p.grad.pow(2).sum().item()
                for p in model.parameters() if p.grad is not None
            ))
            model.zero_grad()

            if grad_norm < config.NN_SADDLE_GRAD_TOL:
                if fast:
                    # In fast mode: skip Lanczos, treat grad-norm plateau as saddle
                    is_saddle_step = True
                    min_eig = float('nan')
                    max_eig = float('nan')
                    saddle_events.append(step)
                    see_nn = 0.0
                    see_ci_lo = see_ci_hi = float('nan')
                else:
                    # Compute top-6 eigenvalues via Lanczos
                    model.zero_grad()
                    out_check = model(X)
                    loss_check = loss_fn(out_check, y)
                    loss_check.backward()
                    model.zero_grad()

                    min_eig, max_eig = model_min_eigenvalue(
                        model, loss_fn, (X, y), device,
                        k=config.NN_LANCZOS_K
                    )

                    if math.isfinite(min_eig) and min_eig < config.NN_SADDLE_EIGEN_THRESH:
                        is_saddle_step = True
                        saddle_events.append(step)

                        see_sub = _run_sub_trials(
                            model, loss_fn, (X, y), device,
                            opt_name, lr, task_info
                        )
                        see_nn = see_sub['SEE_quality']
                        see_ci_lo = see_sub['SEE_quality_CI_lo']
                        see_ci_hi = see_sub['SEE_quality_CI_hi']
        else:
            # Compute batch grad norm for logging
            with torch.no_grad():
                grad_norm = math.sqrt(sum(
                    p.grad.pow(2).sum().item()
                    for p in model.parameters() if p.grad is not None
                )) if any(p.grad is not None for p in model.parameters()) else 0.0

        rows.append({
            'task': task_name,
            'optimizer': opt_name,
            'run_seed': run_seed,
            'step': step,
            'loss': current_loss,
            'grad_norm': grad_norm,
            'is_saddle_step': int(is_saddle_step),
            'SEE_NN': see_nn,
            'SEE_NN_CI_lo': see_ci_lo,
            'SEE_NN_CI_hi': see_ci_hi,
            'min_eigenvalue': min_eig,
            'max_eigenvalue': max_eig,
            'steps_to_convergence': steps_to_convergence,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
        })

    # Compute final accuracy for classification
    if task_type == 'binary':
        with torch.no_grad():
            out_final = model(X)
            preds = (out_final > 0).float()
            final_accuracy = float((preds == y).float().mean())
        for row in rows:
            row['final_accuracy'] = final_accuracy

    return rows


def run_part3(fast: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)
    device = config.DEVICE
    best_lrs = load_best_lrs()

    print(f'Loading real datasets (device={device})...')
    tasks = load_all_nn_tasks(device)

    n_runs = config.FAST_NN_RUNS if fast else config.NN_RUNS
    all_rows = []
    out_csv = os.path.join(config.RESULTS_DIR, 'part3_results.csv')

    for task_name, task_info in tasks.items():
        print(f'\n── Task: {task_name} ──')
        for opt_name in config.OPTIMIZER_NAMES:
            lr = best_lrs.get(opt_name, 0.01)
            print(f'  Optimizer: {opt_name}, lr={lr}')
            for run_idx in range(n_runs):
                seed = config.GLOBAL_SEED + run_idx
                print(f'    Run {run_idx + 1}/{n_runs} (seed={seed})...')
                try:
                    rows = run_one_task_optimizer(
                        task_name, task_info, opt_name, lr, seed, device, fast
                    )
                    all_rows.extend(rows)
                except Exception as e:
                    print(f'    ERROR: {e}')
                    continue

    df = pd.DataFrame(all_rows, columns=PART3_FIELDS)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nPart 3 results saved to {out_csv}')
    return df
