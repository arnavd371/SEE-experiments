"""
Part 4: LLM Training Proxy — Tiny GPT on Wikitext-2.

Gradient norm plateau detection is used as a saddle proxy.
(Full Hessian infeasible at ~3M params — this is an approximation, labeled as such.)

Plateau event: rolling_grad_norm < 0.1 for 3 consecutive 100-step checks.
Results saved to results/part4_results.csv.
"""

import os
import sys
import math
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import config
from src.utils.seeding import set_all_seeds
from src.utils.logging_utils import PART4_FIELDS
from src.data.loaders import load_wikitext2
from src.models.tiny_gpt import TinyGPT
from src.optimizers.wrapper import make_optimizer, cosine_warmup_scheduler


def load_best_lrs() -> dict:
    if not os.path.exists(config.BEST_LRS_PATH):
        return {opt: 0.001 for opt in config.OPTIMIZER_NAMES}
    with open(config.BEST_LRS_PATH) as f:
        return yaml.safe_load(f)


def run_one_llm_run(opt_name: str, lr: float, run_seed: int,
                    data: dict, device: torch.device,
                    n_steps: int, fast: bool = False) -> list:
    """
    Run one LLM training run.
    Returns list of row dicts for part4_results.csv.
    """
    set_all_seeds(run_seed)

    model = TinyGPT().to(device)
    print(f'    Model params: {model.n_params():,}')

    opt = make_optimizer(opt_name, list(model.parameters()), lr)
    scheduler = cosine_warmup_scheduler(opt, config.LLM_WARMUP_STEPS, n_steps)

    train_loader = data['train_loader']
    val_loader = data['val_loader']

    # Plateau detection state
    grad_norm_window = collections.deque(maxlen=config.LLM_PLATEAU_WINDOW)
    plateau_check_history = collections.deque(maxlen=config.LLM_PLATEAU_CONSECUTIVE)
    plateau_event_id = 0
    in_plateau = False
    total_plateau_steps = 0

    steps_to_ppl_threshold = n_steps  # sentinel
    final_val_ppl = float('nan')

    rows = []
    data_iter = iter(train_loader)

    for step in range(1, n_steps + 1):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)

        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        logits = model(xb)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), yb.view(B * T))

        if not torch.isfinite(loss):
            print(f'    NaN loss at step {step}; stopping run.')
            break

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.LLM_GRAD_CLIP)

        # Compute grad norm before step (after clipping)
        grad_norm = math.sqrt(sum(
            p.grad.pow(2).sum().item()
            for p in model.parameters() if p.grad is not None
        ))
        grad_norm_window.append(grad_norm)

        opt.step()
        scheduler.step()

        train_loss = loss.item()

        # ── Plateau detection every LLM_PLATEAU_CHECK_INTERVAL steps ──────────
        is_plateau_step = False
        current_plateau_id = 0

        if step % config.LLM_PLATEAU_CHECK_INTERVAL == 0:
            rolling_grad_norm = np.mean(list(grad_norm_window)) if grad_norm_window else grad_norm
            below_threshold = (rolling_grad_norm < config.LLM_PLATEAU_GRAD_TOL)
            plateau_check_history.append(below_threshold)

            if (len(plateau_check_history) == config.LLM_PLATEAU_CONSECUTIVE
                    and all(plateau_check_history)):
                if not in_plateau:
                    in_plateau = True
                    plateau_event_id += 1
            else:
                in_plateau = False

        if in_plateau:
            is_plateau_step = True
            total_plateau_steps += 1
            current_plateau_id = plateau_event_id

        # ── Validation perplexity every 100 steps ─────────────────────────────
        val_ppl = float('nan')
        if step % config.LLM_PLATEAU_CHECK_INTERVAL == 0:
            val_ppl = model.compute_perplexity(val_loader, device)
            if val_ppl < config.LLM_PPL_THRESHOLD and steps_to_ppl_threshold == n_steps:
                steps_to_ppl_threshold = step
            final_val_ppl = val_ppl

        rows.append({
            'optimizer': opt_name,
            'run_seed': run_seed,
            'step': step,
            'train_loss': train_loss,
            'val_ppl': val_ppl,
            'grad_norm': grad_norm,
            'is_plateau_step': int(is_plateau_step),
            'plateau_event_id': current_plateau_id,
            'steps_to_ppl_threshold': steps_to_ppl_threshold,
            'final_val_ppl': final_val_ppl,
            'total_plateau_steps': total_plateau_steps,
            'plateau_fraction': total_plateau_steps / step,
        })

    # Update all rows with final values
    for row in rows:
        row['steps_to_ppl_threshold'] = steps_to_ppl_threshold
        row['final_val_ppl'] = final_val_ppl
        row['total_plateau_steps'] = total_plateau_steps
        row['plateau_fraction'] = total_plateau_steps / max(len(rows), 1)

    return rows


def run_part4(fast: bool = False) -> pd.DataFrame:
    set_all_seeds(config.GLOBAL_SEED)
    device = config.DEVICE
    best_lrs = load_best_lrs()

    n_steps = config.FAST_GPT_STEPS if fast else config.LLM_STEPS
    n_runs = config.FAST_LLM_RUNS if fast else config.LLM_RUNS

    print('Loading Wikitext-2...')
    try:
        data = load_wikitext2(device=device)
        print(f'  Train chunks: {len(data["train_x"])}, Val chunks: {len(data["val_x"])}')
    except Exception as e:
        print(f'  WARNING: Could not load Wikitext-2: {e}')
        print('  Using synthetic data for structure verification.')
        data = _make_synthetic_wikitext(device)

    all_rows = []
    out_csv = os.path.join(config.RESULTS_DIR, 'part4_results.csv')

    for opt_name in config.OPTIMIZER_NAMES:
        lr = best_lrs.get(opt_name, 0.001)
        print(f'\n── Optimizer: {opt_name}, lr={lr} ──')
        for run_idx in range(n_runs):
            seed = config.GLOBAL_SEED + run_idx
            print(f'  Run {run_idx + 1}/{n_runs} (seed={seed}, steps={n_steps})...')
            try:
                rows = run_one_llm_run(opt_name, lr, seed, data, device, n_steps, fast)
                all_rows.extend(rows)
                last = rows[-1] if rows else {}
                print(f'  Final val_ppl={last.get("final_val_ppl", "?"):.1f}, '
                      f'plateau_fraction={last.get("plateau_fraction", 0):.3f}')
            except Exception as e:
                print(f'  ERROR: {e}')
                continue

    df = pd.DataFrame(all_rows, columns=PART4_FIELDS)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f'\nPart 4 results saved to {out_csv}')
    return df


def _make_synthetic_wikitext(device: torch.device) -> dict:
    """Fallback synthetic data when Wikitext-2 is unavailable."""
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    ctx = config.GPT_CONTEXT_LENGTH
    n_train = 2000
    n_val = 200

    train_x = torch.randint(0, config.GPT_VOCAB_SIZE, (n_train, ctx))
    train_y = torch.randint(0, config.GPT_VOCAB_SIZE, (n_train, ctx))
    val_x = torch.randint(0, config.GPT_VOCAB_SIZE, (n_val, ctx))
    val_y = torch.randint(0, config.GPT_VOCAB_SIZE, (n_val, ctx))

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=config.LLM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y),
                            batch_size=config.LLM_BATCH_SIZE, shuffle=False)
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_x': train_x, 'train_y': train_y,
        'val_x': val_x, 'val_y': val_y,
    }
