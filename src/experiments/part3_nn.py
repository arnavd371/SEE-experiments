"""
Part 3: Neural-network SEE experiments (real data only).

For each task × optimizer × seed:
  - Train 2000 steps, batch 64.
  - Every 25 steps: if full-batch ||grad|| < 0.05, run Lanczos top-6 eigs.
  - If lambda_min < -0.01 → SADDLE EVENT → launch 50 sub-trials.
  - Compute SEE_NN from sub-trials (same two-phase criterion, param-space r).
"""
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from scipy.sparse.linalg import eigsh, LinearOperator

from config import Config
from src.data.loaders import TASK_LOADERS, CONVERGENCE_TARGETS
from src.models.mlp import MLP
from src.metrics.see import compute_see
from src.utils.seeding import set_all_seeds
from src.experiments.part1_vectorized import BatchedOptimizer


# ── Flat param helpers ────────────────────────────────────────────────────────

def _flat_params(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.reshape(-1) for p in model.parameters()])


def _load_flat(model: nn.Module, flat: torch.Tensor):
    offset = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[offset:offset + n].reshape(p.shape))
        offset += n


# ── Lanczos lambda_min for neural-network Hessian ────────────────────────────

def _nn_hvp(model: nn.Module, loss_fn, X: torch.Tensor, y: torch.Tensor,
            v_flat: torch.Tensor) -> torch.Tensor:
    """Full-batch Hessian-vector product for model parameters."""
    for p in model.parameters():
        p.grad = None
    out   = model(X)
    loss  = loss_fn(out, y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    g_flat = torch.cat([g.reshape(-1) for g in grads])
    hvp   = torch.autograd.grad((g_flat * v_flat).sum(), model.parameters(),
                                 retain_graph=False)
    return torch.cat([h.reshape(-1) for h in hvp]).detach()


def _nn_lambda_min(model: nn.Module, loss_fn,
                   X: torch.Tensor, y: torch.Tensor,
                   k_eig: int = 6) -> float:
    d = sum(p.numel() for p in model.parameters())

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=torch.float32).to(X.device)
        hv = _nn_hvp(model, loss_fn, X, y, v)
        return hv.cpu().numpy().astype(np.float64)

    A  = LinearOperator((d, d), matvec=matvec)
    k  = min(k_eig, d - 1) if d > 2 else 1
    try:
        vals = eigsh(A, k=k, which="SA", return_eigenvectors=False,
                     maxiter=d * 10, tol=1e-3)
        return float(vals.min())
    except Exception:
        return float("nan")


# ── Sub-trial runner for SEE_NN ───────────────────────────────────────────────

def _run_nn_subtrials(theta_saddle: torch.Tensor,
                      model_template: nn.Module,
                      loss_fn, X: torch.Tensor, y: torch.Tensor,
                      optimizer_name: str, lr: float,
                      lambda_min: float,
                      N_sub: int, T_sub: int, perturb_std: float,
                      device: str) -> dict:
    d = theta_saddle.numel()
    param_norm = theta_saddle.norm().item()
    diam       = max(1.0, 2.0 * param_norm)
    r_esc      = min(0.25 * diam, 0.5 / math.sqrt(abs(lambda_min) + 1e-6))
    r_div      = 0.5 * diam

    escaped_ever  = np.zeros(N_sub, dtype=bool)
    quality_ever  = np.zeros(N_sub, dtype=bool)
    diverged_ever = np.zeros(N_sub, dtype=bool)
    escape_step   = np.zeros(N_sub, dtype=float)
    quality_step  = np.zeros(N_sub, dtype=float)

    for trial in range(N_sub):
        theta = (theta_saddle
                 + perturb_std * torch.randn_like(theta_saddle)).to(device)
        opt_state = BatchedOptimizer(optimizer_name, lr, 1, d, device)

        for step in range(T_sub):
            # Load params and compute full-batch loss + grad
            _load_flat(model_template, theta)
            for p in model_template.parameters():
                p.grad = None

            out  = model_template(X)
            loss = loss_fn(out, y)
            loss.backward()
            g = torch.cat([p.grad.reshape(-1) for p in model_template.parameters()]).detach()
            g = torch.nan_to_num(g, nan=0.0, posinf=1e3, neginf=-1e3)

            with torch.no_grad():
                dist = (theta - theta_saddle).norm().item()
                gn   = g.norm().item()

                if not quality_ever[trial] and gn < Config.GRAD_NORM_THRESH:
                    try:
                        lm = _nn_lambda_min(model_template, loss_fn, X, y, k_eig=3)
                        if (not math.isnan(lm)) and lm > Config.LAMBDA_MIN_THRESH:
                            quality_ever[trial]  = True
                            quality_step[trial]  = float(step + 1)
                    except Exception:
                        pass

                if not escaped_ever[trial] and dist > r_esc:
                    escaped_ever[trial] = True
                    escape_step[trial]  = float(step + 1)

                if dist > r_div:
                    diverged_ever[trial] = True

                # Update
                g_2d  = g.unsqueeze(0)           # (1, d)
                delta = opt_state.step(g_2d)[0]  # (d,)
                if optimizer_name == "AdamW":
                    theta = theta * (1.0 - lr * opt_state.wd) - delta
                else:
                    theta = theta - delta

    esc_s_m = np.where(escaped_ever, escape_step, np.nan)
    qlt_s_m = np.where(quality_ever, quality_step, np.nan)
    return compute_see(escaped_ever, esc_s_m, quality_ever, qlt_s_m, diverged_ever,
                       n_bootstrap=min(200, Config.N_BOOTSTRAP))


# ── Part-3 driver ─────────────────────────────────────────────────────────────

def run_part3(cfg: type, results_dir: Path,
              best_lrs: dict | None = None) -> pd.DataFrame:
    device = cfg.DEVICE
    rows   = []
    training_curves = {}   # (task, opt_name, seed) -> {"loss": [...], "grad": [...], "saddle_steps": [...]}

    if best_lrs is None:
        import yaml
        lr_file = results_dir / "best_lrs.yaml"
        if lr_file.exists():
            with open(lr_file) as fh:
                best_lrs = yaml.safe_load(fh)
        else:
            best_lrs = {opt: 0.01 for opt in cfg.OPTIMIZERS}

    for task_name, loader_fn in TASK_LOADERS.items():
        print(f"  Task: {task_name}")
        conv_target = CONVERGENCE_TARGETS[task_name]

        try:
            tr_loader, _, in_dim, out_dim, loss_type = loader_fn(seed=42, batch=cfg.NN_BATCH)
        except Exception as e:
            print(f"    SKIP {task_name}: {e}")
            continue

        # Full dataset for gradient/Hessian checks
        Xf, yf = zip(*[(X, y) for X, y in tr_loader])
        Xf = torch.cat(Xf).to(device)
        yf = torch.cat(yf).to(device)

        for opt_name in cfg.OPTIMIZERS:
            lr = best_lrs.get(opt_name, 0.01)

            for seed in cfg.NN_SEEDS:
                set_all_seeds(seed)

                model = MLP(in_dim, out_dim).to(device)
                if loss_type == "bce":
                    loss_fn = nn.BCEWithLogitsLoss()
                else:
                    loss_fn = nn.MSELoss()

                # PyTorch optimizer for training
                if opt_name == "GD_fixed":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
                elif opt_name == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                elif opt_name == "AdamW":
                    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
                elif opt_name == "RMSProp":
                    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
                elif opt_name == "AdaGrad":
                    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
                elif opt_name == "SGD_momentum":
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

                loss_curve  = []
                grad_curve  = []
                saddle_steps = []
                see_nn_list  = []
                converged_step = None

                X_iter  = iter(tr_loader)
                step_ok = True

                for step in range(cfg.NN_STEPS):
                    try:
                        Xb, yb = next(X_iter)
                    except StopIteration:
                        X_iter = iter(tr_loader)
                        Xb, yb = next(X_iter)
                    Xb, yb = Xb.to(device), yb.to(device)

                    optimizer.zero_grad()
                    out  = model(Xb)
                    loss = loss_fn(out, yb)
                    loss.backward()
                    optimizer.step()

                    loss_val = loss.item()
                    loss_curve.append(loss_val)

                    if converged_step is None and loss_val < conv_target:
                        converged_step = step + 1

                    # Every 25 steps: full-batch gradient check
                    if (step + 1) % cfg.NN_GRAD_CHECK_INTERVAL == 0:
                        for p in model.parameters():
                            p.grad = None
                        out_f  = model(Xf)
                        loss_f = loss_fn(out_f, yf)
                        loss_f.backward()
                        g_flat = torch.cat([p.grad.reshape(-1)
                                            for p in model.parameters()]).detach()
                        gn = g_flat.norm().item()
                        grad_curve.append((step + 1, gn))

                        if gn < cfg.NN_GRAD_NORM_THRESH:
                            lmin = _nn_lambda_min(model, loss_fn, Xf, yf,
                                                  k_eig=cfg.LANCZOS_K)
                            if (not math.isnan(lmin)) and lmin < cfg.NN_SADDLE_LAMBDA_THRESH:
                                print(f"    SADDLE EVENT at step {step+1}, "
                                      f"λ_min={lmin:.4f}")
                                saddle_steps.append(step + 1)
                                theta_sad = _flat_params(model).detach().clone()
                                see_r = _run_nn_subtrials(
                                    theta_sad, model, loss_fn, Xf, yf,
                                    opt_name, lr, lmin,
                                    cfg.NN_SUB_TRIALS, cfg.NN_SUB_T_MAX,
                                    cfg.NN_PERTURB_STD, device,
                                )
                                see_nn_list.append(see_r)

                final_loss = loss_curve[-1] if loss_curve else float("nan")
                n_saddle   = len(saddle_steps)
                see_basic_nn  = float(np.mean([r["SEE_basic"]   for r in see_nn_list])) if see_nn_list else float("nan")
                see_quality_nn= float(np.mean([r["SEE_quality"] for r in see_nn_list])) if see_nn_list else float("nan")

                rows.append({
                    "task":               task_name,
                    "optimizer":          opt_name,
                    "seed":               seed,
                    "lr":                 lr,
                    "final_loss":         final_loss,
                    "steps_to_convergence": converged_step if converged_step else cfg.NN_STEPS,
                    "n_saddle_events":    n_saddle,
                    "SEE_NN_basic":       see_basic_nn,
                    "SEE_NN_quality":     see_quality_nn,
                })
                training_curves[(task_name, opt_name, seed)] = {
                    "loss":         loss_curve,
                    "grad":         grad_curve,
                    "saddle_steps": saddle_steps,
                }
                print(f"    {opt_name} seed={seed}: final_loss={final_loss:.4f}, "
                      f"saddles={n_saddle}")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "part3.csv", index=False)

    import pickle
    with open(results_dir / "training_curves.pkl", "wb") as fh:
        pickle.dump(training_curves, fh)

    print(f"  Saved part3.csv ({len(df)} rows)")
    return df
