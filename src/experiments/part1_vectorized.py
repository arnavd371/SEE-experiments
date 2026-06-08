"""Part 1: Vectorized 2-D benchmark experiments (all N trials as GPU tensors)."""
import math
import numpy as np
import torch
import pandas as pd
import yaml
from pathlib import Path

from config import Config
from src.functions.classical_2d import FUNCTIONS, DOMAINS, domain_diameter
from src.functions.saddle_finder import find_saddles, lambda_min_2d
from src.metrics.see import compute_see
from src.metrics.statistics import pairwise_wilcoxon
from src.utils.seeding import set_all_seeds


# ── Batched optimizer (all state tensors are (N, d)) ──────────────────────────

class BatchedOptimizer:
    def __init__(self, name, lr, N, d, device, wd=0.01):
        self.name = name
        self.lr   = lr
        self.wd   = wd
        self.t    = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.rho      = 0.99
        self.momentum = 0.9
        z = lambda: torch.zeros(N, d, device=device)
        self.m   = z()   # Adam first moment
        self.v   = z()   # Adam/RMSProp second moment
        self.G   = z()   # AdaGrad accumulator
        self.vel = z()   # SGD momentum

    def step(self, grads: torch.Tensor) -> torch.Tensor:
        """Return (N,d) update to subtract from x."""
        self.t += 1
        lr = self.lr
        if self.name == "GD_fixed":
            return lr * grads
        elif self.name in ("Adam", "AdamW"):
            self.m = self.beta1 * self.m + (1 - self.beta1) * grads
            self.v = self.beta2 * self.v + (1 - self.beta2) * grads ** 2
            m_h = self.m / (1 - self.beta1 ** self.t)
            v_h = self.v / (1 - self.beta2 ** self.t)
            return lr * m_h / (v_h.sqrt() + self.eps)
        elif self.name == "RMSProp":
            self.v = self.rho * self.v + (1 - self.rho) * grads ** 2
            return lr * grads / (self.v.sqrt() + self.eps)
        elif self.name == "AdaGrad":
            self.G = self.G + grads ** 2
            return lr * grads / (self.G.sqrt() + self.eps)
        elif self.name == "SGD_momentum":
            self.vel = self.momentum * self.vel + grads
            return lr * self.vel
        else:
            raise ValueError(f"Unknown optimizer: {self.name}")


# ── Hessian helpers ───────────────────────────────────────────────────────────

def _hessian_2d(f, x_single: torch.Tensor) -> torch.Tensor:
    """2×2 Hessian of f at a single (2,) point."""
    x = x_single.detach().clone().float().requires_grad_(True)
    val = f(x.unsqueeze(0))[0]
    g = torch.autograd.grad(val, x, create_graph=True)[0]
    H = torch.stack([
        torch.autograd.grad(g[0], x, retain_graph=True)[0],
        torch.autograd.grad(g[1], x, retain_graph=False)[0],
    ])
    return H.detach()


# ── Core trial loop ───────────────────────────────────────────────────────────

def run_trials(f, x_saddle, r_escape, r_diverge, optimizer_name, lr,
               N, T_MAX, perturbation_std, device):
    """
    Run N trials simultaneously.  Returns numpy arrays:
      escaped_mask, escape_steps, quality_mask, quality_steps, diverged_mask
    All step arrays are 1-indexed (1 = condition met at step 1).
    Zero means 'never met'.
    """
    x_s = torch.tensor(x_saddle, dtype=torch.float32, device=device)

    x = (x_s.unsqueeze(0).expand(N, -1).clone()
         + perturbation_std * torch.randn(N, 2, device=device))

    opt = BatchedOptimizer(optimizer_name, lr, N, 2, device)

    escaped_ever  = torch.zeros(N, dtype=torch.bool,  device=device)
    quality_ever  = torch.zeros(N, dtype=torch.bool,  device=device)
    diverged_ever = torch.zeros(N, dtype=torch.bool,  device=device)
    escape_step   = torch.zeros(N, dtype=torch.float32, device=device)
    quality_step  = torch.zeros(N, dtype=torch.float32, device=device)

    for step in range(T_MAX):
        # ── Gradient computation (single pass) ───────────────────────────
        x_v = x.detach().clone().requires_grad_(True)
        grads = torch.autograd.grad(f(x_v).sum(), x_v)[0].detach()  # (N,2)
        grads = torch.nan_to_num(grads, nan=0.0, posinf=1e6, neginf=-1e6)

        with torch.no_grad():
            dist       = (x - x_s).norm(dim=1)   # (N,)
            grad_norms = grads.norm(dim=1)         # (N,)

            # QUALITY_MIN — only for small-grad trials not yet marked
            small = (grad_norms < Config.GRAD_NORM_THRESH) & (~quality_ever)
            if small.any():
                for idx in small.nonzero(as_tuple=True)[0].tolist():
                    try:
                        H = _hessian_2d(f, x[idx])
                        lmin = torch.linalg.eigvalsh(H)[0].item()
                        if lmin > Config.LAMBDA_MIN_THRESH:
                            quality_ever[idx] = True
                            quality_step[idx] = float(step + 1)
                    except Exception:
                        pass

            # ESCAPED
            just_esc = (~escaped_ever) & (dist > r_escape)
            if just_esc.any():
                escaped_ever |= just_esc
                escape_step[just_esc] = float(step + 1)

            # DIVERGED
            diverged_ever |= dist > r_diverge

        # ── Optimizer update ──────────────────────────────────────────────
        with torch.no_grad():
            delta = opt.step(grads)
            if optimizer_name == "AdamW":
                x.mul_(1.0 - lr * opt.wd)
            x.sub_(delta)

    return (
        escaped_ever.cpu().numpy(),
        escape_step.cpu().numpy(),
        quality_ever.cpu().numpy(),
        quality_step.cpu().numpy(),
        diverged_ever.cpu().numpy(),
    )


# ── Part-1 driver ─────────────────────────────────────────────────────────────

def run_part1(cfg: type, results_dir: Path) -> pd.DataFrame:
    set_all_seeds(42)
    device = cfg.DEVICE
    rows = []
    escape_data = {}   # key -> np.array of per-trial escape_steps (violin plots)

    for func_name, f in FUNCTIONS.items():
        print(f"  [{func_name}] finding saddles …")
        domain = DOMAINS[func_name]
        diam   = domain_diameter(func_name)
        saddles = find_saddles(f, domain, device)

        if not saddles:
            print(f"    WARNING: no saddles found for {func_name}, skipping.")
            continue

        for sid, xs in enumerate(saddles):
            lmin_s  = lambda_min_2d(f, xs, device)
            r_esc   = min(0.25 * diam, 0.5 / math.sqrt(abs(lmin_s) + 1e-6))
            r_div   = 0.5 * diam

            print(f"    saddle {sid}: x={xs.round(4)}, λ_min={lmin_s:.4f}, "
                  f"r_esc={r_esc:.4f}, r_div={r_div:.4f}")

            for opt_name in cfg.OPTIMIZERS:
                for lr in cfg.LEARNING_RATES:
                    set_all_seeds(42)
                    esc, esc_s, qlt, qlt_s, div = run_trials(
                        f, xs, r_esc, r_div,
                        opt_name, lr,
                        cfg.N_TRIALS, cfg.T_MAX, cfg.PERTURBATION_STD, device,
                    )

                    # Accumulate per-trial escape_steps for violin plots
                    key = f"{func_name}|{opt_name}|{lr}|{sid}"
                    escape_data[key] = esc_s[esc]  # only escapers' steps

                    # Replace 0 (never) with NaN for metric computation
                    esc_s_m = np.where(esc, esc_s, np.nan)
                    qlt_s_m = np.where(qlt, qlt_s, np.nan)

                    metrics = compute_see(esc, esc_s_m, qlt, qlt_s_m, div,
                                         n_bootstrap=cfg.N_BOOTSTRAP)
                    rows.append({
                        "function":   func_name,
                        "optimizer":  opt_name,
                        "lr":         lr,
                        "saddle_id":  sid,
                        "saddle_x":   float(xs[0]),
                        "saddle_y":   float(xs[1]),
                        "lambda_min": lmin_s,
                        "r_escape":   r_esc,
                        "r_diverge":  r_div,
                        **metrics,
                    })

    df = pd.DataFrame(rows)
    if df.empty:
        print("WARNING: no results generated in Part 1.")
        return df, {}

    # Save per-trial escape data for violin plots
    import pickle
    with open(results_dir / "escape_data.pkl", "wb") as fh:
        pickle.dump(escape_data, fh)

    # ── Best LR per optimizer (argmax mean SEE_basic across all functions/saddles)
    best_lrs = {}
    for opt_name in cfg.OPTIMIZERS:
        sub = df[df.optimizer == opt_name]
        by_lr = sub.groupby("lr")["SEE_basic"].mean()
        best_lrs[opt_name] = float(by_lr.idxmax())

    # Save best_lrs.yaml
    with open(results_dir / "best_lrs.yaml", "w") as fh:
        yaml.dump(best_lrs, fh)
    print(f"  Best LRs: {best_lrs}")

    # ── Pairwise Wilcoxon at best LR ─────────────────────────────────────
    best_rows = df[df.apply(lambda r: r.lr == best_lrs[r.optimizer], axis=1)]
    opt_see = {
        opt: best_rows[best_rows.optimizer == opt]["SEE_basic"].values
        for opt in cfg.OPTIMIZERS
    }
    # Best optimizer by mean SEE_basic at best LR
    means = {k: v.mean() for k, v in opt_see.items() if len(v) > 0}
    best_opt = max(means, key=means.get)
    print(f"  Best optimizer: {best_opt}")

    n_comp = len(cfg.OPTIMIZERS) - 1
    wlx = pairwise_wilcoxon(opt_see, best_opt, n_comparisons=n_comp)

    # Fill columns (same value for all rows of an optimizer)
    df["wilcoxon_p_vs_best"] = float("nan")
    df["cohens_d_vs_best"]   = float("nan")
    for opt_name in cfg.OPTIMIZERS:
        if opt_name == best_opt:
            df.loc[df.optimizer == opt_name, "wilcoxon_p_vs_best"] = float("nan")
            df.loc[df.optimizer == opt_name, "cohens_d_vs_best"]   = 0.0
        elif opt_name in wlx:
            df.loc[df.optimizer == opt_name, "wilcoxon_p_vs_best"] = wlx[opt_name]["p"]
            df.loc[df.optimizer == opt_name, "cohens_d_vs_best"]   = wlx[opt_name]["d"]

    df.to_csv(results_dir / "part1.csv", index=False)
    print(f"  Saved part1.csv ({len(df)} rows)")
    return df, escape_data
