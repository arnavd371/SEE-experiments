"""Part 2: High-dimensional scaling experiments."""
import math
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from scipy.sparse.linalg import eigsh, LinearOperator

from config import Config
from src.functions.nd_functions import (
    synthetic_saddle, synthetic_saddle_lambda_min,
    styblinski_nd, styblinski_saddle_point, STYBLT_LAMBDA_MIN,
    domain_diameter_nd,
)
from src.metrics.see import compute_see
from src.utils.seeding import set_all_seeds
from src.experiments.part1_vectorized import BatchedOptimizer


# ── Lanczos lambda_min for high-d Hessians ───────────────────────────────────

def _lanczos_lambda_min(f_scalar, x_pt: torch.Tensor, k_eig: int = 6) -> float:
    """
    Approximate lambda_min of the Hessian at x_pt via Lanczos (eigsh, k smallest).
    f_scalar: (d,) -> scalar callable.
    """
    d = x_pt.numel()
    x_pt = x_pt.detach().clone().float()

    def hvp_fn(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=torch.float32, device=x_pt.device)
        x = x_pt.clone().requires_grad_(True)
        g = torch.autograd.grad(f_scalar(x), x, create_graph=True)[0]
        hv = torch.autograd.grad((g * v).sum(), x)[0]
        return hv.detach().cpu().numpy().astype(np.float64)

    A = LinearOperator((d, d), matvec=hvp_fn)
    try:
        k = min(k_eig, d - 1) if d > k_eig else max(1, d - 1)
        vals = eigsh(A, k=k, which="SA", return_eigenvectors=False,
                     maxiter=d * 10, tol=1e-4)
        return float(vals.min())
    except Exception:
        return float("nan")


# ── Per-sample grad for batched nd trials ────────────────────────────────────

def _batch_grads(f, x: torch.Tensor) -> torch.Tensor:
    x_v = x.detach().clone().requires_grad_(True)
    grads = torch.autograd.grad(f(x_v).sum(), x_v)[0].detach()
    return torch.nan_to_num(grads, nan=0.0, posinf=1e6, neginf=-1e6)


# ── nd trial loop (vectorized) ───────────────────────────────────────────────

def run_trials_nd(f, f_scalar, x_saddle: torch.Tensor,
                  r_escape: float, r_diverge: float,
                  optimizer_name: str, lr: float,
                  N: int, T_MAX: int, perturbation_std: float,
                  device: str, d: int, use_lanczos: bool,
                  lanczos_k: int = 6):
    x_s = x_saddle.to(device=device, dtype=torch.float32)

    x = (x_s.unsqueeze(0).expand(N, -1).clone()
         + perturbation_std * torch.randn(N, d, device=device))

    opt = BatchedOptimizer(optimizer_name, lr, N, d, device)

    escaped_ever  = torch.zeros(N, dtype=torch.bool,  device=device)
    quality_ever  = torch.zeros(N, dtype=torch.bool,  device=device)
    diverged_ever = torch.zeros(N, dtype=torch.bool,  device=device)
    escape_step   = torch.zeros(N, dtype=torch.float32, device=device)
    quality_step  = torch.zeros(N, dtype=torch.float32, device=device)

    for step in range(T_MAX):
        grads = _batch_grads(f, x)

        with torch.no_grad():
            dist       = (x - x_s).norm(dim=1)
            grad_norms = grads.norm(dim=1)

            # QUALITY_MIN
            small = (grad_norms < Config.GRAD_NORM_THRESH) & (~quality_ever)
            if small.any():
                for idx in small.nonzero(as_tuple=True)[0].tolist():
                    try:
                        if use_lanczos:
                            lmin = _lanczos_lambda_min(
                                lambda xp: f_scalar(xp), x[idx], lanczos_k)
                        else:
                            # 2-D: full Hessian
                            xi = x[idx].clone().requires_grad_(True)
                            val = f_scalar(xi)
                            g = torch.autograd.grad(val, xi, create_graph=True)[0]
                            H = torch.stack([
                                torch.autograd.grad(g[i], xi, retain_graph=True)[0]
                                for i in range(d)
                            ])
                            lmin = torch.linalg.eigvalsh(H)[0].item()
                        if (not math.isnan(lmin)) and lmin > Config.LAMBDA_MIN_THRESH:
                            quality_ever[idx] = True
                            quality_step[idx] = float(step + 1)
                    except Exception:
                        pass

            # ESCAPED
            just_esc = (~escaped_ever) & (dist > r_escape)
            if just_esc.any():
                escaped_ever |= just_esc
                escape_step[just_esc] = float(step + 1)

            diverged_ever |= dist > r_diverge

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


# ── Compute radii for nd synthetic saddle ────────────────────────────────────

def _radii_synthetic(d: int) -> tuple:
    lmin   = synthetic_saddle_lambda_min()          # -2.0
    diam   = domain_diameter_nd(d, -5.0, 5.0)
    r_esc  = min(0.25 * diam, 0.5 / math.sqrt(abs(lmin) + 1e-6))
    r_div  = 0.5 * diam
    return r_esc, r_div, lmin


def _radii_styblinski(d: int) -> tuple:
    lmin   = STYBLT_LAMBDA_MIN                     # ≈ -15.85
    diam   = domain_diameter_nd(d, -5.0, 5.0)
    r_esc  = min(0.25 * diam, 0.5 / math.sqrt(abs(lmin) + 1e-6))
    r_div  = 0.5 * diam
    return r_esc, r_div, lmin


# ── Part-2 driver ─────────────────────────────────────────────────────────────

def run_part2(cfg: type, results_dir: Path,
              best_lrs: dict | None = None) -> pd.DataFrame:
    set_all_seeds(42)
    device = cfg.DEVICE
    rows   = []

    # Load best LRs if available; else fall back to cfg.LEARNING_RATES[1]
    if best_lrs is None:
        import yaml
        lr_file = results_dir / "best_lrs.yaml"
        if lr_file.exists():
            with open(lr_file) as fh:
                best_lrs = yaml.safe_load(fh)
        else:
            best_lrs = {opt: cfg.LEARNING_RATES[len(cfg.LEARNING_RATES)//2]
                        for opt in cfg.OPTIMIZERS}

    saddle_indices_at50 = cfg.SADDLE_INDICES_D50   # [1, 12, 25] for d=50

    for d in cfg.DIMENSIONS:
        use_lanczos = d > 20
        N = cfg.N_TRIALS_HIGHDIM_SMALL if d <= 50 else cfg.N_TRIALS_HIGHDIM_LARGE
        T_MAX = cfg.T_MAX_HIGHDIM

        # Effective perturbation std — scale so L2 ≈ r_escape/2
        r_esc_syn, _, _ = _radii_synthetic(d)
        std_eff = min(cfg.PERTURBATION_STD, r_esc_syn / (2.0 * math.sqrt(d)))

        # ── A: Synthetic saddle (vary k) ──────────────────────────────────
        k_vals = [1, max(1, d // 4), max(1, d // 2)]
        for k in k_vals:
            k = min(k, d)
            r_esc, r_div, lmin = _radii_synthetic(d)

            def f_syn(x, _k=k):
                return synthetic_saddle(x, _k)

            def f_syn_scalar(x, _k=k):
                return synthetic_saddle(x.unsqueeze(0), _k)[0]

            x_sad = torch.zeros(d, dtype=torch.float32)

            for opt_name in cfg.OPTIMIZERS:
                lr = best_lrs.get(opt_name, cfg.LEARNING_RATES[1])
                set_all_seeds(42)
                esc, esc_s, qlt, qlt_s, div = run_trials_nd(
                    f_syn, f_syn_scalar, x_sad, r_esc, r_div,
                    opt_name, lr, N, T_MAX, std_eff, device, d,
                    use_lanczos, cfg.LANCZOS_K,
                )
                esc_s_m = np.where(esc, esc_s, np.nan)
                qlt_s_m = np.where(qlt, qlt_s, np.nan)
                metrics = compute_see(esc, esc_s_m, qlt, qlt_s_m, div,
                                      n_bootstrap=cfg.N_BOOTSTRAP)
                rows.append({
                    "function":  f"Synthetic_k{k}",
                    "dim":       d,
                    "k":         k,
                    "optimizer": opt_name,
                    "lr":        lr,
                    "lambda_min": lmin,
                    "r_escape":  r_esc,
                    "r_diverge": r_div,
                    **metrics,
                })

        # ── B: Styblinski-nD (one saddle per d, varying k=1 default) ──────
        k_stb = max(1, d // 4)
        x_sad_stb = styblinski_saddle_point(d, k_stb).float()
        r_esc_stb, r_div_stb, lmin_stb = _radii_styblinski(d)
        std_stb = min(cfg.PERTURBATION_STD, r_esc_stb / (2.0 * math.sqrt(d)))

        def f_stb(x):
            return styblinski_nd(x)

        def f_stb_scalar(x):
            return styblinski_nd(x.unsqueeze(0))[0]

        for opt_name in cfg.OPTIMIZERS:
            lr = best_lrs.get(opt_name, cfg.LEARNING_RATES[1])
            set_all_seeds(42)
            esc, esc_s, qlt, qlt_s, div = run_trials_nd(
                f_stb, f_stb_scalar, x_sad_stb, r_esc_stb, r_div_stb,
                opt_name, lr, N, T_MAX, std_stb, device, d,
                use_lanczos, cfg.LANCZOS_K,
            )
            esc_s_m = np.where(esc, esc_s, np.nan)
            qlt_s_m = np.where(qlt, qlt_s, np.nan)
            metrics = compute_see(esc, esc_s_m, qlt, qlt_s_m, div,
                                  n_bootstrap=cfg.N_BOOTSTRAP)
            rows.append({
                "function":  "Styblinski_nD",
                "dim":       d,
                "k":         k_stb,
                "optimizer": opt_name,
                "lr":        lr,
                "lambda_min": lmin_stb,
                "r_escape":  r_esc_stb,
                "r_diverge": r_div_stb,
                **metrics,
            })

        print(f"  d={d} done.")

    df = pd.DataFrame(rows)
    df.to_csv(results_dir / "part2.csv", index=False)
    print(f"  Saved part2.csv ({len(df)} rows)")
    return df
