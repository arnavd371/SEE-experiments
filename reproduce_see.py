#!/usr/bin/env python3
"""
Reproduce results from:
  "Saddle Escape Efficiency: A Novel Metric to Benchmark
   Learning Rates in Non-Convex Optimization"

Run: python reproduce_see.py
"""
import math, itertools, time
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

matplotlib.rcParams.update({"font.family": "serif", "font.size": 9,
                             "axes.spines.top": False, "axes.spines.right": False})

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

# ── Paper hyper-parameters ────────────────────────────────────────────────────
N, T_MAX, ESCAPE_R, N_BOOT = 200, 100, 2.0, 2000
LRS  = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
OPTS = ["GD_fixed", "Adam", "RMSProp", "AdaGrad"]
COLORS = {"GD_fixed": "C0", "Adam": "C1", "RMSProp": "C2", "AdaGrad": "C3"}

# ── Benchmark functions  (N,d) → (N,) ────────────────────────────────────────

def himmelblau(x):
    a, b = x[:, 0], x[:, 1]
    return (a**2 + b - 11)**2 + (a + b**2 - 7)**2

def rosenbrock(x):
    return (100*(x[:, 1:] - x[:, :-1]**2)**2 + (1 - x[:, :-1])**2).sum(-1)

def ackley(x, a=20, b=0.2, c=2*math.pi):
    s1 = (x**2).mean(1)
    s2 = torch.cos(c * x).mean(1)
    return -a * torch.exp(-b * s1.sqrt()) - torch.exp(s2) + a + math.e

def rastrigin(x):
    return 10 * x.shape[1] + (x**2 - 10*torch.cos(2*math.pi*x)).sum(-1)

def levy(x):
    w  = 1 + (x - 1) / 4
    t1 = torch.sin(math.pi * w[:, 0])**2
    t2 = ((w[:, :-1]-1)**2 * (1 + 10*torch.sin(math.pi*w[:, 1:])**2)).sum(-1)
    t3 = (w[:, -1]-1)**2 * (1 + torch.sin(2*math.pi*w[:, -1])**2)
    return t1 + t2 + t3

FUNCS_2D = [("Himmelblau", himmelblau, 2),
            ("Rosenbrock",  rosenbrock,  2),
            ("Ackley_2D",   ackley,       2),
            ("Rastrigin",   rastrigin,    2),
            ("Levy",        levy,         2)]

TABLE_CSV = {"Himmelblau": "table1_himmelblau",
             "Rosenbrock":  "table2_rosenbrock",
             "Ackley_2D":   "table3_ackley",
             "Rastrigin":   "table4_rastrigin",
             "Levy":        "table5_levy"}

# ── Batched optimizer  state: (N,d) ──────────────────────────────────────────

class Opt:
    def __init__(self, name, lr, N, d):
        self.name, self.lr, self.t = name, lr, 0
        z = lambda: torch.zeros(N, d, device=DEVICE)
        self.m, self.v, self.G = z(), z(), z()
        self.b1, self.b2, self.eps, self.rho = 0.9, 0.999, 1e-8, 0.99

    def step(self, g):
        self.t += 1
        if self.name == "GD_fixed":
            return self.lr * g
        if self.name == "Adam":
            self.m = self.b1*self.m + (1-self.b1)*g
            self.v = self.b2*self.v + (1-self.b2)*g**2
            mh = self.m / (1 - self.b1**self.t)
            vh = self.v / (1 - self.b2**self.t)
            return self.lr * mh / (vh.sqrt() + self.eps)
        if self.name == "RMSProp":
            self.v = self.rho*self.v + (1-self.rho)*g**2
            return self.lr * g / (self.v.sqrt() + self.eps)
        if self.name == "AdaGrad":
            self.G += g**2
            return self.lr * g / (self.G.sqrt() + self.eps)

# ── Vectorized trial runner ───────────────────────────────────────────────────

def run_trials(f, d, opt_name, lr, noise_sigma=0.0, escape_thresh=ESCAPE_R):
    # Per-trial seeded init: torch.manual_seed(i) for trial i
    x0s = []
    for i in range(N):
        torch.manual_seed(i)
        x0s.append(torch.randn(d) * 0.1)
    x = torch.stack(x0s).to(DEVICE)          # (N, d)
    torch.manual_seed(42)                     # deterministic noise seed

    opt      = Opt(opt_name, lr, N, d)
    escaped  = torch.zeros(N, dtype=torch.bool,    device=DEVICE)
    esc_step = torch.zeros(N, dtype=torch.float32, device=DEVICE)

    for step in range(1, T_MAX + 1):
        xv = x.detach().clone().requires_grad_(True)
        g  = torch.autograd.grad(f(xv).sum(), xv)[0].detach()
        g  = torch.nan_to_num(g, nan=0., posinf=1e6, neginf=-1e6)
        if noise_sigma > 0:
            g = g + noise_sigma * torch.randn_like(g)
        with torch.no_grad():
            x = x - opt.step(g)
            new = (~escaped) & (x.norm(dim=1) > escape_thresh)
            if new.any():
                esc_step[new] = float(step)
                escaped |= new

    return escaped.cpu().numpy(), esc_step.cpu().numpy()

# ── SEE, bootstrap CI, Mann-Whitney ──────────────────────────────────────────

def compute_see(escaped, esc_step):
    n = escaped.sum()
    return float(n / N / esc_step[escaped].mean()) if n > 0 else 0.0

def bootstrap_ci(escaped, esc_step):
    rng = np.random.default_rng(42)
    idx = rng.integers(0, N, size=(N_BOOT, N))
    e   = escaped[idx]
    s   = np.where(escaped, esc_step, 0.)[idx]
    n   = e.sum(1); safe = np.maximum(n, 1)
    tau = s.sum(1) / safe
    p   = n / N
    see = np.where((p > 0) & (tau > 0), p / tau, 0.)
    return float(np.percentile(see, 2.5)), float(np.percentile(see, 97.5))

def mann_whitney(esc1, s1, esc2, s2):
    x1 = np.where(esc1, s1, float(T_MAX + 1))
    x2 = np.where(esc2, s2, float(T_MAX + 1))
    res = mannwhitneyu(x1, x2, alternative="two-sided")
    r   = 1 - 2 * res.statistic / (len(x1) * len(x2))
    return float(res.pvalue), float(r)

# ── Cache to avoid re-running identical configurations ───────────────────────

_CACHE = {}

def _get(f, fname, d, opt, lr, noise=0.0, thresh=ESCAPE_R):
    key = (fname, opt, lr, noise, thresh)
    if key not in _CACHE:
        _CACHE[key] = run_trials(f, d, opt, lr,
                                  noise_sigma=noise, escape_thresh=thresh)
    return _CACHE[key]

# ── Experiment: Tables I–V ────────────────────────────────────────────────────

def run_main_tables():
    all_rows = []
    for fname, f, d in FUNCS_2D:
        rows = []
        print(f"\n{'='*62}\n  {fname}\n{'='*62}")
        hdr = f"{'LR':>6} | " + " | ".join(f"{o:>16}" for o in OPTS)
        print(hdr); print("-" * len(hdr))
        for lr in LRS:
            cells = {}
            for opt in OPTS:
                esc, stp    = _get(f, fname, d, opt, lr)
                see         = compute_see(esc, stp)
                lo, hi      = bootstrap_ci(esc, stp)
                half        = (hi - lo) / 2
                cells[opt]  = (see, half, lo, hi)
            line = f"{lr:>6.3f} | " + " | ".join(
                f"{cells[o][0]:.4f} ±{cells[o][1]:.4f}" for o in OPTS)
            print(line)
            for opt in OPTS:
                see, half, lo, hi = cells[opt]
                rows.append({"function": fname, "optimizer": opt, "lr": lr,
                             "SEE": see, "CI_lo": lo, "CI_hi": hi,
                             "half_CI": half})
        all_rows.extend(rows)
        pd.DataFrame(rows).to_csv(
            RESULTS / f"{TABLE_CSV[fname]}.csv", index=False)
    return pd.DataFrame(all_rows)

# ── Experiment: Table VI – gradient noise ────────────────────────────────────

def run_noise_table():
    SIGMAS, LR = [0.0, 0.01, 0.1, 0.5], 0.1
    rows = []
    print(f"\n{'='*62}\n  Table VI: Gradient Noise  (lr={LR})\n{'='*62}")
    for fname, f, d in FUNCS_2D:
        for sig in SIGMAS:
            for opt in OPTS:
                esc, stp = _get(f, fname, d, opt, LR, noise=sig)
                see      = compute_see(esc, stp)
                lo, hi   = bootstrap_ci(esc, stp)
                rows.append({"function": fname, "optimizer": opt,
                             "noise_sigma": sig, "lr": LR,
                             "SEE": see, "half_CI": (hi - lo) / 2})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "table6_noise.csv", index=False)
    pivot = df.pivot_table(index=["function", "noise_sigma"],
                           columns="optimizer", values="SEE")
    print(pivot.to_string())
    return df

# ── Experiment: Fig 3 – threshold sensitivity ─────────────────────────────────

def run_threshold_exp():
    THRESHS = [1.5, 2.0, 3.0]
    rows = []
    print(f"\n  Running threshold sensitivity (r ∈ {THRESHS}) …")
    for thresh in THRESHS:
        for fname, f, d in FUNCS_2D:
            for opt in OPTS:
                for lr in LRS:
                    esc, stp = _get(f, fname, d, opt, lr, thresh=thresh)
                    rows.append({"function": fname, "optimizer": opt,
                                 "lr": lr, "threshold": thresh,
                                 "SEE": compute_see(esc, stp)})
    return pd.DataFrame(rows)

# ── Experiment: Fig 2 – Ackley dimensionality ────────────────────────────────

def run_dim_exp():
    rows = []
    print("  Running Ackley dimensionality (d = 2, 5, 10) …")
    for d in [2, 5, 10]:
        fname_d = f"Ackley_{d}D"
        for opt in OPTS:
            for lr in LRS:
                esc, stp = _get(ackley, fname_d, d, opt, lr)
                rows.append({"dim": d, "optimizer": opt, "lr": lr,
                             "SEE": compute_see(esc, stp)})
    return pd.DataFrame(rows)

# ── Experiment: Table VII – Mann-Whitney ─────────────────────────────────────

def run_mannwhitney():
    LR_MW = 0.2
    rows  = []
    print(f"\n{'='*62}\n  Table VII: Mann-Whitney pairwise (lr={LR_MW})\n{'='*62}")
    for fname, f, d in FUNCS_2D:
        data = {opt: _get(f, fname, d, opt, LR_MW) for opt in OPTS}
        for o1, o2 in itertools.combinations(OPTS, 2):
            p, r = mann_whitney(*data[o1], *data[o2])
            sig  = ("***" if p < 0.001 else "**" if p < 0.01
                    else "*" if p < 0.05 else "ns")
            rows.append({"function": fname, "opt1": o1, "opt2": o2,
                         "p_value": p, "r": r, "sig": sig})
            print(f"  {fname:12s}: {o1:12s} vs {o2:12s} "
                  f"p={p:.4f} {sig:3s}  r={r:+.3f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "table7_mannwhitney.csv", index=False)
    return df

# ── Figures ───────────────────────────────────────────────────────────────────

def fig1(df_main):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=False)
    for ax, (fname, _, _) in zip(axes, FUNCS_2D):
        sub = df_main[df_main.function == fname]
        for opt in OPTS:
            o = sub[sub.optimizer == opt]
            ax.errorbar(o.lr, o.SEE, yerr=o.half_CI, label=opt,
                        marker="o", ms=4, capsize=3, color=COLORS[opt])
        ax.set_xscale("log"); ax.set_title(fname, fontsize=8)
        ax.set_xlabel("Learning rate"); ax.set_ylabel("SEE")
    h, l = axes[0].get_legend_handles_labels()
    if h:
        fig.legend(h, l, loc="upper center", ncol=4, fontsize=8,
                   bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Fig 1 – SEE vs Learning Rate", y=1.09, fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig1_see_vs_lr.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  Saved fig1_see_vs_lr.png")


def fig2(df_dim):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, d in zip(axes, [2, 5, 10]):
        sub = df_dim[df_dim.dim == d]
        for opt in OPTS:
            o = sub[sub.optimizer == opt]
            ax.plot(o.lr, o.SEE, marker="o", ms=4, label=opt, color=COLORS[opt])
        ax.set_xscale("log"); ax.set_title(f"Ackley d={d}")
        ax.set_xlabel("Learning rate"); ax.set_ylabel("SEE")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, fontsize=7)
    fig.suptitle("Fig 2 – Ackley: SEE vs Dimensionality", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig2_dimensionality.png", dpi=300, bbox_inches="tight")
    plt.close(); print("  Saved fig2_dimensionality.png")


def fig3(df_thresh):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8), sharey=False)
    LS = {1.5: "--", 2.0: "-", 3.0: ":"}
    for ax, (fname, _, _) in zip(axes, FUNCS_2D):
        for thresh in [1.5, 2.0, 3.0]:
            agg = (df_thresh[(df_thresh.function == fname)
                             & (df_thresh.threshold == thresh)]
                   .groupby("lr")["SEE"].mean().reset_index())
            ax.plot(agg.lr, agg.SEE, linestyle=LS[thresh],
                    marker="o", ms=4, label=f"r={thresh}")
        ax.set_xscale("log"); ax.set_title(fname, fontsize=8)
        ax.set_xlabel("Learning rate"); ax.set_ylabel("SEE")
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, fontsize=7)
    fig.suptitle("Fig 3 – Escape Threshold Robustness (avg over optimizers)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_threshold_robustness.png", dpi=300,
                bbox_inches="tight")
    plt.close(); print("  Saved fig3_threshold_robustness.png")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    t0 = time.time()
    print(f"Device: {DEVICE}  |  N={N}  |  T_max={T_MAX}  |  "
          f"escape_r={ESCAPE_R}  |  boot={N_BOOT}")

    df_main   = run_main_tables()
    df_noise  = run_noise_table()
    df_dim    = run_dim_exp()
    df_thresh = run_threshold_exp()
    df_mw     = run_mannwhitney()

    print("\nGenerating figures …")
    fig1(df_main)
    fig2(df_dim)
    fig3(df_thresh)

    print(f"\n=== All done in {time.time()-t0:.1f}s ===")
    print(f"Results saved to {RESULTS.resolve()}/")
