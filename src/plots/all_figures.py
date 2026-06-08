"""
All 8 figures for the SEE experiment suite.
300 dpi, serif font, no top/right spines.
"""
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

matplotlib.rcParams.update({
    "font.family":  "serif",
    "font.size":    9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   300,
})

OPTIMIZER_ORDER = ["GD_fixed", "Adam", "AdamW", "RMSProp", "AdaGrad", "SGD_momentum"]
COLORS = plt.cm.tab10.colors


def _savefig(fig, path: Path, name: str):
    fig.tight_layout()
    fig.savefig(path / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


def _no_empty_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels)


# ── Fig1 & Fig2: SEE heatmaps ─────────────────────────────────────────────────

def _see_heatmap(df: pd.DataFrame, metric: str, title: str, fig_path: Path, fname: str):
    funcs = [f for f in df["function"].unique() if f in df["function"].values]
    n = len(funcs)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for ax, func in zip(axes, funcs):
        sub = df[df["function"] == func]
        opts = [o for o in OPTIMIZER_ORDER if o in sub["optimizer"].values]
        lrs  = sorted(sub["lr"].unique())
        mat  = np.full((len(lrs), len(opts)), float("nan"))
        for i, lr in enumerate(lrs):
            for j, opt in enumerate(opts):
                vals = sub[(sub["lr"] == lr) & (sub["optimizer"] == opt)][metric]
                if len(vals) > 0:
                    mat[i, j] = vals.mean()
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=np.nanmax(mat) if not np.all(np.isnan(mat)) else 1)
        ax.set_xticks(range(len(opts)))
        ax.set_xticklabels(opts, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(lrs)))
        ax.set_yticklabels([f"{lr:.3f}" for lr in lrs], fontsize=7)
        ax.set_title(func, fontsize=9)
        ax.set_xlabel("Optimizer", fontsize=8)
        ax.set_ylabel("LR", fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.01)
    _savefig(fig, fig_path, fname)


def fig1_see_basic(df: pd.DataFrame, fig_path: Path):
    _see_heatmap(df, "SEE_basic", "Fig1: SEE_basic (opt × lr)", fig_path, "fig1_see_basic.png")


def fig2_see_quality(df: pd.DataFrame, fig_path: Path):
    _see_heatmap(df, "SEE_quality", "Fig2: SEE_quality (opt × lr)", fig_path, "fig2_see_quality.png")


# ── Fig3: SEE vs dimension ────────────────────────────────────────────────────

def fig3_see_vs_dim(df2: pd.DataFrame, fig_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, func_filter, title in [
        (axes[0], "Synthetic",       "Synthetic saddle (k=d//2)"),
        (axes[1], "Styblinski_nD",   "Styblinski-nD"),
    ]:
        sub = df2[df2["function"].str.contains(func_filter)]
        if sub.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        for j, opt in enumerate(OPTIMIZER_ORDER):
            o = sub[sub["optimizer"] == opt]
            if o.empty:
                continue
            agg = o.groupby("dim")["SEE_basic"].mean()
            ax.plot(agg.index, agg.values, marker="o", label=opt, color=COLORS[j])
        ax.set_xscale("log")
        ax.set_xlabel("Dimension d")
        ax.set_ylabel("SEE_basic")
        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7)

    fig.suptitle("Fig3: SEE_basic vs Dimension", fontsize=11)
    _savefig(fig, fig_path, "fig3_see_vs_dim.png")


# ── Fig4: SEE vs saddle index k at d=50 ──────────────────────────────────────

def fig4_see_vs_k(df2: pd.DataFrame, fig_path: Path):
    fig, ax = plt.subplots(figsize=(6, 4))
    sub = df2[(df2["dim"] == 50) & (df2["function"].str.contains("Synthetic"))]
    if sub.empty:
        ax.text(0.5, 0.5, "No d=50 data", ha="center", va="center", transform=ax.transAxes)
    else:
        for j, opt in enumerate(OPTIMIZER_ORDER):
            o = sub[sub["optimizer"] == opt]
            if o.empty:
                continue
            agg = o.groupby("k")["SEE_basic"].mean()
            ax.plot(agg.index, agg.values, marker="s", label=opt, color=COLORS[j])
        ax.set_xlabel("Saddle index k")
        ax.set_ylabel("SEE_basic")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, fontsize=7)
    ax.set_title("Fig4: SEE_basic vs Saddle Index k (d=50)")
    _savefig(fig, fig_path, "fig4_see_vs_k.png")


# ── Fig5: NN training curves with saddle bands ───────────────────────────────

def fig5_nn_training(fig_path: Path, curves_pkl: Path):
    if not curves_pkl.exists():
        print("  Fig5: training_curves.pkl not found, skipping.")
        return

    with open(curves_pkl, "rb") as fh:
        curves = pickle.load(fh)

    tasks = sorted({k[0] for k in curves.keys()})
    n = len(tasks)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, task in enumerate(tasks):
        ax_loss = axes[row, 0]
        ax_grad = axes[row, 1]

        saddle_positions = set()
        plotted = False
        for j, opt in enumerate(OPTIMIZER_ORDER):
            key = (task, opt, 0)   # seed=0 representative
            if key not in curves:
                continue
            c = curves[key]
            loss  = c["loss"]
            steps_loss = list(range(1, len(loss) + 1))
            ax_loss.plot(steps_loss, loss, alpha=0.7, label=opt,
                         color=COLORS[j], linewidth=0.8)

            if c.get("grad"):
                gsteps, gnorms = zip(*c["grad"])
                ax_grad.plot(gsteps, gnorms, alpha=0.7, label=opt,
                             color=COLORS[j], linewidth=0.8)

            for sp in c.get("saddle_steps", []):
                saddle_positions.add(sp)
            plotted = True

        # Saddle event bands
        for sp in saddle_positions:
            ax_loss.axvline(sp, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax_grad.axvline(sp, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

        ax_loss.set_title(f"{task} – Loss", fontsize=9)
        ax_loss.set_xlabel("Step")
        ax_loss.set_ylabel("Loss")
        ax_grad.set_title(f"{task} – Full-batch ‖∇‖", fontsize=9)
        ax_grad.set_xlabel("Step")
        ax_grad.set_ylabel("Grad norm")
        if plotted:
            for ax in (ax_loss, ax_grad):
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles, labels, fontsize=6)

    fig.suptitle("Fig5: NN Training (red = saddle events)", fontsize=11)
    _savefig(fig, fig_path, "fig5_nn_training.png")


# ── Fig6: Escape-type stacked bars ───────────────────────────────────────────

def fig6_escape_stacked(df: pd.DataFrame, fig_path: Path, best_lrs: dict):
    fig, ax = plt.subplots(figsize=(8, 4))
    opts     = [o for o in OPTIMIZER_ORDER if o in df["optimizer"].values]
    cat_cols = {"QUALITY_MIN": "steelblue", "ESCAPED_only": "mediumseagreen",
                "DIVERGED": "tomato",       "STUCK": "lightgray"}

    bottoms = np.zeros(len(opts))
    first_pass = True
    for cat, color in cat_cols.items():
        vals = []
        for opt in opts:
            lr   = best_lrs.get(opt, df[df.optimizer == opt]["lr"].mode().iloc[0])
            sub  = df[(df.optimizer == opt) & (df.lr == lr)]
            if sub.empty:
                vals.append(0.0)
                continue
            if cat == "QUALITY_MIN":
                v = sub["P_quality"].mean()
            elif cat == "ESCAPED_only":
                v = max(0, sub["P_escape"].mean() - sub["P_quality"].mean())
            elif cat == "DIVERGED":
                v = max(0, sub["diverge_rate"].mean() - sub["P_escape"].mean())
            else:
                v = sub["stuck_rate"].mean()
            vals.append(float(v))
        vals = np.array(vals)
        bars = ax.bar(opts, vals, bottom=bottoms, label=cat, color=color)
        bottoms += vals

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of trials")
    ax.set_xlabel("Optimizer")
    ax.set_title("Fig6: Escape-type stacked bars (best LR)")
    ax.set_xticklabels(opts, rotation=30, ha="right")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)
    _savefig(fig, fig_path, "fig6_escape_stacked.png")


# ── Fig7: Escape-time violins ─────────────────────────────────────────────────

def fig7_escape_violins(df: pd.DataFrame, fig_path: Path, best_lrs: dict,
                        escape_pkl: Path | None = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    opts = [o for o in OPTIMIZER_ORDER if o in df["optimizer"].values]

    # Try to load per-trial data
    escape_data = {}
    if escape_pkl is not None and escape_pkl.exists():
        with open(escape_pkl, "rb") as fh:
            escape_data = pickle.load(fh)

    positions, all_data, labels = [], [], []
    for i, opt in enumerate(opts):
        lr  = best_lrs.get(opt, df[df.optimizer == opt]["lr"].mode().iloc[0])
        # Collect all per-trial escape steps for this optimizer at best LR
        steps = []
        if escape_data:
            for key, arr in escape_data.items():
                parts = key.split("|")
                if len(parts) == 4 and parts[1] == opt and abs(float(parts[2]) - lr) < 1e-9:
                    steps.extend(arr.tolist())

        if not steps:
            # Fall back to tau_escape_mean ± rough spread
            sub = df[(df.optimizer == opt) & (df.lr == lr)]
            tau = sub["tau_escape_mean"].dropna()
            if len(tau) > 0:
                m = tau.mean()
                steps = [m]   # single point
        if steps:
            positions.append(i)
            all_data.append(steps)
            labels.append(opt)

    if all_data:
        parts = ax.violinplot(all_data, positions=positions, showmedians=True,
                              showextrema=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "No escape data", ha="center", va="center",
                transform=ax.transAxes)

    ax.set_ylabel("Escape step")
    ax.set_xlabel("Optimizer")
    ax.set_title("Fig7: Escape-time violins (best LR)")
    _savefig(fig, fig_path, "fig7_escape_violins.png")


# ── Fig8: 6×6 ranking-consistency heatmap ────────────────────────────────────

def fig8_ranking_matrix(df1: pd.DataFrame, fig_path: Path):
    funcs = df1["function"].unique().tolist()
    n = len(funcs)
    if n == 0:
        return
    from scipy.stats import spearmanr

    opt_rank = {}
    for func in funcs:
        sub = df1[df1["function"] == func]
        agg = sub.groupby("optimizer")["SEE_basic"].mean()
        opt_rank[func] = agg.to_dict()

    mat = np.full((n, n), float("nan"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, fa in enumerate(funcs):
            for j, fb in enumerate(funcs):
                va = [opt_rank[fa].get(o, np.nan) for o in OPTIMIZER_ORDER]
                vb = [opt_rank[fb].get(o, np.nan) for o in OPTIMIZER_ORDER]
                va, vb = np.array(va), np.array(vb)
                mask = ~(np.isnan(va) | np.isnan(vb))
                if mask.sum() < 3:
                    continue
                if np.all(va[mask] == va[mask][0]) or np.all(vb[mask] == vb[mask][0]):
                    continue
                res = spearmanr(va[mask], vb[mask])
                mat[i, j] = float(res.statistic)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(funcs, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(funcs, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Spearman ρ")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_title("Fig8: Optimizer ranking consistency across function pairs")
    _savefig(fig, fig_path, "fig8_ranking_matrix.png")


# ── Master entry point ────────────────────────────────────────────────────────

def generate_all(results_dir: Path, fig_path: Path, best_lrs: dict):
    fig_path.mkdir(parents=True, exist_ok=True)
    p1 = results_dir / "part1.csv"
    p2 = results_dir / "part2.csv"

    if p1.exists():
        df1 = pd.read_csv(p1)
        print("Generating Fig1 …")
        fig1_see_basic(df1, fig_path)
        print("Generating Fig2 …")
        fig2_see_quality(df1, fig_path)
        print("Generating Fig6 …")
        fig6_escape_stacked(df1, fig_path, best_lrs)
        print("Generating Fig7 …")
        fig7_escape_violins(df1, fig_path, best_lrs,
                            escape_pkl=results_dir / "escape_data.pkl")
        print("Generating Fig8 …")
        fig8_ranking_matrix(df1, fig_path)
    else:
        print("  part1.csv not found; skipping Figs 1,2,6,7,8.")

    if p2.exists():
        df2 = pd.read_csv(p2)
        print("Generating Fig3 …")
        fig3_see_vs_dim(df2, fig_path)
        print("Generating Fig4 …")
        fig4_see_vs_k(df2, fig_path)
    else:
        print("  part2.csv not found; skipping Figs 3,4.")

    print("Generating Fig5 …")
    fig5_nn_training(fig_path, results_dir / "training_curves.pkl")

    print("All figures done.")
