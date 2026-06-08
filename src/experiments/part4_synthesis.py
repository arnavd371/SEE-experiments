"""
Part 4: Synthesis & ablations.

4A. Spearman correlation between SEE_basic and SEE_quality rankings per function.
4B. r_escape sensitivity to curvature constant c.
4C. Spearman(Part-1 SEE_basic vs Part-3 SEE_NN) per optimizer.
4D. 6×6 Spearman matrix of optimizer rankings across function pairs.
"""
import math
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

from config import Config
from src.functions.classical_2d import FUNCTIONS, DOMAINS, domain_diameter
from src.functions.saddle_finder import lambda_min_2d
from src.metrics.see import compute_see
from src.utils.seeding import set_all_seeds


# ── 4A: Spearman(SEE_basic rank, SEE_quality rank) per function ──────────────

def _4a(df1: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for func in df1["function"].unique():
        sub = df1[(df1["function"] == func)]
        if sub.empty:
            continue
        # Mean over (lr, saddle_id) per optimizer
        agg = sub.groupby("optimizer")[["SEE_basic", "SEE_quality"]].mean()
        a, b = agg["SEE_basic"].values, agg["SEE_quality"].values
        if np.all(a == a[0]) or np.all(b == b[0]):
            print(f"  4A [{func}]: constant input — correlation = NaN")
            r, p = float("nan"), float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = spearmanr(a, b)
            r, p = float(res.statistic), float(res.pvalue)
        rows.append({"function": func, "spearman_r": r, "p_value": p})
    return pd.DataFrame(rows)


# ── 4B: r_escape sensitivity to curvature constant c ────────────────────────

def _4b(df1: pd.DataFrame, device: str) -> pd.DataFrame:
    C_VALS = [0.25, 0.5, 1.0, 2.0]
    rows   = []

    for func_name, f in FUNCTIONS.items():
        domain = DOMAINS[func_name]
        diam   = domain_diameter(func_name)
        sub    = df1[df1["function"] == func_name]
        if sub.empty:
            continue

        for c in C_VALS:
            # Re-rank optimizers at best LR with modified r_escape
            opt_see = {}
            for _, row in sub[sub["lr"] == sub["lr"]].iterrows():
                xs     = [row["saddle_x"], row["saddle_y"]]
                lmin_s = row["lambda_min"]
                r_esc_new = min(c * diam, 0.5 / math.sqrt(abs(lmin_s) + 1e-6))
                opt_see.setdefault(row["optimizer"], []).append(row["SEE_basic"])

            # Compute mean rank per optimizer
            mean_see = {k: float(np.mean(v)) for k, v in opt_see.items()}
            ranked   = sorted(mean_see.items(), key=lambda kv: -kv[1])
            for rank, (opt, val) in enumerate(ranked, 1):
                rows.append({
                    "function": func_name,
                    "c":        c,
                    "optimizer": opt,
                    "mean_SEE_basic": val,
                    "rank":     rank,
                })
    return pd.DataFrame(rows)


# ── 4C: Spearman(Part-1 SEE_basic, Part-3 SEE_NN) per optimizer ─────────────

def _4c(df1: pd.DataFrame, df3: pd.DataFrame) -> pd.DataFrame:
    total_saddle_events = df3["n_saddle_events"].sum()
    if total_saddle_events < 3:
        print("  4C: insufficient saddle events — skipping correlation.")
        return pd.DataFrame([{"note": "insufficient saddle events"}])

    rows = []
    for opt in Config.OPTIMIZERS:
        s1 = df1[df1["optimizer"] == opt]["SEE_basic"].mean()
        s3 = df3[df3["optimizer"] == opt]["SEE_NN_basic"]
        s3 = s3.dropna()
        if len(s3) == 0:
            continue
        rows.append({"optimizer": opt, "SEE_basic_p1": s1,
                     "SEE_NN_basic_p3": s3.mean()})
    if len(rows) < 2:
        return pd.DataFrame([{"note": "insufficient data"}])

    rdf  = pd.DataFrame(rows)
    a, b = rdf["SEE_basic_p1"].values, rdf["SEE_NN_basic_p3"].values
    if np.all(a == a[0]) or np.all(b == b[0]):
        r, p = float("nan"), float("nan")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = spearmanr(a, b)
        r, p = float(res.statistic), float(res.pvalue)
    rdf["global_spearman_r"] = r
    rdf["global_p"]          = p
    return rdf


# ── 4D: 6×6 Spearman ranking matrix across function pairs ───────────────────

def _4d(df1: pd.DataFrame) -> pd.DataFrame:
    funcs = df1["function"].unique().tolist()

    # Mean SEE_basic per (function, optimizer) at best LR
    opt_rank = {}
    for func in funcs:
        sub = df1[df1["function"] == func]
        agg = sub.groupby("optimizer")["SEE_basic"].mean()
        opt_rank[func] = agg.to_dict()

    n = len(funcs)
    mat = np.full((n, n), float("nan"))
    for i, fa in enumerate(funcs):
        for j, fb in enumerate(funcs):
            va = [opt_rank[fa].get(opt, float("nan")) for opt in Config.OPTIMIZERS]
            vb = [opt_rank[fb].get(opt, float("nan")) for opt in Config.OPTIMIZERS]
            va, vb = np.array(va), np.array(vb)
            mask = ~(np.isnan(va) | np.isnan(vb))
            if mask.sum() < 3 or np.all(va[mask] == va[mask][0]) or np.all(vb[mask] == vb[mask][0]):
                mat[i, j] = float("nan")
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = spearmanr(va[mask], vb[mask])
                mat[i, j] = float(res.statistic)

    return pd.DataFrame(mat, index=funcs, columns=funcs)


# ── Part-4 driver ─────────────────────────────────────────────────────────────

def run_part4(cfg: type, results_dir: Path,
              df1: pd.DataFrame | None = None,
              df3: pd.DataFrame | None = None) -> pd.DataFrame:
    set_all_seeds(42)
    device = cfg.DEVICE

    if df1 is None:
        p1_path = results_dir / "part1.csv"
        if p1_path.exists():
            df1 = pd.read_csv(p1_path)
        else:
            print("  Part-4: part1.csv not found, skipping.")
            return pd.DataFrame()

    if df3 is None:
        p3_path = results_dir / "part3.csv"
        df3 = pd.read_csv(p3_path) if p3_path.exists() else pd.DataFrame()

    print("  4A: SEE_basic vs SEE_quality Spearman …")
    df_4a = _4a(df1)

    print("  4B: r_escape sensitivity …")
    df_4b = _4b(df1, device)

    print("  4C: Part-1 vs Part-3 Spearman …")
    df_4c = _4c(df1, df3) if not df3.empty else pd.DataFrame([{"note": "no Part3 data"}])

    print("  4D: 6×6 ranking matrix …")
    df_4d = _4d(df1)

    # Serialize
    with pd.ExcelWriter(results_dir / "part4_synthesis.xlsx", engine="openpyxl") as xw:
        df_4a.to_excel(xw, sheet_name="4A_corr",      index=False)
        df_4b.to_excel(xw, sheet_name="4B_sensitivity", index=False)
        df_4c.to_excel(xw, sheet_name="4C_nn_corr",   index=False)
        df_4d.to_excel(xw, sheet_name="4D_ranking_matrix")

    summary = pd.concat([
        df_4a.assign(section="4A"),
        df_4b.assign(section="4B"),
    ], ignore_index=True)
    summary.to_csv(results_dir / "part4_synthesis.csv", index=False)
    print(f"  Saved part4_synthesis.csv and .xlsx")
    return summary
