"""Pairwise optimizer comparison: Wilcoxon signed-rank + Cohen's d."""
import numpy as np
import warnings


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    sd = diff.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diff.mean() / sd)


def pairwise_wilcoxon(scores: dict,
                      best_name: str,
                      n_comparisons: int | None = None) -> dict:
    """
    scores: {optimizer_name: array of per-trial scalar values}
    best_name: reference optimizer
    Returns: {opt_name: {'p': bonferroni-corrected p, 'd': cohen_d}}
    """
    try:
        import pingouin as pg
    except ImportError:
        return {}

    best_arr = np.array(scores[best_name], dtype=float)
    n_comp = n_comparisons or (len(scores) - 1)
    results = {}

    for name, arr in scores.items():
        if name == best_name:
            continue
        arr = np.array(arr, dtype=float)
        if len(arr) != len(best_arr):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = pg.wilcoxon(best_arr, arr, alternative="two-sided")
            p_raw = float(res["p-val"].iloc[0])
        except Exception:
            p_raw = float("nan")
        p_corr = min(1.0, p_raw * n_comp)
        results[name] = {
            "p": p_corr,
            "d": cohens_d(best_arr, arr),
        }
    return results
