"""
Statistical tests: pairwise Wilcoxon signed-rank, Bonferroni, Cohen's d.
"""
from __future__ import annotations
import numpy as np
import pingouin


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    pooled_std = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
                         / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def pairwise_wilcoxon(
    scores_dict: dict[str, np.ndarray],   # {name: per-trial array}
    alpha: float = 0.05,
) -> dict[tuple[str, str], dict]:
    """
    Run Wilcoxon signed-rank for every ordered pair (a, b) where
    a is the 'best' optimizer (highest mean).
    Returns {(best_name, other_name): {p_bonf, significant, cohens_d}}.
    """
    names = list(scores_dict.keys())
    means = {n: np.mean(v) for n, v in scores_dict.items()}
    best = max(means, key=means.get)

    n_pairs = len(names) - 1
    bonf_alpha = alpha / max(n_pairs, 1)
    results = {}

    for name in names:
        if name == best:
            continue
        a = scores_dict[best]
        b = scores_dict[name]
        n = min(len(a), len(b))
        try:
            res = pingouin.wilcoxon(a[:n], b[:n])
            p = float(res['p-val'].iloc[0])
        except Exception:
            p = np.nan
        d = cohens_d(a[:n], b[:n])
        results[(best, name)] = {
            'p_bonf': p,
            'significant': (p < bonf_alpha) if not np.isnan(p) else False,
            'cohens_d': d,
        }

    return results
