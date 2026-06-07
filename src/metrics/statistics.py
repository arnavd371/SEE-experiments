"""Wilcoxon signed-rank tests, Bonferroni correction, and Cohen's d."""

import numpy as np
import warnings


def _per_trial_score(outcomes: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """
    Per-trial score for comparison: 1/tau if local_min, 0 otherwise.
    Higher score = faster quality escape.
    """
    scores = np.zeros(len(outcomes), dtype=float)
    min_mask = (outcomes == 'local_min')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        scores[min_mask] = 1.0 / np.where(taus[min_mask] > 0, taus[min_mask], 1.0)
    return scores


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Pooled Cohen's d between two score arrays."""
    na, nb = len(scores_a), len(scores_b)
    if na < 2 or nb < 2:
        return float('nan')
    var_a = scores_a.var(ddof=1)
    var_b = scores_b.var(ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((scores_a.mean() - scores_b.mean()) / pooled_std)


def wilcoxon_pvalue(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Wilcoxon signed-rank test on paired per-trial scores.
    Pairs are by index; unpaired (different-length) arrays are truncated to min length.
    Returns p-value.
    """
    try:
        import pingouin as pg
        import pandas as pd
    except ImportError:
        from scipy.stats import wilcoxon as _wilcoxon
        n = min(len(scores_a), len(scores_b))
        if n < 10:
            return float('nan')
        diff = scores_a[:n] - scores_b[:n]
        if np.all(diff == 0):
            return 1.0
        _, p = _wilcoxon(scores_a[:n], scores_b[:n], alternative='two-sided',
                         zero_method='wilcox')
        return float(p)

    n = min(len(scores_a), len(scores_b))
    if n < 10:
        return float('nan')
    df = pd.DataFrame({'A': scores_a[:n], 'B': scores_b[:n]})
    result = pg.wilcoxon(df['A'], df['B'], alternative='two-sided')
    # column name differs across pingouin versions: 'p-val' or 'p_val'
    pcol = 'p-val' if 'p-val' in result.columns else 'p_val'
    return float(result[pcol].iloc[0])


def pairwise_wilcoxon(optimizer_names: list,
                      outcomes_dict: dict,
                      taus_dict: dict) -> dict:
    """
    Run pairwise Wilcoxon signed-rank tests with Bonferroni correction.

    Parameters
    ----------
    optimizer_names : list of str
    outcomes_dict   : {opt_name: outcomes_array}
    taus_dict       : {opt_name: taus_array}

    Returns
    -------
    dict: {(opt_a, opt_b): {'p_raw': ..., 'p_bonf': ..., 'cohens_d': ..., 'significant': bool}}
    """
    scores = {opt: _per_trial_score(outcomes_dict[opt], taus_dict[opt])
              for opt in optimizer_names}

    pairs = [(a, b) for i, a in enumerate(optimizer_names)
             for b in optimizer_names[i + 1:]]
    n_pairs = len(pairs)
    bonferroni_threshold = 0.05 / max(n_pairs, 1)

    results = {}
    for a, b in pairs:
        p_raw = wilcoxon_pvalue(scores[a], scores[b])
        p_bonf = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else float('nan')
        d = cohens_d(scores[a], scores[b])
        results[(a, b)] = {
            'p_raw': p_raw,
            'p_bonf': p_bonf,
            'cohens_d': d,
            'significant': (not np.isnan(p_bonf)) and (p_bonf < 0.05),
        }
    return results


def p_vs_best(opt_name: str, best_opt: str,
              outcomes_dict: dict, taus_dict: dict) -> tuple:
    """
    Return (p_value, cohen_d) comparing opt_name vs best_opt.
    """
    scores_opt = _per_trial_score(outcomes_dict[opt_name], taus_dict[opt_name])
    scores_best = _per_trial_score(outcomes_dict[best_opt], taus_dict[best_opt])
    p = wilcoxon_pvalue(scores_opt, scores_best)
    d = cohens_d(scores_opt, scores_best)
    return p, d
