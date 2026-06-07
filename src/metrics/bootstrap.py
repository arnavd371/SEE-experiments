"""Vectorized percentile bootstrap confidence intervals for SEE metrics."""

import numpy as np
import config


def _compute_see_basic(outcomes: np.ndarray, taus: np.ndarray) -> float:
    """SEE_basic: P_esc = (local_min + diverge)/N, τ over all escaping."""
    n = len(outcomes)
    esc_mask = (outcomes == 'local_min') | (outcomes == 'diverge')
    p_esc = esc_mask.sum() / n
    if p_esc == 0:
        return 0.0
    tau_avg = taus[esc_mask].mean()
    return float(p_esc / tau_avg) if tau_avg > 0 else 0.0


def _compute_see_quality(outcomes: np.ndarray, taus: np.ndarray) -> float:
    """SEE_quality: P_esc = local_min/N, τ over quality-escape trials only."""
    n = len(outcomes)
    min_mask = (outcomes == 'local_min')
    p_esc = min_mask.sum() / n
    if p_esc == 0:
        return 0.0
    tau_avg = taus[min_mask].mean()
    return float(p_esc / tau_avg) if tau_avg > 0 else 0.0


def _batch_see_basic(outcomes_mat: np.ndarray, taus_mat: np.ndarray) -> np.ndarray:
    """
    Vectorized SEE_basic over rows of (R, N) matrices.
    outcomes_mat: dtype object, values 'local_min'/'diverge'/'stuck'
    taus_mat:     dtype float
    Returns: (R,) SEE_basic values.
    """
    R, N = outcomes_mat.shape
    esc_mask = (outcomes_mat == 'local_min') | (outcomes_mat == 'diverge')  # (R, N)
    p_esc = esc_mask.sum(axis=1) / N  # (R,)

    tau_sum = np.where(esc_mask, taus_mat, 0.0).sum(axis=1)  # (R,)
    n_esc = esc_mask.sum(axis=1).clip(min=1)                  # (R,)
    tau_avg = tau_sum / n_esc                                  # (R,)

    see = np.where(p_esc > 0, p_esc / np.where(tau_avg > 0, tau_avg, 1.0), 0.0)
    return see.astype(float)


def _batch_see_quality(outcomes_mat: np.ndarray, taus_mat: np.ndarray) -> np.ndarray:
    """Vectorized SEE_quality over rows of (R, N) matrices."""
    R, N = outcomes_mat.shape
    min_mask = (outcomes_mat == 'local_min')
    p_esc = min_mask.sum(axis=1) / N

    tau_sum = np.where(min_mask, taus_mat, 0.0).sum(axis=1)
    n_min = min_mask.sum(axis=1).clip(min=1)
    tau_avg = tau_sum / n_min

    see = np.where(p_esc > 0, p_esc / np.where(tau_avg > 0, tau_avg, 1.0), 0.0)
    return see.astype(float)


def bootstrap_ci(outcomes: np.ndarray, taus: np.ndarray,
                 metric: str = 'quality',
                 n_resamples: int = None) -> tuple:
    """
    Percentile bootstrap CI for SEE_basic or SEE_quality.

    Parameters
    ----------
    outcomes  : (N,) str array — 'local_min', 'diverge', 'stuck'
    taus      : (N,) float array — steps taken (any value for 'stuck')
    metric    : 'basic' or 'quality'
    n_resamples : defaults to config.BOOTSTRAP_RESAMPLES

    Returns
    -------
    (ci_lo, ci_hi) floats at 2.5th and 97.5th percentiles
    """
    n_resamples = n_resamples or config.BOOTSTRAP_RESAMPLES
    N = len(outcomes)
    if N == 0:
        return 0.0, 0.0

    # Vectorized resampling: (n_resamples, N)
    idx = np.random.randint(0, N, size=(n_resamples, N))
    outcomes_mat = outcomes[idx]
    taus_mat = taus.astype(float)[idx]

    if metric == 'basic':
        see_resamples = _batch_see_basic(outcomes_mat, taus_mat)
    else:
        see_resamples = _batch_see_quality(outcomes_mat, taus_mat)

    return (float(np.percentile(see_resamples, config.BOOTSTRAP_CI_LO)),
            float(np.percentile(see_resamples, config.BOOTSTRAP_CI_HI)))
