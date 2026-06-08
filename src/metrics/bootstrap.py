"""Vectorized bootstrap 95% CIs for SEE metrics."""
import numpy as np


def _see_from_samples(escaped: np.ndarray, steps: np.ndarray) -> np.ndarray:
    """
    escaped: bool (B, N)
    steps:   float (B, N)  — escape/quality step (large sentinel when not achieved)
    Returns SEE values (B,).
    """
    n = escaped.shape[1]
    count = escaped.sum(axis=1).astype(float)           # (B,)
    sum_steps = (steps * escaped).sum(axis=1).astype(float)  # (B,)
    # SEE = (count/n) / (sum_steps/count)  = count^2 / (n * sum_steps)
    see = np.where((count > 0) & (sum_steps > 0),
                   count**2 / (n * sum_steps),
                   0.0)
    return see


def bootstrap_ci(escaped_mask: np.ndarray,
                 step_arr: np.ndarray,
                 n_bootstrap: int = 1000,
                 seed: int = 0) -> tuple:
    """
    Parameters
    ----------
    escaped_mask : bool (N,)
    step_arr     : float (N,)  — step when condition first met; NaN or large for non-occurrences
    Returns (ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(escaped_mask)
    idx = rng.integers(0, n, size=(n_bootstrap, n))

    e = escaped_mask.astype(bool)[idx]              # (B, N)
    # Replace NaN with 0 in step array (they won't contribute because escaped=False)
    s = np.nan_to_num(step_arr, nan=0.0)[idx]       # (B, N)

    see_vals = _see_from_samples(e, s)
    return float(np.percentile(see_vals, 2.5)), float(np.percentile(see_vals, 97.5))
