"""Vectorized percentile bootstrap CI."""
import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    statistic=np.mean,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Returns (lo, hi) percentile bootstrap CI for statistic(values).
    Fully vectorized: generates all resamples at once.
    """
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    # (n_resamples, n) index matrix — vectorized
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_samples = values[idx]          # (n_resamples, n)
    boot_stats = statistic(boot_samples, axis=1)   # (n_resamples,)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi
