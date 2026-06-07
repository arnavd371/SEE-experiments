"""N-dimensional benchmark functions and the synthetic saddle for Part 2."""

import math
import torch


def rastrigin_nd(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    return 10.0 * n + (x ** 2 - 10.0 * torch.cos(2.0 * math.pi * x)).sum()


def styblinski_nd(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (x ** 4 - 16.0 * x ** 2 + 5.0 * x).sum()


def ackley_nd(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    sum_sq = (x ** 2).mean()
    sum_cos = torch.cos(2.0 * math.pi * x).mean()
    return (-20.0 * torch.exp(-0.2 * torch.sqrt(sum_sq))
            - torch.exp(sum_cos) + 20.0 + math.e)


def make_synthetic_saddle(k: int):
    """
    Factory: returns f(x) = -sum(x[:k]^2) + sum(x[k:]^2).
    Saddle at origin analytically.
    k: number of negative-curvature directions.
    """
    def synthetic_saddle(x: torch.Tensor) -> torch.Tensor:
        return -x[:k].pow(2).sum() + x[k:].pow(2).sum()
    synthetic_saddle.__name__ = f'Synthetic-Saddle-k{k}'
    return synthetic_saddle


_REGISTRY = {
    'Rastrigin-nD': rastrigin_nd,
    'Styblinski-nD': styblinski_nd,
    'Ackley-nD': ackley_nd,
}


def get_nd_function(name: str, k: int = None):
    """
    name: one of 'Rastrigin-nD', 'Styblinski-nD', 'Ackley-nD', 'Synthetic-Saddle'
    k   : saddle index (required when name == 'Synthetic-Saddle')
    """
    if name == 'Synthetic-Saddle':
        if k is None:
            raise ValueError('k must be specified for Synthetic-Saddle')
        return make_synthetic_saddle(k)
    if name not in _REGISTRY:
        raise KeyError(f'Unknown nD function: {name!r}')
    return _REGISTRY[name]


def saddle_location_nd(name: str, d: int, k: int = None) -> torch.Tensor:
    """Return the analytically known saddle location for nD functions."""
    if name == 'Synthetic-Saddle':
        return torch.zeros(d, dtype=torch.float64)
    # For Rastrigin, Styblinski-Tang, Ackley, origin is a saddle/local-min
    return torch.zeros(d, dtype=torch.float64)


def saddle_indices_for_dim(d: int) -> list:
    """Return the list of k values to test at dimension d."""
    candidates = [1, d // 4, d // 2, 3 * d // 4, d - 1]
    # Deduplicate while preserving order, keep valid k in [1, d-1]
    seen = set()
    result = []
    for k in candidates:
        k = max(1, min(k, d - 1))
        if k not in seen:
            seen.add(k)
            result.append(k)
    return result
