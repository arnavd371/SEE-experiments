"""
High-dimensional functions for Part 2.
All functions accept a 1-D tensor of shape (d,).
"""
import math
import torch


def rastrigin_nd(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 10n + Σ[xi²-10cos(2πxi)]"""
    n = x.shape[0]
    return 10 * n + torch.sum(x ** 2 - 10 * torch.cos(2 * math.pi * x))


def styblinski_nd(x: torch.Tensor) -> torch.Tensor:
    """f(x) = 0.5Σ(xi⁴-16xi²+5xi)"""
    return 0.5 * torch.sum(x ** 4 - 16 * x ** 2 + 5 * x)


def synthetic_saddle(k: int):
    """
    Returns a closure f(x) = -Σi<k xi² + Σi>=k xi²
    Saddle at origin; lambda_min = -2 analytically, r ≈ 0.354.
    k controls saddle index (number of negative-curvature dimensions).
    """
    def _f(x: torch.Tensor) -> torch.Tensor:
        neg = -torch.sum(x[:k] ** 2)
        pos = torch.sum(x[k:] ** 2)
        return neg + pos
    _f.__name__ = f'SyntheticSaddle_k{k}'
    return _f


ND_FUNCTIONS = {
    'Rastrigin-nD': rastrigin_nd,
    'Styblinski-nD': styblinski_nd,
}
