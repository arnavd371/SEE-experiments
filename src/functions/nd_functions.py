"""
n-D benchmark functions for Part 2.
All accept (N, d) and return (N,).
"""
import math
import torch


def synthetic_saddle(x: torch.Tensor, k: int) -> torch.Tensor:
    """f(x) = -||x[:k]||^2 + ||x[k:]||^2.  Exact saddle at origin."""
    neg = (x[..., :k]**2).sum(dim=-1)
    pos = (x[..., k:]**2).sum(dim=-1)
    return -neg + pos


def synthetic_saddle_lambda_min() -> float:
    return -2.0


def styblinski_nd(x: torch.Tensor) -> torch.Tensor:
    """Separable Styblinski-Tang extended to d dimensions."""
    return 0.5 * (x**4 - 16*x**2 + 5*x).sum(dim=-1)


# 1-D stationary points of Styblinski-Tang component:
#   f'(t) = 2t^3 - 16t + 2.5 = 0
# Approximately: t ≈ -2.9035 (min), 0.1563 (max/saddle-dir), 2.7472 (min)
_STYBLT_SADDLE_DIR  = 0.15634896   # 1-D local max → negative f''
_STYBLT_MIN_DIR_POS = 2.74720720   # 1-D local min (positive side)

# f''(t) = 6t^2 - 16
def _styblt_fpp(t: float) -> float:
    return 6*t**2 - 16

# lambda_min of Styblinski-nD saddle = f''(saddle_dir) (most negative)
STYBLT_LAMBDA_MIN = _styblt_fpp(_STYBLT_SADDLE_DIR)   # ≈ -15.85


def styblinski_saddle_point(d: int, k: int) -> torch.Tensor:
    """Saddle with k negative-curvature dims at _STYBLT_SADDLE_DIR, rest at min."""
    x = torch.full((d,), _STYBLT_MIN_DIR_POS, dtype=torch.float64)
    x[:k] = _STYBLT_SADDLE_DIR
    return x


def domain_diameter_nd(d: int, lo: float = -5.0, hi: float = 5.0) -> float:
    return math.sqrt(d) * (hi - lo)
