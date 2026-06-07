"""
Automatic saddle point finder for 2D benchmark functions.

Pipeline:
1. Grid of starting points over the domain
2. Newton refinement via scipy.optimize.fsolve from each start (∇f=0)
3. Hessian verification: must have ≥1 positive AND ≥1 negative eigenvalue
4. Deduplication within radius SADDLE_DEDUP_RADIUS

fsolve is run in parallel (joblib) for speed.
"""

import numpy as np
import torch
import torch.autograd.functional as taf
from scipy.optimize import fsolve
from joblib import Parallel, delayed

import config
from src.functions.classical_2d import get_function_2d


def _grad_np(xy: np.ndarray, fn_name: str) -> np.ndarray:
    """Compute ∇f at xy. First arg is variable (scipy fsolve convention)."""
    # Re-get fn inside to keep this function picklable for joblib
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    from src.functions.classical_2d import get_function_2d
    fn = get_function_2d(fn_name)
    x = torch.tensor(np.asarray(xy, dtype=np.float64), requires_grad=True)
    fn(x).backward()
    return x.grad.detach().numpy()


def _try_one_start(start: np.ndarray, fn_name: str,
                   x_lo: float, x_hi: float,
                   y_lo: float, y_hi: float) -> list:
    """
    Run fsolve from one start. Returns list of candidate saddle arrays.
    Returns empty list if fsolve fails or point is not a saddle.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    import numpy as np, torch
    import torch.autograd.functional as taf
    from src.functions.classical_2d import get_function_2d

    fn = get_function_2d(fn_name)

    def grad_fn(xy):
        x = torch.tensor(np.asarray(xy, dtype=np.float64), requires_grad=True)
        fn(x).backward()
        return x.grad.detach().numpy()

    def hessian_eigs(xy):
        x = torch.tensor(np.asarray(xy, dtype=np.float64), requires_grad=True)
        H = taf.hessian(fn, x)
        H_np = H.detach().numpy().reshape(2, 2)
        return np.linalg.eigvalsh(H_np)

    try:
        sol, _, ier, _ = fsolve(grad_fn, start, full_output=True)
        if ier != 1:
            return []
        g_norm = np.linalg.norm(grad_fn(sol))
        if g_norm > 1e-5:
            return []
        # Must lie within domain (allow small overshoot)
        margin = 0.5
        if (sol[0] < x_lo - margin or sol[0] > x_hi + margin or
                sol[1] < y_lo - margin or sol[1] > y_hi + margin):
            return []
        eigs = hessian_eigs(sol)
        if eigs.min() >= -1e-6 or eigs.max() <= 1e-6:
            return []  # not a saddle
        return [sol.copy()]
    except Exception:
        return []


def find_saddles_2d(fn_name: str, grid_size: int = None) -> list:
    """
    Find all saddle points for a 2D benchmark function.

    Returns list of (saddle_x, saddle_y) tuples (Python floats).
    """
    grid_size = grid_size or config.GRID_SIZE
    domain = config.FUNCTION_DOMAINS[fn_name]
    x_lo, x_hi = domain[0]
    y_lo, y_hi = domain[1]

    xs = np.linspace(x_lo, x_hi, grid_size)
    ys = np.linspace(y_lo, y_hi, grid_size)
    starts = [np.array([xi, yi]) for xi in xs for yi in ys]

    # Parallel fsolve from each starting point
    n_jobs = min(config.N_JOBS, len(starts))
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(_try_one_start)(s, fn_name, x_lo, x_hi, y_lo, y_hi)
        for s in starts
    )

    # Flatten and deduplicate
    all_pts = [pt for batch in results for pt in batch]
    unique = []
    for pt in all_pts:
        if not any(np.linalg.norm(pt - kept) < config.SADDLE_DEDUP_RADIUS
                   for kept in unique):
            unique.append(pt)

    return [(float(np.clip(pt[0], x_lo, x_hi)),
             float(np.clip(pt[1], y_lo, y_hi)))
            for pt in unique]


def cap_saddles(saddles: list, max_saddles: int) -> list:
    if len(saddles) <= max_saddles:
        return saddles
    pts = np.array(saddles)
    dists_to_origin = np.linalg.norm(pts, axis=1)
    selected_idx = [int(np.argmin(dists_to_origin))]
    while len(selected_idx) < max_saddles:
        selected_pts = pts[selected_idx]
        min_dists = []
        for i, pt in enumerate(pts):
            if i in selected_idx:
                min_dists.append(-1)
            else:
                d = np.min(np.linalg.norm(selected_pts - pt, axis=1))
                min_dists.append(d)
        next_idx = int(np.argmax(min_dists))
        selected_idx.append(next_idx)
    return [saddles[i] for i in selected_idx]
