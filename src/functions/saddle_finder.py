"""
Saddle point finding: grid search → refinement → deduplication → cap.
"""
from __future__ import annotations
import math
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.optimize
import torch
import torch.autograd.functional as AF


def _grad_np(fn: Callable, x_np: np.ndarray, device) -> np.ndarray:
    x = torch.tensor(x_np, dtype=torch.float32, device=device, requires_grad=True)
    v = fn(x)
    v.backward()
    return x.grad.detach().cpu().numpy().astype(np.float64)


def _hessian_np(fn: Callable, x_np: np.ndarray, device) -> np.ndarray:
    x = torch.tensor(x_np, dtype=torch.float32, device=device)
    H = AF.hessian(fn, x)
    return H.detach().cpu().numpy().reshape(2, 2)


def _make_grid(x_lo, x_hi, y_lo, y_hi, n):
    xs = np.linspace(x_lo, x_hi, n)
    ys = np.linspace(y_lo, y_hi, n)
    XX, YY = np.meshgrid(xs, ys)
    return np.stack([XX.ravel(), YY.ravel()], axis=1)


def _scan_candidates(
    fn: Callable,
    pts: np.ndarray,
    grad_thresh: float,
    device,
    max_candidates: int = 200,
) -> List[np.ndarray]:
    """Grid scan: return points where |∇f| < thresh AND Hessian has mixed eigenvalues.

    Caps at max_candidates (sorted by gradient norm) to limit downstream fsolve calls.
    """
    grad_pts = []
    for pt in pts:
        g = _grad_np(fn, pt, device)
        gn = np.linalg.norm(g)
        if gn < grad_thresh:
            grad_pts.append((gn, pt.copy()))

    # Sort by gradient norm; if too many, keep best max_candidates before Hessian check
    grad_pts.sort(key=lambda x: x[0])
    grad_pts = grad_pts[:max_candidates]

    candidates = []
    for _, pt in grad_pts:
        H = _hessian_np(fn, pt, device)
        eigs = np.linalg.eigvalsh(H)
        if eigs.min() < 0 < eigs.max():
            candidates.append(pt)
    return candidates


def find_saddles_2d(
    fn: Union[str, Callable],
    domain: Optional[Tuple[float, float, float, float]] = None,
    device=None,
    grid_n: int = 200,
    grad_thresh: float = 2.0,
    dedup_radius: float = 0.5,
    max_saddles: int = 3,
) -> List[dict]:
    """
    Returns up to max_saddles saddle dicts with keys: x_s, r, lambda_min.

    fn may be a string name (looked up from FUNCTIONS_2D) or a callable.
    If fn is a string, domain is inferred from the registry.
    """
    if isinstance(fn, str):
        from src.functions.classical_2d import FUNCTIONS_2D
        fn_callable, domain = FUNCTIONS_2D[fn]
        fn = fn_callable

    if device is None:
        device = torch.device('cpu')

    x_lo, x_hi, y_lo, y_hi = domain
    domain_diameter = math.sqrt((x_hi - x_lo) ** 2 + (y_hi - y_lo) ** 2)
    r_max = 0.25 * domain_diameter

    # --- primary scan: fine grid, tight threshold ---
    grid_pts = _make_grid(x_lo, x_hi, y_lo, y_hi, grid_n)
    candidates = _scan_candidates(fn, grid_pts, grad_thresh, device)
    print(f"  Primary scan: {len(candidates)} candidates (thresh={grad_thresh}, grid={grid_n}²)")

    # --- fallback: coarser grid, relaxed threshold for high-curvature functions ---
    # (e.g. Rastrigin has saddle curvature ~390, so nearby grid points have |∇f|~14)
    if not candidates:
        fallback_n = 50
        fallback_thresh = 50.0
        coarse_pts = _make_grid(x_lo, x_hi, y_lo, y_hi, fallback_n)
        candidates = _scan_candidates(fn, coarse_pts, fallback_thresh, device)
        print(f"  Fallback scan: {len(candidates)} candidates "
              f"(thresh={fallback_thresh}, grid={fallback_n}²)")

    if not candidates:
        return []

    # --- refine with fsolve ---
    def grad_fn(xy):
        return _grad_np(fn, xy, device)

    refined = []
    for pt in candidates:
        try:
            x_ref = scipy.optimize.fsolve(grad_fn, pt)
            if not (x_lo <= x_ref[0] <= x_hi and y_lo <= x_ref[1] <= y_hi):
                continue
            if np.linalg.norm(grad_fn(x_ref)) > 0.05:
                continue
            H = _hessian_np(fn, x_ref, device)
            eigs = np.linalg.eigvalsh(H)
            if eigs.min() < 0 < eigs.max():
                refined.append(x_ref)
        except Exception:
            continue

    print(f"  After fsolve + Hessian check: {len(refined)} refined saddles")

    if not refined:
        return []

    # --- deduplicate ---
    unique = []
    for pt in refined:
        if all(np.linalg.norm(pt - u) > dedup_radius for u in unique):
            unique.append(pt)

    # --- cap at max_saddles with greedy max-distance selection ---
    center = np.array([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2])
    if len(unique) <= 1:
        selected = unique
    else:
        dists_to_center = [np.linalg.norm(u - center) for u in unique]
        first_idx = int(np.argmin(dists_to_center))
        selected = [unique[first_idx]]
        remaining = [u for i, u in enumerate(unique) if i != first_idx]
        while len(selected) < max_saddles and remaining:
            far_idx = max(
                range(len(remaining)),
                key=lambda i: min(np.linalg.norm(remaining[i] - s) for s in selected),
            )
            selected.append(remaining.pop(far_idx))

    # --- compute r per saddle ---
    saddles = []
    for x_s in selected:
        H = _hessian_np(fn, x_s, device)
        eigs = np.linalg.eigvalsh(H)
        lmin = eigs.min()
        r = min(r_max, 0.5 / math.sqrt(abs(lmin) + 1e-6))
        saddles.append({'x_s': x_s, 'r': float(r), 'lambda_min': float(lmin)})

    return saddles


def compute_r(lambda_min: float, r_max: float) -> float:
    return min(r_max, 0.5 / math.sqrt(abs(lambda_min) + 1e-6))
