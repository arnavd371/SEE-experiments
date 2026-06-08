"""
Saddle point finding: grid search → refinement → deduplication → cap.
"""
from __future__ import annotations
import math
from typing import Callable, List, Tuple

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


def find_saddles_2d(
    fn: Callable,
    domain: Tuple[float, float, float, float],  # x_lo, x_hi, y_lo, y_hi
    device,
    grid_n: int = 200,
    grad_thresh: float = 0.5,
    dedup_radius: float = 0.5,
    max_saddles: int = 3,
) -> List[dict]:
    """
    Returns up to max_saddles saddle dicts with keys:
      x_s, r, lambda_min
    """
    x_lo, x_hi, y_lo, y_hi = domain
    xs = np.linspace(x_lo, x_hi, grid_n)
    ys = np.linspace(y_lo, y_hi, grid_n)
    XX, YY = np.meshgrid(xs, ys)
    grid_pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (G, 2)

    domain_diameter = math.sqrt((x_hi - x_lo) ** 2 + (y_hi - y_lo) ** 2)
    r_max = 0.25 * domain_diameter

    # --- grid scan ---
    candidates = []
    for pt in grid_pts:
        g = _grad_np(fn, pt, device)
        if np.linalg.norm(g) < grad_thresh:
            H = _hessian_np(fn, pt, device)
            eigs = np.linalg.eigvalsh(H)
            if eigs.min() < 0 < eigs.max():
                candidates.append(pt.copy())

    if not candidates:
        return []

    # --- refine with fsolve ---
    refined = []
    for pt in candidates:
        def grad_fn(xy):
            return _grad_np(fn, xy, device)
        try:
            sol = scipy.optimize.fsolve(grad_fn, pt, full_output=True)
            x_ref = sol[0]
            # Verify it's in domain
            if not (x_lo <= x_ref[0] <= x_hi and y_lo <= x_ref[1] <= y_hi):
                continue
            g = _grad_np(fn, x_ref, device)
            if np.linalg.norm(g) > 0.05:
                continue
            H = _hessian_np(fn, x_ref, device)
            eigs = np.linalg.eigvalsh(H)
            if eigs.min() < 0 < eigs.max():
                refined.append(x_ref)
        except Exception:
            continue

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
        # Start with saddle closest to center
        dists_to_center = [np.linalg.norm(u - center) for u in unique]
        first_idx = int(np.argmin(dists_to_center))
        selected = [unique[first_idx]]
        remaining = [u for i, u in enumerate(unique) if i != first_idx]
        while len(selected) < max_saddles and remaining:
            # Add saddle farthest from all selected
            far_idx = max(range(len(remaining)),
                          key=lambda i: min(np.linalg.norm(remaining[i] - s)
                                            for s in selected))
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
