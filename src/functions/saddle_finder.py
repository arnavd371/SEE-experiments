"""
Saddle-point finder for 2-D functions.

Algorithm
---------
1. 200×200 grid; candidates where ||grad|| < 2.0 AND Hessian has mixed signs.
2. Refine each candidate with scipy.optimize.fsolve (solve grad=0).
3. Verify mixed-sign Hessian at refined point.
4. Dedupe within distance 0.5.
5. Cap at 3 saddles via greedy max-distance from domain centre.
Fallback: if zero candidates, repeat on 50×50 grid with threshold 50.0.
"""
import numpy as np
import torch
from scipy.optimize import fsolve


def _torch_grad_hess(f, x_np, device):
    """Return numpy grad (2,) and Hessian (2,2) at x_np."""
    x = torch.tensor(x_np, dtype=torch.float64, device=device).unsqueeze(0).requires_grad_(True)
    val = f(x)[0]
    g = torch.autograd.grad(val, x, create_graph=True)[0]  # (1, 2)
    H = torch.zeros(2, 2, dtype=torch.float64, device=device)
    for i in range(2):
        H[i] = torch.autograd.grad(g[0, i], x, retain_graph=(i < 1))[0][0]
    return g[0].detach().cpu().numpy(), H.detach().cpu().numpy()


def _has_mixed_signs(H: np.ndarray) -> bool:
    eigs = np.linalg.eigvalsh(H)
    return eigs[0] < 0 and eigs[-1] > 0


def find_saddles(f, domain, device, grid_size=200, grad_thresh=2.0):
    xl, xh, yl, yh = domain
    xs = np.linspace(xl, xh, grid_size)
    ys = np.linspace(yl, yh, grid_size)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.stack([XX.ravel(), YY.ravel()], axis=1)  # (G, 2)

    # Batch gradient on grid
    with torch.no_grad():
        x_t = torch.tensor(pts, dtype=torch.float64, device=device)
    x_t.requires_grad_(True)
    fval = f(x_t.float()).double()
    grad_t = torch.autograd.grad(fval.sum(), x_t)[0]  # (G, 2)
    gnorms = grad_t.norm(dim=1).detach().cpu().numpy()

    candidate_mask = gnorms < grad_thresh
    candidate_pts = pts[candidate_mask]

    if candidate_pts.shape[0] == 0:
        # Fallback: coarser grid, higher threshold
        return find_saddles(f, domain, device, grid_size=50, grad_thresh=50.0)

    # Refine candidates with fsolve, verify, dedupe
    verified = []
    for c in candidate_pts:
        def grad_eq_zero(xy):
            try:
                g, _ = _torch_grad_hess(f, np.array(xy, dtype=np.float64), device)
                return g
            except Exception:
                return np.array([1e6, 1e6])

        try:
            sol = fsolve(grad_eq_zero, c, full_output=True)
            xr = sol[0]
        except Exception:
            continue

        # Check point is inside domain
        if xr[0] < xl or xr[0] > xh or xr[1] < yl or xr[1] > yh:
            continue

        # Verify mixed Hessian signs
        try:
            g_ref, H_ref = _torch_grad_hess(f, xr, device)
        except Exception:
            continue

        if np.linalg.norm(g_ref) > 1e-3:
            continue
        if not _has_mixed_signs(H_ref):
            continue

        verified.append(xr)

    if len(verified) == 0:
        if grad_thresh < 50.0:
            return find_saddles(f, domain, device, grid_size=50, grad_thresh=50.0)
        return []

    # Dedupe within 0.5
    unique = [verified[0]]
    for p in verified[1:]:
        if all(np.linalg.norm(p - q) > 0.5 for q in unique):
            unique.append(p)

    # Cap at 3: greedy max-distance from domain centre
    cx, cy = 0.5*(xl + xh), 0.5*(yl + yh)
    centre = np.array([cx, cy])
    if len(unique) <= 3:
        return unique

    # Sort by distance from centre descending; greedily pick 3 most spread
    chosen = [max(unique, key=lambda p: np.linalg.norm(p - centre))]
    for _ in range(2):
        best, best_dist = None, -1
        for p in unique:
            if any(np.allclose(p, c) for c in chosen):
                continue
            d = min(np.linalg.norm(p - q) for q in chosen)
            if d > best_dist:
                best_dist, best = d, p
        if best is not None:
            chosen.append(best)
    return chosen


def lambda_min_2d(f, x_np, device):
    """Smallest Hessian eigenvalue at x_np (numpy array of shape (2,))."""
    _, H = _torch_grad_hess(f, x_np, device)
    return float(np.linalg.eigvalsh(H)[0])
