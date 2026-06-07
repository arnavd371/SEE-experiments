"""Hessian eigenvalue computation.

Full Hessian (torch.autograd.functional.hessian) for d <= HESSIAN_DIM_THRESHOLD.
Lanczos via scipy.sparse.linalg.eigsh + HVPs for larger d.
"""

import numpy as np
import torch
import torch.autograd.functional as taf

import config


def _full_hessian_min_eig(fn, x: torch.Tensor) -> float:
    """Exact minimum eigenvalue via full Hessian. Safe for d <= 20."""
    x_req = x.clone().detach().requires_grad_(True)
    H = taf.hessian(fn, x_req)
    H_np = H.detach().cpu().numpy().reshape(x.numel(), x.numel())
    eigs = np.linalg.eigvalsh(H_np)
    return float(eigs.min())


def _hvp(fn, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute H @ v using two backward passes. x must not require grad externally."""
    x_req = x.clone().detach().requires_grad_(True)
    loss = fn(x_req)
    grad = torch.autograd.grad(loss, x_req, create_graph=True)[0]
    # v^T H = d/dx (grad · v)  →  result is H v for symmetric H
    hv = torch.autograd.grad(grad, x_req, grad_outputs=v.detach())[0]
    return hv.detach()


def _lanczos_min_eig(fn, x: torch.Tensor, k: int = None) -> float:
    """Minimum eigenvalue via Lanczos. Safe for any d."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    k = k or config.LANCZOS_K
    d = x.numel()
    k_actual = min(k, max(1, d - 1))

    x_cpu = x.detach().cpu()

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=x_cpu.dtype, device='cpu')
        hv = _hvp(fn, x_cpu, v)
        return hv.cpu().numpy().astype(np.float64)

    A = LinearOperator((d, d), matvec=matvec, dtype=np.float64)
    try:
        eigs, _ = eigsh(A, k=k_actual, which='SA', tol=1e-3,
                        maxiter=d * 20, ncv=min(d, max(2 * k_actual + 1, 20)))
        return float(eigs.min())
    except Exception:
        # Fallback: single power iteration estimate
        return float('nan')


def compute_min_eigenvalue(fn, x: torch.Tensor, d: int = None) -> float:
    """
    Return the minimum eigenvalue of the Hessian of fn at x.

    Uses the exact Hessian for d <= HESSIAN_DIM_THRESHOLD, Lanczos otherwise.
    """
    d = d or x.numel()
    if d <= config.HESSIAN_DIM_THRESHOLD:
        return _full_hessian_min_eig(fn, x)
    return _lanczos_min_eig(fn, x)


def compute_top_k_eigenvalues(fn, x: torch.Tensor, k: int = None) -> np.ndarray:
    """Return the k smallest-algebraic eigenvalues (most negative first)."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    k = k or config.LANCZOS_K
    d = x.numel()
    k_actual = min(k, max(1, d - 1))

    x_cpu = x.detach().cpu()

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=x_cpu.dtype, device='cpu')
        hv = _hvp(fn, x_cpu, v)
        return hv.cpu().numpy().astype(np.float64)

    A = LinearOperator((d, d), matvec=matvec, dtype=np.float64)
    try:
        eigs, _ = eigsh(A, k=k_actual, which='SA', tol=1e-3,
                        maxiter=d * 20, ncv=min(d, max(2 * k_actual + 1, 20)))
        return np.sort(eigs)
    except Exception:
        return np.array([float('nan')] * k_actual)


def model_hvp_flat(model, loss_val: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
    """
    Compute H @ v for neural network loss, where v is a flat parameter vector.
    loss_val must have been computed with create_graph=True.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss_val, params, create_graph=True, allow_unused=True)
    grads = [g if g is not None else torch.zeros_like(p)
             for g, p in zip(grads, params)]
    grad_flat = torch.cat([g.reshape(-1) for g in grads])

    hvp = torch.autograd.grad(grad_flat, params, grad_outputs=v_flat.reshape(-1),
                               retain_graph=False, allow_unused=True)
    hvp = [h if h is not None else torch.zeros_like(p)
           for h, p in zip(hvp, params)]
    return torch.cat([h.reshape(-1) for h in hvp]).detach()


def model_min_eigenvalue(model, loss_fn, data, device, k: int = None) -> tuple:
    """
    Compute (min_eigenvalue, max_eigenvalue) for NN model via Lanczos.
    Returns (nan, nan) on failure.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    k = k or config.NN_LANCZOS_K
    params = [p for p in model.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    k_actual = min(k, max(1, d - 1))

    def matvec(v_np: np.ndarray) -> np.ndarray:
        v = torch.tensor(v_np, dtype=torch.float32, device=device)
        x_data, y_data = data
        model.zero_grad()
        out = model(x_data)
        loss = loss_fn(out, y_data)
        loss_cg = loss  # need create_graph for HVP
        # recompute with create_graph
        model.zero_grad()
        out2 = model(x_data)
        loss2 = loss_fn(out2, y_data)
        grads = torch.autograd.grad(loss2, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p)
                 for g, p in zip(grads, params)]
        grad_flat = torch.cat([g.reshape(-1) for g in grads])
        hv = torch.autograd.grad(grad_flat, params, grad_outputs=v.reshape(-1),
                                  retain_graph=False, allow_unused=True)
        hv = [h if h is not None else torch.zeros_like(p)
              for h, p in zip(hv, params)]
        return torch.cat([h.reshape(-1) for h in hv]).detach().cpu().numpy().astype(np.float64)

    A = LinearOperator((d, d), matvec=matvec, dtype=np.float64)
    try:
        eigs_lo, _ = eigsh(A, k=k_actual, which='SA', tol=1e-2,
                           maxiter=d * 10, ncv=min(d, max(2 * k_actual + 1, 20)))
        eigs_hi, _ = eigsh(A, k=k_actual, which='LA', tol=1e-2,
                           maxiter=d * 10, ncv=min(d, max(2 * k_actual + 1, 20)))
        return float(eigs_lo.min()), float(eigs_hi.max())
    except Exception:
        return float('nan'), float('nan')
