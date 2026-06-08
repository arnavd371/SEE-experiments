"""
Core SEE metric computation.

SEE_basic   = P_escape / tau_escape_mean
SEE_quality = P_quality / tau_quality_mean
diverge_rate = P_diverge
"""
import numpy as np
from .bootstrap import bootstrap_ci


def compute_see(escaped_mask: np.ndarray,
                escape_steps: np.ndarray,
                quality_mask: np.ndarray,
                quality_steps: np.ndarray,
                diverged_mask: np.ndarray,
                n_bootstrap: int = 1000) -> dict:
    """
    All arrays length N.
    *_steps: step index (1-based) when condition first met; NaN when not.
    Returns dict of SEE_basic, SEE_quality, CI bounds, rates, taus.
    """
    N = len(escaped_mask)
    escaped_mask  = np.asarray(escaped_mask,  dtype=bool)
    quality_mask  = np.asarray(quality_mask,  dtype=bool)
    diverged_mask = np.asarray(diverged_mask, dtype=bool)
    escape_steps  = np.asarray(escape_steps,  dtype=float)
    quality_steps = np.asarray(quality_steps, dtype=float)

    P_escape  = escaped_mask.sum()  / N
    P_quality = quality_mask.sum()  / N
    P_diverge = diverged_mask.sum() / N
    P_stuck   = ((~escaped_mask) & (~diverged_mask) & (~quality_mask)).sum() / N

    tau_e = float(np.nanmean(escape_steps[escaped_mask]))   if escaped_mask.any()  else float("nan")
    tau_q = float(np.nanmean(quality_steps[quality_mask]))  if quality_mask.any()  else float("nan")

    see_basic   = (P_escape  / tau_e)  if (P_escape  > 0 and not np.isnan(tau_e))  else 0.0
    see_quality = (P_quality / tau_q) if (P_quality > 0 and not np.isnan(tau_q)) else 0.0

    # Bootstrap CIs
    esc_step_for_boot = np.where(escaped_mask,  escape_steps,  0.0)
    qlt_step_for_boot = np.where(quality_mask, quality_steps, 0.0)

    ci_e_lo, ci_e_hi = bootstrap_ci(escaped_mask, esc_step_for_boot, n_bootstrap=n_bootstrap)
    ci_q_lo, ci_q_hi = bootstrap_ci(quality_mask, qlt_step_for_boot, n_bootstrap=n_bootstrap)

    return {
        "SEE_basic":        see_basic,
        "SEE_basic_CI_lo":  ci_e_lo,
        "SEE_basic_CI_hi":  ci_e_hi,
        "SEE_quality":      see_quality,
        "SEE_quality_CI_lo": ci_q_lo,
        "SEE_quality_CI_hi": ci_q_hi,
        "P_escape":   P_escape,
        "P_quality":  P_quality,
        "diverge_rate": P_diverge,
        "stuck_rate":   P_stuck,
        "tau_escape_mean":  tau_e,
        "tau_quality_mean": tau_q,
    }
