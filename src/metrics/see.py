"""
Saddle Escape Efficiency (SEE) metric computation.

SEE = P_esc / τ_avg

Three variants are always computed:

  SEE_basic:
    P_esc = (local_min_count + diverge_count) / N
    τ_avg = mean escape steps over ALL escaping trials

  SEE_quality:   [PRIMARY METRIC — used in all comparisons]
    P_esc = local_min_count / N
    τ_avg = mean escape steps over quality-escape (local_min) trials only

  SEE_diverge:   [DIAGNOSTIC ONLY]
    = diverge_count / N

KNOWN LIMITATIONS:
1. Hessian computation is O(p²) — infeasible for p > 100K.
   Parts 3 and 4 use gradient norm as saddle proxy; this is an approximation.
2. SEE is a LOCAL metric computed near specific saddle initializations.
   Results may not generalize to all saddles of a given function.
3. τ_avg is unreliable when P_esc < 0.1 (fewer than 50 successful escapes
   out of 500 trials). These configurations are flagged with reliable=False.
4. SEE does not measure escape destination quality. SEE_quality partially
   addresses this (counts only local_min escapes) but does not compare
   minimum values across destinations.
5. The LLM proxy (Part 4) uses gradient norm plateau detection, not true
   Hessian-based saddle detection. This is a proxy; labeled as such in all figures.
6. AdaGrad theory (Antonakopoulos et al. 2022) proves asymptotic saddle
   avoidance but not escape rate. Low empirical SEE for AdaGrad reflects a
   theory-practice gap that SEE helps characterize, not a contradiction.
"""

from __future__ import annotations
import numpy as np
import config
from src.metrics.bootstrap import bootstrap_ci


def compute_see(outcomes: np.ndarray, taus: np.ndarray,
                n_resamples: int = None) -> dict:
    """
    Compute all three SEE variants plus bootstrap CIs from trial outcomes.

    Parameters
    ----------
    outcomes   : (N,) str array — 'local_min', 'diverge', 'stuck'
    taus       : (N,) int/float array — steps taken per trial
    n_resamples: bootstrap resamples (default from config)

    Returns
    -------
    dict with keys: SEE_basic, SEE_quality, SEE_diverge,
                    SEE_basic_CI_lo, SEE_basic_CI_hi,
                    SEE_quality_CI_lo, SEE_quality_CI_hi,
                    escape_min_pct, escape_diverge_pct, stuck_pct,
                    tau_mean, tau_median, tau_std,
                    n_trials, best_loss_at_escape_mean, reliable
    """
    n_resamples = n_resamples or config.BOOTSTRAP_RESAMPLES
    N = len(outcomes)
    taus = np.asarray(taus, dtype=float)
    outcomes = np.asarray(outcomes, dtype=object)

    min_mask = (outcomes == 'local_min')
    div_mask = (outcomes == 'diverge')
    stuck_mask = (outcomes == 'stuck')
    esc_mask = min_mask | div_mask

    n_min = int(min_mask.sum())
    n_div = int(div_mask.sum())
    n_stuck = int(stuck_mask.sum())

    # ── SEE_basic ──────────────────────────────────────────────────────────────
    p_esc_basic = (n_min + n_div) / N
    if p_esc_basic > 0:
        tau_avg_basic = taus[esc_mask].mean()
        see_basic = float(p_esc_basic / tau_avg_basic) if tau_avg_basic > 0 else 0.0
    else:
        tau_avg_basic = float('nan')
        see_basic = 0.0

    # ── SEE_quality ────────────────────────────────────────────────────────────
    p_esc_quality = n_min / N
    if p_esc_quality > 0:
        tau_avg_quality = taus[min_mask].mean()
        see_quality = float(p_esc_quality / tau_avg_quality) if tau_avg_quality > 0 else 0.0
    else:
        tau_avg_quality = float('nan')
        see_quality = 0.0

    # ── SEE_diverge (diagnostic) ───────────────────────────────────────────────
    see_diverge = n_div / N

    # ── Bootstrap CIs ─────────────────────────────────────────────────────────
    see_basic_ci = bootstrap_ci(outcomes, taus, metric='basic', n_resamples=n_resamples)
    see_quality_ci = bootstrap_ci(outcomes, taus, metric='quality', n_resamples=n_resamples)

    # ── τ statistics (over quality-escape trials only) ─────────────────────────
    if n_min > 0:
        tau_quality = taus[min_mask]
        tau_mean = float(tau_quality.mean())
        tau_median = float(np.median(tau_quality))
        tau_std = float(tau_quality.std())
    else:
        tau_mean = tau_median = tau_std = float('nan')

    reliable = (n_min >= config.RELIABLE_ESCAPE_MIN)

    return {
        'SEE_basic': see_basic,
        'SEE_quality': see_quality,
        'SEE_diverge': float(see_diverge),
        'SEE_basic_CI_lo': see_basic_ci[0],
        'SEE_basic_CI_hi': see_basic_ci[1],
        'SEE_quality_CI_lo': see_quality_ci[0],
        'SEE_quality_CI_hi': see_quality_ci[1],
        'escape_min_pct': 100.0 * n_min / N,
        'escape_diverge_pct': 100.0 * n_div / N,
        'stuck_pct': 100.0 * n_stuck / N,
        'tau_mean': tau_mean,
        'tau_median': tau_median,
        'tau_std': tau_std,
        'n_trials': N,
        'reliable': reliable,
    }
