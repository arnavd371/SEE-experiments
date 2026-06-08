"""
SEE metric computation.

SEE = P_esc / tau_avg

SEE_basic:
  P_esc = (local_min + diverge) / N
  tau_avg = mean escape steps over all escaping trials

SEE_quality:
  P_esc = local_min_count / N
  tau_avg = mean escape steps over quality-escape (local_min) trials only

SEE_diverge = diverge_count / N  (diagnostic)
"""
import numpy as np

from .bootstrap import bootstrap_ci


def compute_see(
    escaped_min: np.ndarray,   # bool (N,)
    escaped_div: np.ndarray,   # bool (N,)
    escape_time: np.ndarray,   # float (N,), T_max for stuck
    T_max: int,
    n_resamples: int = 1000,
) -> dict:
    N = len(escaped_min)

    n_min  = int(escaped_min.sum())
    n_div  = int(escaped_div.sum())
    n_esc  = n_min + n_div                       # basic escapers
    n_stuck = N - n_esc

    # --- SEE_basic ---
    p_esc_basic = n_esc / N
    basic_times = escape_time[escaped_min | escaped_div]
    tau_basic = float(np.mean(basic_times)) if len(basic_times) > 0 else float(T_max)
    SEE_basic = p_esc_basic / tau_basic if tau_basic > 0 else 0.0

    # Bootstrap CI for SEE_basic via per-trial indicator / time
    # We resample over trials and recompute P_esc/tau for each resample.
    see_basic_arr = _per_trial_see_basic(escaped_min, escaped_div, escape_time, T_max)
    ci_basic = bootstrap_ci(see_basic_arr, statistic=np.mean, n_resamples=n_resamples)

    # --- SEE_quality ---
    p_esc_qual = n_min / N
    qual_times = escape_time[escaped_min]
    tau_qual = float(np.mean(qual_times)) if len(qual_times) > 0 else float(T_max)
    SEE_quality = p_esc_qual / tau_qual if tau_qual > 0 else 0.0

    see_qual_arr = _per_trial_see_quality(escaped_min, escape_time, T_max)
    ci_qual = bootstrap_ci(see_qual_arr, statistic=np.mean, n_resamples=n_resamples)

    # --- SEE_diverge (diagnostic) ---
    SEE_diverge = n_div / N

    # --- tau stats (over all escapers) ---
    if len(basic_times) > 0:
        tau_median = float(np.median(basic_times))
        tau_std    = float(np.std(basic_times))
    else:
        tau_median = float(T_max)
        tau_std    = 0.0

    return {
        'SEE_basic':        SEE_basic,
        'SEE_quality':      SEE_quality,
        'SEE_diverge':      SEE_diverge,
        'SEE_basic_CI_lo':  ci_basic[0],
        'SEE_basic_CI_hi':  ci_basic[1],
        'SEE_quality_CI_lo': ci_qual[0],
        'SEE_quality_CI_hi': ci_qual[1],
        'escape_min_pct':   100.0 * n_min  / N,
        'escape_diverge_pct': 100.0 * n_div / N,
        'stuck_pct':        100.0 * n_stuck / N,
        'tau_mean':         tau_basic,
        'tau_median':       tau_median,
        'tau_std':          tau_std,
        'n_min':            n_min,
        'n_div':            n_div,
        'n_stuck':          n_stuck,
    }


# --- helpers for per-trial bootstrap arrays ---

def _per_trial_see_basic(escaped_min, escaped_div, escape_time, T_max):
    """
    One scalar per trial suitable for bootstrap resampling.
    We use the approach: for each bootstrap resample, compute P_esc and tau.
    To make this vectorizable, we return a binary escape indicator array;
    bootstrap_ci will average them (giving P_esc), but that ignores tau.

    Instead we return a score: (escaped) / max(escape_time, 1)
    so that mean over N approximates SEE_basic proportionally.
    This is a smooth proxy used only for CI width, not the point estimate.
    """
    N = len(escaped_min)
    escaped = escaped_min | escaped_div
    times = np.where(escaped, escape_time, float(T_max))
    # per-trial contribution: (1/N) * P_esc / tau_avg  ≈ mean(escaped_i / time_i)
    # This decomposes additively and is the natural bootstrap target.
    scores = np.where(escaped, 1.0 / np.maximum(times, 1.0), 0.0)
    return scores


def _per_trial_see_quality(escaped_min, escape_time, T_max):
    N = len(escaped_min)
    times = np.where(escaped_min, escape_time, float(T_max))
    scores = np.where(escaped_min, 1.0 / np.maximum(times, 1.0), 0.0)
    return scores
