"""joblib parallel helpers for CPU-bound experiment loops (Parts 1 & 2)."""

from joblib import Parallel, delayed
import config


def run_parallel(fn, tasks: list, n_jobs: int = None, fast: bool = False) -> list:
    """Run fn over tasks in parallel. Returns list of results in same order."""
    n_jobs = n_jobs or config.N_JOBS
    if n_jobs == 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    return Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(fn)(t) for t in tasks
    )
