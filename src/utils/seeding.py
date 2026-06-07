"""Global seed utilities for full reproducibility across PyTorch, NumPy, and Python random."""

import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def trial_seed(base_seed: int, trial_idx: int, n_trials: int = 100_000) -> int:
    """Unique, reproducible seed for each trial index."""
    return base_seed * n_trials + trial_idx
