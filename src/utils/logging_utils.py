"""CSV logging helpers with atomic append and environment logging."""

import os
import sys
import platform
import csv
import json
import datetime
from pathlib import Path


def log_environment(out_path: str) -> None:
    """Write package versions, device info, and timestamp to env_log.txt."""
    import torch
    import numpy as np
    import scipy
    import sklearn

    lines = [
        f"Timestamp: {datetime.datetime.now().isoformat()}",
        f"Python: {sys.version}",
        f"PyTorch: {torch.__version__}",
        f"NumPy: {np.__version__}",
        f"SciPy: {scipy.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"Platform: {platform.platform()}",
        f"MPS available: {torch.backends.mps.is_available()}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]
    try:
        import transformers
        lines.append(f"Transformers: {transformers.__version__}")
    except ImportError:
        lines.append("Transformers: not installed")
    try:
        import pingouin
        lines.append(f"Pingouin: {pingouin.__version__}")
    except ImportError:
        lines.append("Pingouin: not installed")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def ensure_csv(path: str, fieldnames: list) -> None:
    """Create CSV with header if it doesn't exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def append_rows(path: str, rows: list, fieldnames: list) -> None:
    """Append rows to CSV. Creates file with header if needed."""
    ensure_csv(path, fieldnames)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writerows(rows)


def save_json(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


PART1_FIELDS = [
    'function', 'optimizer', 'lr', 'saddle_id', 'saddle_x', 'saddle_y',
    'SEE_basic', 'SEE_quality', 'SEE_diverge',
    'SEE_basic_CI_lo', 'SEE_basic_CI_hi',
    'SEE_quality_CI_lo', 'SEE_quality_CI_hi',
    'escape_min_pct', 'escape_diverge_pct', 'stuck_pct',
    'tau_mean', 'tau_median', 'tau_std', 'n_trials',
    'best_loss_at_escape_mean',
    'wilcoxon_p_vs_best', 'cohens_d_vs_best',
    'reliable',
]

PART2_FIELDS = [
    'function', 'optimizer', 'best_lr', 'dimension', 'saddle_index_k',
    'SEE_basic', 'SEE_quality', 'SEE_diverge',
    'SEE_basic_CI_lo', 'SEE_basic_CI_hi',
    'tau_mean', 'tau_std', 'escape_min_pct', 'stuck_pct',
    'power_law_alpha',
]

PART3_FIELDS = [
    'task', 'optimizer', 'run_seed', 'step',
    'loss', 'grad_norm', 'is_saddle_step',
    'SEE_NN', 'SEE_NN_CI_lo', 'SEE_NN_CI_hi',
    'min_eigenvalue', 'max_eigenvalue',
    'steps_to_convergence', 'final_loss', 'final_accuracy',
]

PART4_FIELDS = [
    'optimizer', 'run_seed', 'step',
    'train_loss', 'val_ppl', 'grad_norm',
    'is_plateau_step', 'plateau_event_id',
    'steps_to_ppl_threshold', 'final_val_ppl',
    'total_plateau_steps', 'plateau_fraction',
]

PART5_FIELDS = [
    'optimizer',
    'mean_SEE_quality', 'SEE_quality_rank',
    'plateau_fraction', 'plateau_fraction_rank',
    'steps_to_ppl_threshold', 'ppl_rank',
    'spearman_r_SEE_ppl', 'spearman_p_SEE_ppl',
    'pearson_r_SEE_ppl', 'pearson_p_SEE_ppl',
    'early_plateau_fraction',
    'early_vs_final_ppl_r',
]
