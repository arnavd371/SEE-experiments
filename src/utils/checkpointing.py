"""Save and resume experiment state via pickle checkpoints."""

import os
import pickle
from pathlib import Path


def checkpoint_path(part: int, tag: str = '') -> str:
    base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                        'results', f'part{part}')
    Path(base).mkdir(parents=True, exist_ok=True)
    suffix = f'_{tag}' if tag else ''
    return os.path.join(base, f'checkpoint{suffix}.pkl')


def save_checkpoint(part: int, state: dict, tag: str = '') -> None:
    path = checkpoint_path(part, tag)
    with open(path, 'wb') as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_checkpoint(part: int, tag: str = '') -> dict:
    """Returns checkpoint dict or empty dict if none exists."""
    path = checkpoint_path(part, tag)
    if not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        return pickle.load(f)


def is_done(completed_keys: set, key) -> bool:
    return key in completed_keys
