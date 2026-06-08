import json
import os
import pickle
from pathlib import Path


def save_checkpoint(data, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def checkpoint_exists(path: str) -> bool:
    return os.path.exists(path)
