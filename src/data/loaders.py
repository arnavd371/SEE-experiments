"""
Data loaders for all real datasets used in Parts 3 and 4.

Tasks:
  A — Moons classification  (sklearn)
  B — MNIST binary 3 vs 8   (torchvision)
  C — California Housing     (sklearn)

Part 4:
  Wikitext-2 tokenized with GPT-2 BPE
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import config


# ── Task A: Moons ─────────────────────────────────────────────────────────────

def load_moons(device: torch.device = None):
    from sklearn.datasets import make_moons
    device = device or config.DEVICE
    X, y = make_moons(
        n_samples=config.MOONS_SAMPLES,
        noise=config.MOONS_NOISE,
        random_state=config.MOONS_RANDOM_STATE,
    )
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True,
                        drop_last=False)
    return {
        'X': X_t, 'y': y_t,
        'loader': loader,
        'full_data': (X_t, y_t),
        'input_dim': 2,
        'output_dim': 1,
        'task_type': 'binary',
        'conv_loss': config.MOONS_CONV_LOSS,
        'name': 'Moons',
    }


# ── Task B: MNIST binary (digit 3 vs digit 8) ─────────────────────────────────

def load_mnist_binary(device: torch.device = None):
    device = device or config.DEVICE
    try:
        from torchvision.datasets import MNIST
        from torchvision import transforms
        import os
        data_root = os.path.join(config.BASE_DIR, 'data', 'mnist_cache')
        transform = transforms.ToTensor()
        train_ds = MNIST(root=data_root, train=True, download=True, transform=transform)

        d3_idx = (train_ds.targets == config.MNIST_DIGITS[0]).nonzero(as_tuple=True)[0]
        d8_idx = (train_ds.targets == config.MNIST_DIGITS[1]).nonzero(as_tuple=True)[0]

        n_each = config.MNIST_PER_CLASS
        d3_idx = d3_idx[:n_each]
        d8_idx = d8_idx[:n_each]

        X3 = train_ds.data[d3_idx].float() / 255.0
        X8 = train_ds.data[d8_idx].float() / 255.0
        X = torch.cat([X3, X8], dim=0).view(-1, 784)
        y = torch.cat([torch.zeros(n_each), torch.ones(n_each)], dim=0)

    except ImportError:
        # Fallback: random data with correct shape if torchvision unavailable
        import warnings
        warnings.warn('torchvision not found; using random MNIST-shaped data')
        n_total = config.MNIST_PER_CLASS * 2
        X = torch.rand(n_total, 784)
        y = torch.cat([torch.zeros(config.MNIST_PER_CLASS),
                       torch.ones(config.MNIST_PER_CLASS)])

    # Shuffle
    perm = torch.randperm(len(X))
    X, y = X[perm].to(device), y[perm].to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True,
                        drop_last=False)
    return {
        'X': X, 'y': y,
        'loader': loader,
        'full_data': (X, y),
        'input_dim': 784,
        'output_dim': 1,
        'task_type': 'binary',
        'conv_loss': config.MNIST_CONV_LOSS,
        'name': 'MNIST-3v8',
    }


# ── Task C: California Housing ─────────────────────────────────────────────────

def load_housing(device: torch.device = None):
    from sklearn.datasets import fetch_california_housing
    from sklearn.preprocessing import StandardScaler, MinMaxScaler

    device = device or config.DEVICE
    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=config.NN_BATCH_SIZE, shuffle=True,
                        drop_last=False)
    return {
        'X': X_t, 'y': y_t,
        'loader': loader,
        'full_data': (X_t, y_t),
        'input_dim': 8,
        'output_dim': 1,
        'task_type': 'regression',
        'conv_loss': config.HOUSING_CONV_LOSS,
        'name': 'CalHousing',
    }


def load_all_nn_tasks(device: torch.device = None):
    return {
        'Moons': load_moons(device),
        'MNIST-3v8': load_mnist_binary(device),
        'CalHousing': load_housing(device),
    }


# ── Part 4: Wikitext-2 ────────────────────────────────────────────────────────

def load_wikitext2(context_length: int = None, device: torch.device = None):
    """
    Load Wikitext-2, tokenize with GPT-2 BPE, pack into non-overlapping chunks.

    Returns dict with 'train_ids', 'val_ids', train/val DataLoaders.
    """
    from datasets import load_dataset
    from transformers import GPT2TokenizerFast

    context_length = context_length or config.GPT_CONTEXT_LENGTH
    device = device or config.DEVICE

    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    raw = load_dataset(config.WIKITEXT_DATASET, config.WIKITEXT_CONFIG)

    def tokenize_split(split_name, max_tokens):
        all_ids = []
        for item in raw[split_name]:
            text = item['text'].strip()
            if text:
                ids = tokenizer.encode(text)
                all_ids.extend(ids)
                if len(all_ids) >= max_tokens:
                    break
        return all_ids[:max_tokens]

    train_ids = tokenize_split('train', config.LLM_TRAIN_TOKENS)
    val_ids = tokenize_split('validation', config.LLM_VAL_TOKENS)

    def make_chunks(ids, ctx_len):
        ids_t = torch.tensor(ids, dtype=torch.long)
        n_chunks = len(ids_t) // (ctx_len + 1)
        chunks = ids_t[:n_chunks * (ctx_len + 1)].view(n_chunks, ctx_len + 1)
        x = chunks[:, :ctx_len]
        y = chunks[:, 1:ctx_len + 1]
        return x, y

    train_x, train_y = make_chunks(train_ids, context_length)
    val_x, val_y = make_chunks(val_ids, context_length)

    train_ds = TensorDataset(train_x, train_y)
    val_ds = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=config.LLM_BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.LLM_BATCH_SIZE,
                            shuffle=False, drop_last=False)

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_x': train_x,
        'train_y': train_y,
        'val_x': val_x,
        'val_y': val_y,
        'vocab_size': config.GPT_VOCAB_SIZE,
        'context_length': context_length,
    }
