"""Data loaders for Part 3 NN experiments."""
from __future__ import annotations
import numpy as np
import torch
from sklearn.datasets import make_moons, fetch_california_housing
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_moons(device) -> tuple[torch.Tensor, torch.Tensor]:
    X, y = make_moons(n_samples=2000, noise=0.15, random_state=42)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return X_t, y_t


def load_mnist_binary(device) -> tuple[torch.Tensor, torch.Tensor]:
    """MNIST digits 3 vs 8, 1500 samples per class, flattened."""
    from torchvision import datasets, transforms
    mnist = datasets.MNIST(root='/tmp/mnist_data', download=True,
                           transform=transforms.ToTensor())
    data = [(img.view(-1).numpy(), int(label))
            for img, label in mnist if int(label) in (3, 8)]
    class3 = [(x, 0.0) for x, l in data if l == 3][:1500]
    class8 = [(x, 1.0) for x, l in data if l == 8][:1500]
    all_data = class3 + class8
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(all_data))
    xs = np.stack([all_data[i][0] for i in perm])
    ys = np.array([all_data[i][1] for i in perm])
    X_t = torch.tensor(xs, dtype=torch.float32, device=device)
    y_t = torch.tensor(ys, dtype=torch.float32, device=device)
    return X_t, y_t


def load_housing(device) -> tuple[torch.Tensor, torch.Tensor]:
    data = fetch_california_housing()
    X, y = data.data, data.target
    sx = StandardScaler()
    sy = MinMaxScaler()
    X = sx.fit_transform(X).astype(np.float32)
    y = sy.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    y_t = torch.tensor(y, dtype=torch.float32, device=device)
    return X_t, y_t
