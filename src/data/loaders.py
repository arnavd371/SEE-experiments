"""Data loaders for Part 3 neural-network experiments."""
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _to_tensors(*arrs):
    return [torch.tensor(a, dtype=torch.float32) for a in arrs]


def moons_loaders(seed=42, batch=64):
    X, y = make_moons(n_samples=2000, noise=0.15, random_state=seed)
    X = X.astype(np.float32)
    y = y.astype(np.float32)[:, None]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    ds_tr = TensorDataset(*_to_tensors(Xtr, ytr))
    ds_te = TensorDataset(*_to_tensors(Xte, yte))
    return (DataLoader(ds_tr, batch_size=batch, shuffle=True),
            DataLoader(ds_te, batch_size=256),
            2, 1, "bce")


def mnist_38_loaders(seed=42, batch=64):
    try:
        from torchvision import datasets, transforms
        import torchvision
    except ImportError:
        return _fallback_mnist38(seed, batch)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: x.view(-1))])
    try:
        train_full = datasets.MNIST("~/.pytorch_datasets", train=True,
                                    download=True, transform=transform)
        test_full  = datasets.MNIST("~/.pytorch_datasets", train=False,
                                    download=True, transform=transform)
    except Exception:
        return _fallback_mnist38(seed, batch)

    def filter38(ds, n_per_class=1500):
        idx3 = [i for i, (_, y) in enumerate(ds) if y == 3][:n_per_class]
        idx8 = [i for i, (_, y) in enumerate(ds) if y == 8][:n_per_class]
        idxs = idx3 + idx8
        X = torch.stack([ds[i][0] for i in idxs])          # (N, 784)
        y = torch.tensor([0.0 if ds[i][1] == 3 else 1.0 for i in idxs])[:, None]
        return TensorDataset(X, y)

    ds_tr = filter38(train_full)
    ds_te = filter38(test_full, n_per_class=300)
    return (DataLoader(ds_tr, batch_size=batch, shuffle=True),
            DataLoader(ds_te, batch_size=256),
            784, 1, "bce")


def _fallback_mnist38(seed=42, batch=64):
    """Synthetic fallback if MNIST unavailable."""
    rng = np.random.RandomState(seed)
    X = rng.randn(3000, 784).astype(np.float32)
    y = (rng.rand(3000) > 0.5).astype(np.float32)[:, None]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    ds_tr = TensorDataset(*_to_tensors(Xtr, ytr))
    ds_te = TensorDataset(*_to_tensors(Xte, yte))
    return DataLoader(ds_tr, batch_size=batch, shuffle=True), DataLoader(ds_te, batch_size=256), 784, 1, "bce"


def california_loaders(seed=42, batch=64):
    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)[:, None]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

    sx = StandardScaler(); Xtr = sx.fit_transform(Xtr); Xte = sx.transform(Xte)
    sy = MinMaxScaler();  ytr = sy.fit_transform(ytr); yte = sy.transform(yte)

    ds_tr = TensorDataset(*_to_tensors(Xtr, ytr))
    ds_te = TensorDataset(*_to_tensors(Xte, yte))
    return (DataLoader(ds_tr, batch_size=batch, shuffle=True),
            DataLoader(ds_te, batch_size=256),
            8, 1, "mse")


TASK_LOADERS = {
    "Moons":      moons_loaders,
    "MNIST_3v8":  mnist_38_loaders,
    "California": california_loaders,
}

CONVERGENCE_TARGETS = {
    "Moons":      0.05,
    "MNIST_3v8":  0.10,
    "California": 0.02,
}
