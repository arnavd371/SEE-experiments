"""Six 2D benchmark functions implemented with PyTorch autograd."""

import math
import torch


# Each function accepts a 1-D torch.Tensor x of shape (2,) and returns a scalar tensor.

def himmelblau(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    return (x1 ** 2 + x2 - 11) ** 2 + (x1 + x2 ** 2 - 7) ** 2


def rosenbrock(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    return (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2


def ackley(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    a = -20.0 * torch.exp(-0.2 * torch.sqrt(0.5 * (x1 ** 2 + x2 ** 2)))
    b = -torch.exp(0.5 * (torch.cos(2.0 * math.pi * x1) + torch.cos(2.0 * math.pi * x2)))
    return a + b + 20.0 + math.e


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    return (20.0
            + x1 ** 2 - 10.0 * torch.cos(2.0 * math.pi * x1)
            + x2 ** 2 - 10.0 * torch.cos(2.0 * math.pi * x2))


def styblinski_tang(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    return 0.5 * ((x1 ** 4 - 16.0 * x1 ** 2 + 5.0 * x1)
                  + (x2 ** 4 - 16.0 * x2 ** 2 + 5.0 * x2))


def levy(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[0], x[1]
    w1 = 1.0 + (x1 - 1.0) / 4.0
    w2 = 1.0 + (x2 - 1.0) / 4.0
    term1 = torch.sin(math.pi * w1) ** 2
    term2 = (w1 - 1.0) ** 2 * (1.0 + 10.0 * torch.sin(math.pi * w1 + 1.0) ** 2)
    term3 = (w2 - 1.0) ** 2 * (1.0 + torch.sin(2.0 * math.pi * w2) ** 2)
    return term1 + term2 + term3


_REGISTRY = {
    'Himmelblau': himmelblau,
    'Rosenbrock': rosenbrock,
    'Ackley': ackley,
    'Rastrigin': rastrigin,
    'Styblinski-Tang': styblinski_tang,
    'Levy': levy,
}


def get_function_2d(name: str):
    if name not in _REGISTRY:
        raise KeyError(f'Unknown 2D function: {name!r}. '
                       f'Available: {list(_REGISTRY)}')
    return _REGISTRY[name]


def list_functions_2d():
    return list(_REGISTRY.keys())
