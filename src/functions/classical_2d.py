"""
2D benchmark functions implemented with PyTorch autograd.
All functions accept a 1D tensor of shape (2,) or a 2D tensor of shape (N, 2).
For the vectorized trial loop they must also work on a single (2,) input
so autograd.functional.hessian can be called on them.
"""
import torch
import math


def himmelblau(x: torch.Tensor) -> torch.Tensor:
    """(x²+y-11)² + (x+y²-7)²   domain [-6,6]²"""
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def six_hump_camel(x: torch.Tensor) -> torch.Tensor:
    """(4-2.1x²+x⁴/3)x² + xy + (-4+4y²)y²   domain [-3,3]×[-2,2]"""
    return ((4 - 2.1 * x[0] ** 2 + x[0] ** 4 / 3) * x[0] ** 2
            + x[0] * x[1]
            + (-4 + 4 * x[1] ** 2) * x[1] ** 2)


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """20 + x²-10cos(2πx) + y²-10cos(2πy)   domain [-5.12,5.12]²"""
    return (20
            + x[0] ** 2 - 10 * torch.cos(2 * math.pi * x[0])
            + x[1] ** 2 - 10 * torch.cos(2 * math.pi * x[1]))


def styblinski_tang(x: torch.Tensor) -> torch.Tensor:
    """0.5[(x⁴-16x²+5x)+(y⁴-16y²+5y)]   domain [-5,5]²"""
    return 0.5 * ((x[0] ** 4 - 16 * x[0] ** 2 + 5 * x[0])
                  + (x[1] ** 4 - 16 * x[1] ** 2 + 5 * x[1]))


def levy(x: torch.Tensor) -> torch.Tensor:
    """Levy function   domain [-10,10]²"""
    w1 = 1 + (x[0] - 1) / 4
    w2 = 1 + (x[1] - 1) / 4
    return (torch.sin(math.pi * w1) ** 2
            + (w1 - 1) ** 2 * (1 + 10 * torch.sin(math.pi * w1 + 1) ** 2)
            + (w2 - 1) ** 2 * (1 + torch.sin(2 * math.pi * w2) ** 2))


def beale(x: torch.Tensor) -> torch.Tensor:
    """Beale function   domain [-4.5,4.5]²"""
    return ((1.5   - x[0] + x[0] * x[1])     ** 2
            + (2.25  - x[0] + x[0] * x[1] ** 2) ** 2
            + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2)


# Registry used by the rest of the code
FUNCTIONS_2D = {
    'Himmelblau':      (himmelblau,      (-6,  6,  -6,  6)),
    'SixHumpCamel':    (six_hump_camel,  (-3,  3,  -2,  2)),
    'Rastrigin':       (rastrigin,       (-5.12, 5.12, -5.12, 5.12)),
    'Styblinski-Tang': (styblinski_tang, (-5,  5,  -5,  5)),
    'Levy':            (levy,            (-10, 10, -10, 10)),
    'Beale':           (beale,           (-4.5, 4.5, -4.5, 4.5)),
}
