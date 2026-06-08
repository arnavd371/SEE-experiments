"""
2-D benchmark functions.  All accept (N, 2) and return (N,).
Domain tuples: (x_lo, x_hi, y_lo, y_hi).
"""
import math
import torch


FUNCTIONS = {}        # name -> callable
DOMAINS   = {}        # name -> (x_lo, x_hi, y_lo, y_hi)


def _register(name, domain):
    def decorator(fn):
        FUNCTIONS[name] = fn
        DOMAINS[name]   = domain
        return fn
    return decorator


@_register("Himmelblau", (-6.0, 6.0, -6.0, 6.0))
def himmelblau(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., 0], x[..., 1]
    return (a**2 + b - 11)**2 + (a + b**2 - 7)**2


@_register("SixHumpCamel", (-3.0, 3.0, -2.0, 2.0))
def six_hump_camel(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., 0], x[..., 1]
    return (4 - 2.1*a**2 + a**4/3)*a**2 + a*b + (-4 + 4*b**2)*b**2


@_register("Rastrigin", (-5.12, 5.12, -5.12, 5.12))
def rastrigin(x: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    a, b = x[..., 0], x[..., 1]
    return 20 + a**2 - 10*torch.cos(2*pi*a) + b**2 - 10*torch.cos(2*pi*b)


@_register("Styblinski_Tang", (-5.0, 5.0, -5.0, 5.0))
def styblinski_tang(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((x**4 - 16*x**2 + 5*x).sum(dim=-1))


@_register("Levy", (-10.0, 10.0, -10.0, 10.0))
def levy(x: torch.Tensor) -> torch.Tensor:
    pi = math.pi
    w = 1 + (x - 1) / 4                       # (N, 2)
    w1, w2 = w[..., 0], w[..., 1]
    term1 = torch.sin(pi * w1)**2
    term2 = (w1 - 1)**2 * (1 + 10*torch.sin(pi*w1 + 1)**2)
    term3 = (w2 - 1)**2 * (1 + torch.sin(2*pi*w2)**2)
    return term1 + term2 + term3


@_register("Beale", (-4.5, 4.5, -4.5, 4.5))
def beale(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., 0], x[..., 1]
    t1 = (1.5   - a + a*b   )**2
    t2 = (2.25  - a + a*b**2)**2
    t3 = (2.625 - a + a*b**3)**2
    return t1 + t2 + t3


def domain_diameter(name: str) -> float:
    xl, xh, yl, yh = DOMAINS[name]
    import math
    return math.sqrt((xh - xl)**2 + (yh - yl)**2)
