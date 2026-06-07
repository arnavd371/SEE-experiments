"""Unified optimizer factory for all 7 optimizers used in SEE experiments."""

import torch
import config


def make_optimizer(name: str, params: list, lr: float) -> torch.optim.Optimizer:
    """
    Create a PyTorch optimizer by name with experiment-standard hyperparameters.

    Parameters
    ----------
    name   : one of config.OPTIMIZER_NAMES
    params : list of torch.Tensor / nn.Parameter
    lr     : learning rate

    Returns
    -------
    torch.optim.Optimizer instance
    """
    if name == 'GD_fixed':
        return torch.optim.SGD(params, lr=lr, momentum=0.0, nesterov=False)

    elif name == 'Adam':
        return torch.optim.Adam(
            params, lr=lr,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            eps=config.ADAM_EPS,
        )

    elif name == 'AdamW':
        return torch.optim.AdamW(
            params, lr=lr,
            betas=(config.ADAM_BETA1, config.ADAM_BETA2),
            eps=config.ADAM_EPS,
            weight_decay=config.ADAMW_WD,
        )

    elif name == 'RMSProp':
        return torch.optim.RMSprop(
            params, lr=lr,
            alpha=config.RMSPROP_ALPHA,
            eps=config.RMSPROP_EPS,
        )

    elif name == 'AdaGrad':
        return torch.optim.Adagrad(
            params, lr=lr,
            eps=config.ADAGRAD_EPS,
        )

    elif name == 'SGD_mom':
        return torch.optim.SGD(
            params, lr=lr,
            momentum=config.SGD_MOMENTUM,
            nesterov=False,
        )

    elif name == 'SGD_nesterov':
        return torch.optim.SGD(
            params, lr=lr,
            momentum=config.SGD_MOMENTUM,
            nesterov=True,
        )

    else:
        raise ValueError(f'Unknown optimizer: {name!r}. '
                         f'Available: {config.OPTIMIZER_NAMES}')


def cosine_warmup_scheduler(optimizer: torch.optim.Optimizer,
                             warmup_steps: int,
                             total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup then constant LR (no decay — isolates optimizer behavior)."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
