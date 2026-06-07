"""Two-layer MLP for Parts 3 (real dataset classification/regression)."""

import torch
import torch.nn as nn
import config


class MLP(nn.Module):
    """
    Two-layer MLP with tanh activations.

    Architecture:
      Input → Linear(input_dim, 64) → tanh
           → Linear(64, 32) → tanh
           → Linear(32, output_dim)

    ~2000-3000 parameters depending on input_dim.
    """

    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.MLP_HIDDEN1),
            nn.Tanh(),
            nn.Linear(config.MLP_HIDDEN1, config.MLP_HIDDEN2),
            nn.Tanh(),
            nn.Linear(config.MLP_HIDDEN2, output_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1)  # (batch,) for output_dim=1

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_loss_fn(task_type: str):
    if task_type == 'binary':
        return nn.BCEWithLogitsLoss()
    elif task_type == 'regression':
        return nn.MSELoss()
    else:
        raise ValueError(f'Unknown task_type: {task_type!r}')


def clone_to_device(model: MLP, device: torch.device) -> MLP:
    new_model = MLP(
        input_dim=model.net[0].in_features,
        output_dim=model.net[-1].out_features,
    ).to(device)
    new_model.load_state_dict(model.state_dict())
    return new_model
