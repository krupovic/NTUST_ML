import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Configurable feed-forward MLP (multilayer perceptron).

    Args:
        input_dim: number of input features
        hidden_sizes: list of hidden layer sizes (can be empty)
        output_dim: number of output units (classes)
        activation: torch activation class (default: nn.ReLU)
        dropout: dropout probability (default: 0.0)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        output_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = h

        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is image-like (N, C, H, W), flatten it
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)
