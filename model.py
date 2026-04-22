import torch
from torch import nn


class MLPRegressor(nn.Module):
    """Simple MLP for regression."""

    def __init__(self, input_dim: int = 3, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.1):
        super().__init__()
        h1, h2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),  # Linear output layer for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
