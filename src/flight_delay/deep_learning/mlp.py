"""Feed-forward flight delay model for encoded schedule and history features."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


EMBEDDING_FIELDS = ("carrier", "origin", "dest")


class FlightDelayMLP(nn.Module):
    """Combine three categorical embeddings with dense inputs to predict delay."""

    def __init__(
        self,
        dense_dim: int,
        *,
        carrier_embedding: nn.Embedding,
        origin_embedding: nn.Embedding,
        dest_embedding: nn.Embedding,
        hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if dense_dim < 1:
            raise ValueError("dense_dim must be positive")
        if not hidden_dims or any(width < 1 for width in hidden_dims):
            raise ValueError("hidden_dims must contain positive layer widths")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.dense_dim = dense_dim
        self.embeddings = nn.ModuleDict(
            {
                "carrier": carrier_embedding,
                "origin": origin_embedding,
                "dest": dest_embedding,
            }
        )
        input_dim = dense_dim + sum(
            embedding.embedding_dim for embedding in self.embeddings.values()
        )
        layers: list[nn.Module] = []
        for width in hidden_dims:
            layers.extend((nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x_dense: Tensor, x_cat: Tensor) -> Tensor:
        """Return logits of shape (batch,) for dense values and long category IDs."""
        if x_dense.ndim != 2 or x_dense.shape[1] != self.dense_dim:
            raise ValueError(f"x_dense must have shape (batch, {self.dense_dim})")
        if x_cat.ndim != 2 or x_cat.shape[1] != len(EMBEDDING_FIELDS):
            raise ValueError(
                "x_cat must have shape (batch, 3) in carrier, origin, dest order"
            )
        if x_dense.shape[0] != x_cat.shape[0]:
            raise ValueError("x_dense and x_cat must have the same batch size")
        if not x_dense.is_floating_point():
            raise TypeError("x_dense must contain floating-point values")
        if x_cat.dtype != torch.long:
            raise TypeError("x_cat must contain torch.long category IDs")

        embedded = [
            self.embeddings[field](x_cat[:, column])
            for column, field in enumerate(EMBEDDING_FIELDS)
        ]
        return self.network(torch.cat([x_dense, *embedded], dim=1)).squeeze(-1)
