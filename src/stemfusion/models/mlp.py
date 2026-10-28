"""Multi-layer perceptrons used by the paper experiments."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


class DeepMLPClassifier(nn.Module):
    """Three-linear-layer classifier used for genre classification."""

    def __init__(
        self,
        input_dimension: int,
        number_of_classes: int,
        hidden_dimensions: Sequence[int] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_dimension <= 0:
            raise ValueError("Input dimension must be positive.")
        if number_of_classes <= 1:
            raise ValueError("At least two classes are required.")
        if len(hidden_dimensions) != 2 or any(
            dimension <= 0 for dimension in hidden_dimensions
        ):
            raise ValueError(
                "The paper MLP requires two positive hidden dimensions."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in the interval [0, 1).")

        first_hidden, second_hidden = hidden_dimensions
        self.network = nn.Sequential(
            nn.Linear(input_dimension, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, number_of_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)
