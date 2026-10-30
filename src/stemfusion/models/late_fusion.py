"""Equal-logit-sum model container for independently selected MLP branches.

The revised training workflow fits branches separately and can assemble
ensembles directly from saved predictions. This container is also available
when inference from loaded branch weights is needed.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from stemfusion.models.mlp import DeepMLPClassifier


def _validate_mlp_settings(
    feature_dimension: int,
    number_of_classes: int,
    hidden_dimensions: Sequence[int],
    dropout: float,
) -> None:
    if feature_dimension <= 0:
        raise ValueError("Feature dimension must be positive.")
    if number_of_classes <= 1:
        raise ValueError("At least two classes are required.")
    if len(hidden_dimensions) != 2 or any(
        dimension <= 0 for dimension in hidden_dimensions
    ):
        raise ValueError("Two positive hidden dimensions are required.")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("Dropout must be in the interval [0, 1).")


class UnweightedLateFusionMLP(nn.Module):
    """Independent source classifiers whose logits are summed equally."""

    def __init__(
        self,
        number_of_sources: int,
        feature_dimension: int,
        number_of_classes: int,
        hidden_dimensions: Sequence[int] = (1024, 512),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if number_of_sources <= 0:
            raise ValueError("At least one source is required.")
        _validate_mlp_settings(
            feature_dimension,
            number_of_classes,
            hidden_dimensions,
            dropout,
        )
        self.number_of_sources = number_of_sources
        self.number_of_classes = number_of_classes
        self.stem_classifiers = nn.ModuleList(
            DeepMLPClassifier(
                feature_dimension,
                number_of_classes,
                hidden_dimensions,
                dropout,
            )
            for _ in range(number_of_sources)
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.number_of_sources:
            raise ValueError(
                "Expected features with shape "
                f"[batch, {self.number_of_sources}, dimension]."
            )
        logits = features.new_zeros((features.shape[0], self.number_of_classes))
        for index, classifier in enumerate(self.stem_classifiers):
            logits = logits + classifier(features[:, index, :])
        return logits
