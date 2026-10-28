"""Late-fusion MLPs used by the FMA-Medium HT-Demucs experiments."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn


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


class _StemClassifier(nn.Module):
    """One three-layer MLP voter with the paper-aligned architecture."""

    def __init__(
        self,
        feature_dimension: int,
        number_of_classes: int,
        hidden_dimensions: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dimensions
        self.network = nn.Sequential(
            nn.Linear(feature_dimension, first_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first_hidden, second_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second_hidden, number_of_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


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
            _StemClassifier(
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


class RMSWeightedLateFusionMLP(UnweightedLateFusionMLP):
    """Late fusion with one sigmoid RMS gate per source."""

    def __init__(
        self,
        number_of_sources: int,
        feature_dimension: int,
        number_of_classes: int,
        hidden_dimensions: Sequence[int] = (1024, 512),
        dropout: float = 0.2,
        energy_hidden_dimension: int = 16,
    ) -> None:
        super().__init__(
            number_of_sources=number_of_sources,
            feature_dimension=feature_dimension,
            number_of_classes=number_of_classes,
            hidden_dimensions=hidden_dimensions,
            dropout=dropout,
        )
        if energy_hidden_dimension <= 0:
            raise ValueError("Energy hidden dimension must be positive.")
        self.energy_gates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(1, energy_hidden_dimension),
                nn.ReLU(),
                nn.Linear(energy_hidden_dimension, 1),
                nn.Sigmoid(),
            )
            for _ in range(number_of_sources)
        )

    def forward(self, features: Tensor, energies: Tensor) -> Tensor:
        if features.ndim != 3 or features.shape[1] != self.number_of_sources:
            raise ValueError(
                "Expected features with shape "
                f"[batch, {self.number_of_sources}, dimension]."
            )
        if energies.shape != (features.shape[0], self.number_of_sources, 1):
            raise ValueError(
                "Expected energies with shape "
                f"[batch, {self.number_of_sources}, 1]."
            )
        logits = features.new_zeros((features.shape[0], self.number_of_classes))
        for index, classifier in enumerate(self.stem_classifiers):
            vote = classifier(features[:, index, :])
            logits = logits + vote * self.energy_gates[index](energies[:, index, :])
        return logits
