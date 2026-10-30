"""Aligned cached features passed from dataset adapters to shared trainers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class FeatureDataError(ValueError):
    """Cached features or dataset partitions violate the data contract."""


@dataclass(frozen=True)
class FeatureSplit:
    track_ids: tuple[str, ...]
    labels: Any
    features: dict[str, Any]

    def concatenate(self, sources: tuple[str, ...]) -> Any:
        import numpy as np

        if not sources or len(set(sources)) != len(sources):
            raise FeatureDataError("Select distinct feature sources.")
        if len(sources) == 1:
            return self.features[sources[0]]
        return np.concatenate([self.features[source] for source in sources], axis=1)


@dataclass(frozen=True)
class FeatureDataset:
    splits: dict[str, FeatureSplit]
    label_mapping: dict[str, int]
    dimension: int
    fingerprint: str
    audit: dict[str, Any]

    @property
    def number_of_classes(self) -> int:
        return len(self.label_mapping)
