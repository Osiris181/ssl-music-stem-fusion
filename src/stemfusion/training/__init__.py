"""Training workflows for downstream music-classification tasks."""

from stemfusion.training.genre import (
    GenreTrainingConfig,
    GenreTrainingError,
    SourceFeatureDirectory,
    train_genre_classifier,
)

from stemfusion.training.late_fusion import (
    LateFusionTrainingConfig,
    train_late_fusion_classifier,
)

__all__ = [
    "GenreTrainingConfig",
    "GenreTrainingError",
    "SourceFeatureDirectory",
    "train_genre_classifier",
    "LateFusionTrainingConfig",
    "train_late_fusion_classifier",
]
