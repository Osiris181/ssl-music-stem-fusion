"""Training workflows for downstream music-classification tasks."""

from stemfusion.training.genre import (
    GenreTrainingConfig,
    GenreTrainingError,
    SourceFeatureDirectory,
    train_genre_classifier,
)

from stemfusion.training._htdemucs_late_fusion import LateFusionTrainingConfig
from stemfusion.training.htdemucs_late_fusion import train_late_fusion_classifier

__all__ = [
    "GenreTrainingConfig",
    "GenreTrainingError",
    "SourceFeatureDirectory",
    "train_genre_classifier",
    "LateFusionTrainingConfig",
    "train_late_fusion_classifier",
]
