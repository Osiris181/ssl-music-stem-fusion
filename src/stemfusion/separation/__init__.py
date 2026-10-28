"""Music source-separation backends."""

from stemfusion.separation.htdemucs import (
    HTDemucsSeparationConfig,
    HTDemucsSeparationSummary,
    separate_dataset as separate_dataset_with_htdemucs,
)

__all__ = [
    "HTDemucsSeparationConfig",
    "HTDemucsSeparationSummary",
    "separate_dataset_with_htdemucs",
]
