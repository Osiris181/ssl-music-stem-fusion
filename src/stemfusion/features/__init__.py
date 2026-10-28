"""Frozen feature-extraction backends."""

from stemfusion.features.htdemucs import (
    MusicFMExtractionConfig,
    MusicFMExtractionSummary,
    extract_musicfm,
    RMSExtractionConfig,
    RMSExtractionSummary,
    extract_rms,
)

__all__ = [
    "MusicFMExtractionConfig",
    "MusicFMExtractionSummary",
    "RMSExtractionConfig",
    "RMSExtractionSummary",
    "extract_musicfm",
    "extract_rms",
]
