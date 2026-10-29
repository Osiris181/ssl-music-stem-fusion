"""Frozen feature-extraction backends."""

from stemfusion.features._musicfm import (
    MusicFMExtractionConfig,
    MusicFMExtractionSummary,
    extract_musicfm,
    extract_musicfm_sources,
)

__all__ = [
    "MusicFMExtractionConfig",
    "MusicFMExtractionSummary",
    "extract_musicfm",
    "extract_musicfm_sources",
]
