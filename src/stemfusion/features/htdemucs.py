"""Public MusicFM and RMS feature-extraction entry point for HT-Demucs."""

from __future__ import annotations

import argparse

from stemfusion.data.fma import HTDEMUCS_SOURCES

HTDEMUCS_FEATURE_SOURCES = frozenset({"mixture", *HTDEMUCS_SOURCES})
from stemfusion.features._htdemucs_rms import (
    RMSExtractionConfig,
    RMSExtractionError,
    RMSExtractionSummary,
    config_from_args as rms_config_from_args,
    configure_parser as _configure_rms_parser,
    extract_rms,
    run_from_args as run_rms_from_args,
)
from stemfusion.features._musicfm import (
    MusicFMExtractionConfig,
    MusicFMExtractionError,
    MusicFMExtractionSummary,
    config_from_args as musicfm_config_from_args,
    configure_parser as _configure_musicfm_parser,
    extract_musicfm,
    run_from_args as run_musicfm_from_args,
)

__all__ = [
    "MusicFMExtractionConfig",
    "MusicFMExtractionError",
    "MusicFMExtractionSummary",
    "RMSExtractionConfig",
    "RMSExtractionError",
    "RMSExtractionSummary",
    "extract_musicfm",
    "extract_rms",
    "musicfm_config_from_args",
    "rms_config_from_args",
    "configure_parser",
    "run_musicfm_from_args",
    "run_rms_from_args",
]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Configure HT-Demucs MusicFM and RMS extraction commands."""

    parser.description = (
        "Extract HT-Demucs MusicFM embeddings or separate RMS energy values."
    )
    feature_parsers = parser.add_subparsers(
        dest="htdemucs_feature",
        required=True,
    )
    musicfm_parser = feature_parsers.add_parser(
        "musicfm",
        help="Extract MusicFM embeddings from HT-Demucs mixtures or stems.",
    )
    _configure_musicfm_parser(
        musicfm_parser,
        allowed_sources=HTDEMUCS_FEATURE_SOURCES,
    )
    rms_parser = feature_parsers.add_parser(
        "rms",
        help="Extract separate RMS energy values from HT-Demucs sources.",
    )
    _configure_rms_parser(rms_parser)
