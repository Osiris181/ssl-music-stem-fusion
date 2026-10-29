"""FMA-Medium single-source and mid-fusion training entry point."""

from __future__ import annotations

import argparse

from stemfusion.configuration import configure_yaml

from stemfusion.training.genre import (
    GenreTrainingConfig,
    GenreTrainingError,
    config_from_args,
    configure_parser as _configure_genre_parser,
    plan_genre_training,
    run_from_args,
    train_genre_classifier,
)

__all__ = [
    "GenreTrainingConfig",
    "GenreTrainingError",
    "config_from_args",
    "configure_parser",
    "plan_genre_training",
    "run_from_args",
    "train_genre_classifier",
]


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Configure the retained FMA-Medium MusicFM MLP runner."""

    _configure_genre_parser(parser)
    parser.description = (
        "Train the FMA-Medium MusicFM mid-fusion MLP. Repeat "
        "--source to concatenate selected mixture and stem features."
    )
    configure_yaml(parser, "mid_fusion", "training/fma_medium_mid_fusion.yaml")
