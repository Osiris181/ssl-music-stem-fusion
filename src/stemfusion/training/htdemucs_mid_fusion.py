"""Public Table 3 mid-fusion training workflow for HT-Demucs features."""

from __future__ import annotations

import argparse

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
    """Configure the HT-Demucs Table 3 MusicFM mid-fusion command."""

    _configure_genre_parser(parser)
    parser.description = (
        "Train the Table 3 HT-Demucs MusicFM mid-fusion MLP. Repeat "
        "--source to concatenate selected mixture and stem features."
    )
    parser.set_defaults(drop_silent_features=True)
