"""Command-line interface for the stem-fusion pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from stemfusion.data.fma import configure_parser as configure_fma_parser
from stemfusion.features.htdemucs import (
    configure_parser as configure_htdemucs_features_parser,
)
from stemfusion.separation.htdemucs import (
    configure_parser as configure_htdemucs_parser,
)
from stemfusion.training.genre import (
    configure_parser as configure_genre_training_parser,
)
from stemfusion.training.htdemucs_late_fusion import (
    configure_parser as configure_htdemucs_late_fusion_parser,
)
from stemfusion.training.htdemucs_mid_fusion import (
    configure_parser as configure_htdemucs_mid_fusion_parser,
)
from stemfusion.training.htdemucs_rms_weighted_late_fusion import (
    configure_parser as configure_htdemucs_rms_weighted_late_fusion_parser,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stemfusion")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )

    commands = parser.add_subparsers(dest="command", required=True)
    separate_parser = commands.add_parser(
        "separate",
        help="Run a music source-separation backend.",
    )
    separators = separate_parser.add_subparsers(
        dest="separator",
        required=True,
    )
    configure_htdemucs_parser(separators.add_parser("htdemucs"))

    data_parser = commands.add_parser(
        "data",
        help="Build and validate dataset manifests.",
    )
    data_commands = data_parser.add_subparsers(
        dest="data_command",
        required=True,
    )
    configure_fma_parser(data_commands.add_parser("fma-manifest"))

    extract_parser = commands.add_parser(
        "extract",
        help="Run a frozen feature extractor.",
    )
    extractors = extract_parser.add_subparsers(
        dest="extractor",
        required=True,
    )
    configure_htdemucs_features_parser(
        extractors.add_parser(
            "htdemucs",
            help="Extract MusicFM embeddings or RMS energy from HT-Demucs outputs.",
        )
    )

    train_parser = commands.add_parser(
        "train",
        help="Train a downstream classifier from cached features.",
    )
    trainers = train_parser.add_subparsers(
        dest="training_task",
        required=True,
    )
    configure_genre_training_parser(trainers.add_parser("fma-genre"))
    configure_htdemucs_mid_fusion_parser(
        trainers.add_parser("fma-htdemucs-mid-fusion")
    )
    configure_htdemucs_late_fusion_parser(
        trainers.add_parser("fma-htdemucs-late-fusion")
    )
    configure_htdemucs_rms_weighted_late_fusion_parser(
        trainers.add_parser("fma-htdemucs-rms-weighted-late-fusion")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return int(args.command_handler(args))
