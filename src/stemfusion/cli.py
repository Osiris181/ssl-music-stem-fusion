"""Command-line interface for the stem-fusion pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from stemfusion.configuration import YAMLArgumentParser, configure_yaml

from stemfusion.data.fma import configure_parser as configure_fma_parser
from stemfusion.experiments.fma_medium.extraction import (
    configure_parser as configure_fma_extraction_parser,
)
from stemfusion.separation.htdemucs import (
    configure_parser as configure_htdemucs_parser,
)
from stemfusion.experiments.fma_medium.train_late import (
    configure_parser as configure_fma_late_fusion_parser,
)
from stemfusion.experiments.fma_medium.train_mid import (
    configure_parser as configure_fma_mid_fusion_parser,
)
from stemfusion.experiments.gtzan import extraction as gtzan_extraction
from stemfusion.experiments.gtzan import prepare_manifest as gtzan_manifest
from stemfusion.experiments.gtzan import train_mid as gtzan_mid
from stemfusion.experiments.gtzan import train_late as gtzan_late


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
        parser_class=YAMLArgumentParser,
    )
    htdemucs_parser = separators.add_parser("htdemucs")
    configure_htdemucs_parser(htdemucs_parser)
    configure_yaml(htdemucs_parser, "separation", "separators/htdemucs_6s.yaml")

    data_parser = commands.add_parser(
        "data",
        help="Build and validate dataset manifests.",
    )
    data_commands = data_parser.add_subparsers(
        dest="data_command",
        required=True,
        parser_class=YAMLArgumentParser,
    )
    configure_fma_parser(data_commands.add_parser("fma-manifest"))
    gtzan_manifest.configure_parser(data_commands.add_parser("gtzan-manifest"))

    extract_parser = commands.add_parser(
        "extract",
        help="Run a frozen feature extractor.",
    )
    extractors = extract_parser.add_subparsers(
        dest="extraction_dataset",
        required=True,
        parser_class=YAMLArgumentParser,
    )
    configure_fma_extraction_parser(
        extractors.add_parser(
            "fma-medium",
            help="Extract FMA-Medium mixture/stem MusicFM features.",
        )
    )
    gtzan_extraction.configure_parser(extractors.add_parser("gtzan"))

    train_parser = commands.add_parser(
        "train",
        help="Train a downstream classifier from cached features.",
    )
    datasets = train_parser.add_subparsers(
        dest="training_dataset",
        required=True,
    )
    fma_parser = datasets.add_parser(
        "fma-medium",
        help="FMA-Medium genre-classification experiments.",
    )
    methods = fma_parser.add_subparsers(
        dest="training_method",
        required=True,
        parser_class=YAMLArgumentParser,
    )
    configure_fma_mid_fusion_parser(methods.add_parser("mid-fusion"))
    configure_fma_late_fusion_parser(methods.add_parser("late-fusion"))
    gtzan_parser = datasets.add_parser("gtzan", help="GTZAN genre-classification experiments.")
    gtzan_methods = gtzan_parser.add_subparsers(dest="training_method", required=True, parser_class=YAMLArgumentParser)
    gtzan_mid.configure_parser(gtzan_methods.add_parser("mid-fusion"))
    gtzan_late.configure_parser(gtzan_methods.add_parser("late-fusion"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return int(args.command_handler(args))
