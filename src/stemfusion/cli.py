"""Command-line interface for the stem-fusion pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from stemfusion.configuration import YAMLArgumentParser, configure_yaml

from stemfusion.experiments.fma_medium.prepare_manifest import configure_parser as configure_fma_parser
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
from stemfusion.experiments.ismir04 import prepare_manifest as ismir04_manifest
from stemfusion.experiments.ismir04 import extraction as ismir04_extraction
from stemfusion.experiments.ismir04 import train_mid as ismir04_mid
from stemfusion.experiments.ismir04 import train_late as ismir04_late
from stemfusion.experiments.mtg_jamendo import prepare_manifest as jamendo_manifest
from stemfusion.experiments.mtg_jamendo import extract_chords_keys as jamendo_chords_keys
from stemfusion.experiments.mtg_jamendo import extract_mert as jamendo_mert
from stemfusion.experiments.mtg_jamendo import train_reproduction as jamendo_reproduction
from stemfusion.experiments.mtg_jamendo import train_mid as jamendo_mid
from stemfusion.experiments.mtg_jamendo import train_late as jamendo_late
from stemfusion.experiments.pmemo import prepare_manifest as pmemo_manifest
from stemfusion.experiments.pmemo import extract_mert as pmemo_mert
from stemfusion.experiments.pmemo import extract_chords_keys as pmemo_chords_keys
from stemfusion.experiments.pmemo import train_mid as pmemo_mid
from stemfusion.experiments.pmemo import train_late as pmemo_late
from stemfusion.experiments.deam import prepare_manifest as deam_manifest
from stemfusion.experiments.deam import extract_mert as deam_mert
from stemfusion.experiments.deam import extract_chords_keys as deam_chords_keys
from stemfusion.experiments.deam import train_mid as deam_mid
from stemfusion.experiments.emomusic import prepare_feature_manifest as emomusic_manifest
from stemfusion.experiments.emomusic import extract_mert as emomusic_mert
from stemfusion.experiments.emomusic import extract_chords_keys as emomusic_chords_keys


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
    ismir04_manifest.configure_parser(data_commands.add_parser("ismir04-manifest"))
    jamendo_manifest.configure_parser(data_commands.add_parser("mtg-jamendo-manifest"))
    jamendo_manifest.configure_parser(data_commands.add_parser("mtg-jamendo-feature-manifest",
        help="Compatibility alias for mtg-jamendo-manifest."))
    jamendo_manifest.configure_parser(data_commands.add_parser("mtg-jamendo-separation-manifest",
        help="Compatibility alias for mtg-jamendo-manifest --purpose separation."), purpose="separation")
    pmemo_manifest.configure_parser(data_commands.add_parser("pmemo-manifest"))
    pmemo_manifest.configure_parser(data_commands.add_parser("pmemo-feature-manifest",
        help="Compatibility alias for pmemo-manifest."))
    pmemo_manifest.configure_parser(data_commands.add_parser("pmemo-separation-manifest",
        help="Compatibility alias for pmemo-manifest --purpose separation."), purpose="separation")
    deam_manifest.configure_parser(data_commands.add_parser("deam-manifest"))
    deam_manifest.configure_parser(data_commands.add_parser("deam-feature-manifest",
        help="Compatibility alias for deam-manifest."))
    deam_manifest.configure_parser(data_commands.add_parser("deam-separation-manifest",
        help="Compatibility alias for deam-manifest --purpose separation."), purpose="separation")
    emomusic_manifest.configure_parser(data_commands.add_parser("emomusic-feature-manifest"))

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
    ismir04_extraction.configure_parser(extractors.add_parser("ismir04"))
    jamendo_parser = extractors.add_parser("mtg-jamendo")
    jamendo_methods = jamendo_parser.add_subparsers(dest="representation", required=True, parser_class=YAMLArgumentParser)
    jamendo_chords_keys.configure_parser(jamendo_methods.add_parser("chords-keys"))
    jamendo_mert.configure_parser(jamendo_methods.add_parser("mert"))
    pmemo_parser = extractors.add_parser("pmemo")
    pmemo_methods = pmemo_parser.add_subparsers(dest="representation", required=True, parser_class=YAMLArgumentParser)
    pmemo_mert.configure_parser(pmemo_methods.add_parser("mert"))
    pmemo_chords_keys.configure_parser(pmemo_methods.add_parser("chords-keys"))
    deam_parser = extractors.add_parser("deam")
    deam_methods = deam_parser.add_subparsers(dest="representation", required=True, parser_class=YAMLArgumentParser)
    deam_mert.configure_parser(deam_methods.add_parser("mert"))
    deam_chords_keys.configure_parser(deam_methods.add_parser("chords-keys"))
    emomusic_parser = extractors.add_parser("emomusic")
    emomusic_methods = emomusic_parser.add_subparsers(dest="representation", required=True, parser_class=YAMLArgumentParser)
    emomusic_mert.configure_parser(emomusic_methods.add_parser("mert"))
    emomusic_chords_keys.configure_parser(emomusic_methods.add_parser("chords-keys"))

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
    ismir04_parser = datasets.add_parser("ismir04", help="ISMIR04 genre-classification experiments.")
    ismir04_methods = ismir04_parser.add_subparsers(dest="training_method", required=True, parser_class=YAMLArgumentParser)
    ismir04_mid.configure_parser(ismir04_methods.add_parser("mid-fusion"))
    ismir04_late.configure_parser(ismir04_methods.add_parser("late-fusion"))
    jamendo_train = datasets.add_parser("mtg-jamendo", help="MTG-Jamendo mood/theme experiments.")
    jamendo_training_methods = jamendo_train.add_subparsers(dest="training_method", required=True, parser_class=YAMLArgumentParser)
    jamendo_reproduction.configure_parser(jamendo_training_methods.add_parser("author-reproduction"))
    jamendo_mid.configure_parser(jamendo_training_methods.add_parser("mid-fusion"))
    jamendo_late.configure_parser(jamendo_training_methods.add_parser("late-fusion"))
    pmemo_train = datasets.add_parser("pmemo", help="PMEmo separate binary valence/arousal experiments.")
    pmemo_training_methods = pmemo_train.add_subparsers(dest="training_method", required=True, parser_class=YAMLArgumentParser)
    pmemo_mid.configure_parser(pmemo_training_methods.add_parser("mid-fusion"))
    pmemo_late.configure_parser(pmemo_training_methods.add_parser("late-fusion"))
    deam_train = datasets.add_parser("deam", help="DEAM separate binary valence/arousal experiments using native 1-9 scores.")
    deam_training_methods = deam_train.add_subparsers(dest="training_method", required=True, parser_class=YAMLArgumentParser)
    deam_mid.configure_parser(deam_training_methods.add_parser("mid-fusion"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return int(args.command_handler(args))
