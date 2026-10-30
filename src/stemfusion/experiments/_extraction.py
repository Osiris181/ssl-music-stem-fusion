"""Shared MusicFM extraction interface for dataset-specific entry points."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.experiments._datasets import get_dataset
from stemfusion.data.fma import HTDEMUCS_SOURCES
from stemfusion.features._musicfm import (
    MusicFMExtractionConfig,
    MusicFMExtractionError,
    MusicFMExtractionSummary,
    extract_musicfm,
    extract_musicfm_sources,
    _atomic_write_json,
)

LOGGER = logging.getLogger(__name__)
FEATURE_SOURCES = frozenset({"mixture", *HTDEMUCS_SOURCES})
SPLITS = ("train", "validation", "test")

__all__ = [
    "MusicFMExtractionConfig",
    "MusicFMExtractionError",
    "MusicFMExtractionSummary",
    "extract_musicfm",
    "extract_musicfm_sources",
    "musicfm_config_from_args",
    "configure_parser",
    "run_musicfm_from_args",
    "main",
]


def configure_parser(parser: argparse.ArgumentParser, *, dataset: str = "fma_medium") -> None:
    """Define a dataset task interface and choose its YAML configuration.

    Scientific defaults are loaded from YAML, not repeated in this parser.
    The dataset YAML can override its referenced shared feature preset.
    """

    parser.description = (
        f"Extract MusicFM features for {get_dataset(dataset).name} mixture audio or HT-Demucs "
        "stems using the dataset's YAML configuration."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source", choices=tuple(sorted(FEATURE_SOURCES)),
        action="append",
        help="Select one source; repeat for a subset. Omit to use the YAML sources list.",
    )
    parser.add_argument("--musicfm-root", type=Path, required=True)
    parser.add_argument("--stats-path", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--layer-index", type=int)
    parser.add_argument("--expected-dimension", type=int)
    parser.add_argument("--temporal-pooling", choices=("mean",))
    parser.add_argument("--is-flash", action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--split", action="append", choices=SPLITS, dest="splits",
    )
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.set_defaults(command_handler=run_musicfm_from_args)
    parser.set_defaults(genre_dataset=dataset)
    configure_yaml(parser, "extraction", f"experiments/{dataset}/extraction.yaml")


def musicfm_config_from_args(
    args: argparse.Namespace, *, source: str | None = None,
) -> MusicFMExtractionConfig:
    """Construct one source configuration from the dataset's parsed settings."""

    metadata = getattr(args, "yaml_provenance", {}).get("experiment_metadata", {})
    if metadata.get("dataset", "fma_medium") != getattr(args, "genre_dataset", "fma_medium"):
        raise MusicFMExtractionError("The YAML dataset must match the selected extraction entry point.")

    if source is None:
        sources = [args.source] if isinstance(args.source, str) else args.source
        if not sources or len(sources) != 1:
            raise MusicFMExtractionError(
                "A single-source configuration requires exactly one source."
            )
        source = sources[0]
    stats_path = args.stats_path
    checkpoint_path = args.checkpoint_path
    if stats_path is None and hasattr(args, "statistics_filename"):
        stats_path = args.musicfm_root / "data" / args.statistics_filename
    if checkpoint_path is None and hasattr(args, "checkpoint_filename"):
        checkpoint_path = args.musicfm_root / "data" / args.checkpoint_filename
    return MusicFMExtractionConfig(
        manifest_path=args.manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        source=source,
        musicfm_root=args.musicfm_root.expanduser().resolve(),
        stats_path=(
            stats_path.expanduser().resolve() if stats_path is not None else None
        ),
        checkpoint_path=(
            checkpoint_path.expanduser().resolve() if checkpoint_path is not None else None
        ),
        device=args.device,
        sample_rate=args.sample_rate,
        layer_index=args.layer_index,
        expected_dimension=args.expected_dimension,
        temporal_pooling=args.temporal_pooling,
        is_flash=args.is_flash,
        max_seconds=args.max_seconds,
        splits=tuple(args.splits),
        limit_per_split=args.limit_per_split,
        overwrite=args.overwrite,
        continue_on_error=not args.fail_fast,
        dry_run=args.dry_run,
        report_path=(
            args.report.expanduser().resolve()
            if args.report is not None
            else None
        ),
        config_provenance=getattr(args, "yaml_provenance", {}),
        dataset=getattr(args, "genre_dataset", "fma_medium"),
    )


def run_musicfm_from_args(args: argparse.Namespace) -> int:
    try:
        sources = [args.source] if isinstance(args.source, str) else args.source
        if not sources:
            raise MusicFMExtractionError("Set sources in YAML or select them with --source.")
        if len(set(sources)) != len(sources):
            raise MusicFMExtractionError("Each extraction source must be selected only once.")
        configs = [musicfm_config_from_args(args, source=source) for source in sources]
        if len(configs) == 1:
            summary = extract_musicfm(configs[0])
            result = summary.to_dict()
            exit_code = summary.exit_code
        else:
            # --report names the aggregate report for a multi-source job.
            # Individual source reports keep their distinct default filenames.
            configs = [replace(config, report_path=None) for config in configs]
            summaries = extract_musicfm_sources(configs)
            exit_code = int(any(summary.exit_code for summary in summaries))
            first = configs[0]
            result = {
                "requested_sources": sources,
                "processed_sources": [summary.source for summary in summaries],
                "unprocessed_sources": sources[len(summaries):],
                "dry_run": first.dry_run,
                "layer_index": first.layer_index,
                "checkpoint_path": str(first.effective_checkpoint_path),
                "stats_path": str(first.effective_stats_path),
                "output_directory": str(first.output_dir),
                "planned_features": sum(summary.planned for summary in summaries),
                "succeeded": sum(summary.succeeded for summary in summaries),
                "failed": sum(summary.failed for summary in summaries),
                "exit_code": exit_code,
                "sources": {summary.source: summary.to_dict() for summary in summaries},
            }
            if not first.dry_run:
                report_path = (
                    args.report.expanduser().resolve() if args.report is not None
                    else first.output_dir / "extraction_report_musicfm_batch.json"
                )
                _atomic_write_json(report_path, result)
    except MusicFMExtractionError as error:
        LOGGER.error("%s", error)
        return 2
    return exit_code


def main(argv: Sequence[str] | None = None, *, dataset: str = "fma_medium") -> int:
    """Run a dataset extraction task directly using the same parser and YAML."""

    parser = YAMLArgumentParser(prog="extraction.py")
    parser.add_argument(
        "--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    configure_parser(parser, dataset=dataset)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    return int(args.command_handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
