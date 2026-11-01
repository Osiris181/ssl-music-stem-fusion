"""Shared separation CLI used by directly runnable dataset entry points."""

from __future__ import annotations

import logging

from stemfusion.configuration import YAMLArgumentParser, configure_yaml
from stemfusion.separation.htdemucs import configure_parser as configure_htdemucs, run_from_args


def configure_parser(parser, *, dataset: str) -> None:
    configure_htdemucs(parser)
    configure_yaml(parser, "separation", f"experiments/{dataset}/separation.yaml")
    parser.set_defaults(separation_dataset=dataset)


def main(argv=None, *, dataset: str) -> int:
    parser = YAMLArgumentParser(prog="separation.py")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    configure_parser(parser, dataset=dataset)
    args = parser.parse_args(argv)
    if args.yaml_provenance["experiment_metadata"].get("dataset") != dataset:
        parser.error("The YAML dataset must match the selected separation entry point.")
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s | %(levelname)s | %(message)s")
    return run_from_args(args)
