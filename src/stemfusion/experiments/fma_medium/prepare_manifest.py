"""Generate the aligned FMA-Medium manifest from metadata and audio paths."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.data.fma import configure_parser as _configure_parser
from stemfusion.configuration import YAMLArgumentParser, configure_yaml

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[4]
    / "data/manifests/fma_medium/fma_medium_htdemucs.jsonl"
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    _configure_parser(parser, default_output=DEFAULT_OUTPUT)
    configure_yaml(parser, "fma_manifest", "experiments/fma_medium/manifest.yaml")


def main(argv: Sequence[str] | None = None) -> int:
    parser = YAMLArgumentParser(prog="prepare_manifest.py")
    configure_parser(parser)
    args = parser.parse_args(argv)
    return int(args.command_handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
