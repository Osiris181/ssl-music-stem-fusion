"""Run GTZAN extraction with the same MusicFM implementation as FMA-Medium."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._extraction import configure_parser as _configure_parser, main as _main


def configure_parser(parser) -> None:
    _configure_parser(parser, dataset="gtzan")


def main(argv=None) -> int:
    return _main(argv, dataset="gtzan")


if __name__ == "__main__":
    raise SystemExit(main())
