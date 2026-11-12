"""Run EmoMusic separation through the shared HT-Demucs implementation."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._separation import configure_parser as _configure_parser, main as _main


def configure_parser(parser) -> None:
    _configure_parser(parser, dataset="emomusic")


def main(argv=None) -> int:
    return _main(argv, dataset="emomusic")


if __name__ == "__main__":
    raise SystemExit(main())
