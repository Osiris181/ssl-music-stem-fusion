"""Run GTZAN mid fusion with the shared genre experiment implementation."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._genre import configure_parser as _configure_parser, direct_main


def configure_parser(parser) -> None:
    _configure_parser(parser, "mid_fusion", dataset="gtzan")


def main(argv=None) -> int:
    return direct_main("mid_fusion", argv, dataset="gtzan")


if __name__ == "__main__":
    raise SystemExit(main())
