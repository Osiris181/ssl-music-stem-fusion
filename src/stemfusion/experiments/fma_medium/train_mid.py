"""Run FMA-Medium mid-fusion and capacity-matched mixture controls."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments.fma_medium.training import configure_parser as _configure_parser, direct_main


def configure_parser(parser) -> None:
    _configure_parser(parser, "mid_fusion")


def main(argv=None) -> int:
    return direct_main("mid_fusion", argv)


if __name__ == "__main__":
    raise SystemExit(main())
