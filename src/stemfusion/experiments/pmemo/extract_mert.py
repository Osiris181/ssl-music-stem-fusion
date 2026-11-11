"""Extract PMEmo mixture/stem MERT segments using the shared Jamendo runtime."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments.pmemo import _extraction


def configure_parser(parser):
    _extraction.configure_parser(parser, "mert")


def main(argv=None):
    return _extraction.main("mert", argv)


if __name__ == "__main__":
    raise SystemExit(main())
