"""Run the Jamendo mid-fusion matrix through the shared experiment controller."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._mood_theme_tagging import configure_parser as _configure_parser, direct_main


def configure_parser(parser):
    _configure_parser(parser, "mid_fusion")


def main(argv=None):
    return direct_main("mid_fusion", argv)


if __name__ == "__main__":
    raise SystemExit(main())
