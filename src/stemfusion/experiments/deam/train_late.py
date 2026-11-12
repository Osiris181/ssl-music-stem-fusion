"""Run DEAM native-score binary late fusion and standalone stem diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._emotion_native_late_fusion import configure_parser as _configure_parser, direct_main


def configure_parser(parser):
    _configure_parser(parser, dataset="deam")


def main(argv=None):
    return direct_main(argv, dataset="deam")


if __name__ == "__main__":
    raise SystemExit(main())
