"""Run EmoMusic native-score binary mid fusion through the shared Music2Emo fitter."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._emomusic_mid_fusion import configure_parser as _configure_parser, direct_main


def configure_parser(parser):
    _configure_parser(parser, dataset="emomusic")


def main(argv=None):
    return direct_main(argv, dataset="emomusic")


if __name__ == "__main__":
    raise SystemExit(main())
