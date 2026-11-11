"""Run PMEmo binary valence/arousal mid-fusion through the shared task controller."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._emotion_recognition import configure_parser, direct_main


def main(argv=None):
    return direct_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
