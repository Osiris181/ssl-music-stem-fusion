"""Run PMEmo binary late fusion, matched mixture ensembles and stem diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._emotion_late_fusion import configure_parser, direct_main


def main(argv=None):
    return direct_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
