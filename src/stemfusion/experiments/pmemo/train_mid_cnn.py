"""Run PMEmo's plain CNN mid-fusion classifiers for valence and arousal."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._cnn_emotion import main as _main


def main(argv=None):
    return _main(argv, dataset="pmemo")


if __name__ == "__main__":
    raise SystemExit(main())
