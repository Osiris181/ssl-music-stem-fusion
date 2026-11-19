"""Run Jamendo mood/theme CNN extraction with the shared official backend."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._cnn_runtime import prepare_runtime

if __name__ == "__main__":
    prepare_runtime()

from stemfusion.experiments._cnn_extraction import main as _main


def main(argv=None):
    return _main(argv, dataset="mtg_jamendo")


if __name__ == "__main__":
    raise SystemExit(main())
