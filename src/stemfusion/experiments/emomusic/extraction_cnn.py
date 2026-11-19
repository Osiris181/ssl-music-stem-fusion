"""Run EmoMusic CNN extraction with the shared official musicnn backend."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._cnn_runtime import prepare_runtime

if __name__ == "__main__":
    prepare_runtime()

from stemfusion.experiments._cnn_extraction import cache_valid, read_records, main as _main

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/experiments/emomusic/extraction_cnn.yaml"


def main(argv=None):
    return _main(argv, dataset="emomusic")


if __name__ == "__main__":
    raise SystemExit(main())
