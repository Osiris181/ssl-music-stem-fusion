"""Extract DEAM retained-tail MERT embeddings through the shared backend."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments import _emotion_features


def configure_parser(parser):
    _emotion_features.configure_parser(parser, "mert", "deam")


def main(argv=None):
    return _emotion_features.main("deam", "mert", argv)


if __name__ == "__main__":
    raise SystemExit(main())
