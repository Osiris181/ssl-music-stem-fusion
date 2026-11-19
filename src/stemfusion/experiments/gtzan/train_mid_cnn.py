"""Train GTZAN CNN mid fusion using the shared genre trainer."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._cnn_genre import config_from_args as _config_from_args, main as _main

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/experiments/gtzan/mid_fusion_cnn.yaml"


def config_from_args(args):
    return _config_from_args(args, dataset="gtzan")


def main(argv=None):
    return _main(argv, dataset="gtzan")


if __name__ == "__main__":
    raise SystemExit(main())
