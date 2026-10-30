"""Run FMA-Medium extraction with the shared MusicFM implementation."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._extraction import (
    MusicFMExtractionConfig, MusicFMExtractionError, MusicFMExtractionSummary,
    configure_parser, extract_musicfm, extract_musicfm_sources, main,
    musicfm_config_from_args, run_musicfm_from_args,
)

if __name__ == "__main__":
    raise SystemExit(main())
