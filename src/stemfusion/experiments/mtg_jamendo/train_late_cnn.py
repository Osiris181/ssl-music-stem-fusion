"""Train plain CNN mood/theme late fusion using Jamendo's existing loss protocol."""
from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from stemfusion.experiments._cnn_mood_late import main


if __name__ == "__main__":
    raise SystemExit(main())
