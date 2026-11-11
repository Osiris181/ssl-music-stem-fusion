"""Temporal sampling of cached segment embeddings for mood/theme experiments."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def segment_files(directory: Path) -> list[Path]:
    """Return contiguous segment files in numeric time order; reject gaps."""
    indexed = []
    for path in directory.glob("segment_*.npy"):
        match = re.fullmatch(r"segment_(0|[1-9][0-9]*)\.npy", path.name)
        if match is None:
            raise ValueError(f"Invalid segment filename: {path}")
        indexed.append((int(match[1]), path))
    indexed.sort()
    if not indexed or [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"Expected nonempty, contiguous segments starting at zero: {directory}")
    return [path for _, path in indexed]


def pool_segment_views(views: dict[str, np.ndarray], *, training: bool,
                       rng: np.random.Generator | None = None) -> dict[str, np.ndarray]:
    """Use one shared temporal window for all mixture/stem feature sequences.

    Training matches the authors' conditional-uniform start/end sampling.
    Evaluation averages every segment with equal weight, including the tail.
    Inputs must be aligned in time, with the same segment partition and count.
    """
    if not views:
        raise ValueError("At least one segment view is required.")
    arrays = {source: np.asarray(value, dtype=np.float32) for source, value in views.items()}
    counts = set()
    for source, value in arrays.items():
        if value.ndim != 2 or not all(value.shape) or not np.isfinite(value).all():
            raise ValueError(f"Expected finite nonempty (segments, dimension) features: {source}")
        counts.add(value.shape[0])
    if len(counts) != 1:
        raise ValueError("Mixture/stem segment counts differ; verify temporal alignment before pooling.")
    count = counts.pop()
    start, end = 0, count
    if training:
        if rng is None:
            raise ValueError("Training segment sampling requires a seeded RNG.")
        start = int(rng.integers(0, count))
        end = int(rng.integers(start + 1, count + 1))
    return {source: value[start:end].mean(axis=0) for source, value in arrays.items()}
