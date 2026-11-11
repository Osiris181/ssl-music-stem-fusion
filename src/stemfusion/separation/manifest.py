"""Label-independent audio manifests consumed by source separation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SeparationRecord:
    track_id: str
    split: str
    paths: dict[str, str]


def read_manifest(path: Path) -> list[SeparationRecord]:
    """Read IDs and audio paths from single- or multi-label dataset manifests."""
    records = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                track_id, split, paths = row["track_id"], row["split"], row["paths"]
                if not isinstance(track_id, str) or not track_id:
                    raise ValueError("track_id must be a non-empty string")
                if split not in ("train", "validation", "test"):
                    raise ValueError("unknown split")
                if not isinstance(paths, dict) or not paths or any(
                    not isinstance(key, str) or not isinstance(value, str) or not value
                    for key, value in paths.items()
                ):
                    raise ValueError("paths must map source names to non-empty strings")
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid separation manifest record on line {line_number}: {error}") from error
            if track_id in seen:
                raise ValueError(f"Duplicate separation track ID: {track_id}")
            seen.add(track_id)
            records.append(SeparationRecord(track_id, split, paths))
    if not records:
        raise ValueError(f"Empty separation manifest: {path}")
    return records
