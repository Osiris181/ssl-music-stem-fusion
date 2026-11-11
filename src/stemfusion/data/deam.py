"""Preserve DEAM's supplied song-level annotations and fixed split assignments."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS
from stemfusion.training.reporting import file_checksum, write_json

SPLITS = ("train", "validation", "test")


def _track_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError(f"Expected a canonical positive DEAM song ID: {value!r}")
    return value


def _scores(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) != {"valence", "arousal"}:
        raise ValueError("DEAM requires continuous valence and arousal scores.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 1 <= value <= 9
           for value in values.values()):
        raise ValueError("Preserve the supplied DEAM 1-9 annotation scale; do not convert labels during extraction.")
    return values


def read_feature_manifest(path: Path) -> list[dict]:
    """Validate DEAM identities without scanning/decoding the audio corpus."""
    records, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            track_id = _track_id(row["track_id"])
            if (row.get("dataset") != "deam" or row["relative_path"] != f"{track_id}.mp3"
                    or row["split"] not in SPLITS or track_id in seen):
                raise ValueError(f"Invalid DEAM identity, path, split or duplicate: {track_id}")
            _scores(row["annotations"])
            paths = row.get("paths")
            if (not isinstance(paths, dict) or "mixture" not in paths
                    or set(paths) - {"mixture", *HTDEMUCS_6S_STEMS}
                    or any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths.values())):
                raise ValueError(f"Expected absolute DEAM source paths: {track_id}")
            seen.add(track_id)
            records.append(row)
    if not records:
        raise ValueError(f"Empty DEAM feature manifest: {path}")
    return records


def build_feature_manifest(*, split_files: dict[str, Path], annotations: Path,
                           mixture_root: Path, htdemucs_root: Path, output: Path,
                           overwrite: bool = False, dry_run: bool = False) -> dict:
    """Bind existing song-level scores and splits; do not average or binarize labels."""
    if set(split_files) != set(SPLITS):
        raise ValueError("Provide DEAM train, validation and test split files.")
    scores = {}
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"song_id", "valence_mean", "arousal_mean"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("Expected supplied DEAM song_id, valence_mean and arousal_mean columns.")
        for row in reader:
            track_id = _track_id(row["song_id"])
            if track_id in scores:
                raise ValueError(f"Repeated annotated DEAM ID: {track_id}")
            scores[track_id] = _scores({"valence": float(row["valence_mean"]),
                                       "arousal": float(row["arousal_mean"])})
    records, seen = [], set()
    for split in SPLITS:
        ids = split_files[split].read_text(encoding="utf-8-sig").split()
        if not ids:
            raise ValueError(f"Empty DEAM split: {split}")
        for value in ids:
            track_id = _track_id(value)
            if track_id in seen:
                raise ValueError(f"Repeated DEAM ID within/across splits: {track_id}")
            if track_id not in scores:
                raise ValueError(f"Split ID lacks a DEAM annotation: {track_id}")
            seen.add(track_id)
            paths = {"mixture": str(mixture_root.resolve() / f"{track_id}.mp3"),
                     **{stem: str(htdemucs_root.resolve() / track_id / f"{stem}.mp3")
                        for stem in HTDEMUCS_6S_STEMS}}
            records.append({"dataset": "deam", "track_id": track_id, "relative_path": f"{track_id}.mp3",
                            "split": split, "annotations": scores[track_id], "paths": paths})
    if seen != scores.keys():
        raise ValueError("Supplied splits must cover exactly the annotated DEAM IDs; no automatic exclusions.")
    inputs = {"annotations": annotations, **{f"{split}_split": path for split, path in split_files.items()}}
    report = {"dataset": "deam", "purpose": "feature_extraction", "records": len(records),
              "split_counts": dict(Counter(row["split"] for row in records)),
              "input_files": {name: {"path": str(path), "sha256": file_checksum(path)} for name, path in inputs.items()},
              "annotation_scale": "supplied_1_9", "annotation_aggregation": "use_supplied_song_level_means",
              "binary_labels_created": False, "audio_files_checked": False, "dry_run": dry_run,
              "mixture_root": str(mixture_root.resolve()), "htdemucs_root": str(htdemucs_root.resolve())}
    if dry_run:
        return report
    sidecar = output.with_suffix(".report.json")
    if not overwrite and any(path.exists() for path in (output, sidecar)):
        raise ValueError(f"DEAM feature manifest exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, report)
    return report
