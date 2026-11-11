"""PMEmo chorus paths and original continuous scores, with fixed supplied splits."""

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
        raise ValueError(f"Expected a canonical positive PMEmo song ID: {value!r}")
    return value


def _scores(values: dict) -> dict:
    if not isinstance(values, dict) or set(values) != {"valence", "arousal"}:
        raise ValueError("PMEmo requires continuous valence and arousal scores.")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in values.values()):
        raise ValueError("Use original PMEmo scores on the 0-1 scale; no label conversion is performed.")
    return values


def read_feature_manifest(path: Path) -> list[dict]:
    """Validate flat numeric PMEmo IDs without pretending they are Jamendo IDs."""
    rows, seen = [], set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            track_id = _track_id(row["track_id"])
            if (row.get("dataset") != "pmemo" or row["relative_path"] != f"{track_id}.mp3"
                    or row["split"] not in SPLITS or track_id in seen):
                raise ValueError(f"Invalid PMEmo identity, path, split or duplicate: {track_id}")
            _scores(row["annotations"])
            paths = row.get("paths")
            if (not isinstance(paths, dict) or "mixture" not in paths
                    or set(paths) - {"mixture", *HTDEMUCS_6S_STEMS}
                    or any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths.values())):
                raise ValueError(f"Expected absolute PMEmo source paths: {track_id}")
            seen.add(track_id)
            rows.append(row)
    if not rows:
        raise ValueError(f"Empty PMEmo feature manifest: {path}")
    return rows


def build_feature_manifest(*, split_files: dict[str, Path], annotations: Path, metadata: Path,
                           mixture_root: Path, htdemucs_root: Path, output: Path,
                           overwrite: bool = False, dry_run: bool = False) -> dict:
    """Preserve all supplied assignments; read metadata only, never resplit audio."""
    if set(split_files) != set(SPLITS):
        raise ValueError("Provide PMEmo train, validation and test split files.")
    scores, audio = {}, {}
    with annotations.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["musicId", "Arousal(mean)", "Valence(mean)"]:
            raise ValueError("Expected the original PMEmo static_annotations.csv header.")
        for row in reader:
            track_id = _track_id(row["musicId"])
            if track_id in scores:
                raise ValueError(f"Repeated annotated PMEmo ID: {track_id}")
            scores[track_id] = _scores({"valence": float(row["Valence(mean)"]),
                                       "arousal": float(row["Arousal(mean)"])})
    with metadata.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"musicId", "fileName", "duration"} <= set(reader.fieldnames or []):
            raise ValueError("PMEmo metadata must provide musicId, fileName and duration.")
        for row in reader:
            track_id = _track_id(row["musicId"])
            duration = float(row["duration"])
            if (track_id in audio or row["fileName"] != f"{track_id}.mp3"
                    or not math.isfinite(duration) or duration <= 0):
                raise ValueError(f"Invalid PMEmo metadata record: {track_id}")
            audio[track_id] = duration
    records, seen = [], set()
    for split in SPLITS:
        ids = split_files[split].read_text(encoding="utf-8-sig").split()
        if not ids:
            raise ValueError(f"Empty PMEmo split: {split}")
        for value in ids:
            track_id = _track_id(value)
            if track_id in seen:
                raise ValueError(f"Repeated PMEmo ID within/across splits: {track_id}")
            if track_id not in scores or track_id not in audio:
                raise ValueError(f"Split ID lacks annotation or audio metadata: {track_id}")
            seen.add(track_id)
            paths = {"mixture": str(mixture_root.resolve() / f"{track_id}.mp3"),
                     **{stem: str(htdemucs_root.resolve() / track_id / f"{stem}.mp3")
                        for stem in HTDEMUCS_6S_STEMS}}
            records.append({"dataset": "pmemo", "track_id": track_id, "relative_path": f"{track_id}.mp3",
                            "split": split, "duration_seconds": audio[track_id],
                            "annotations": scores[track_id], "paths": paths})
    if seen != scores.keys():
        raise ValueError("Supplied splits must cover exactly the annotated PMEmo IDs; no automatic exclusions.")
    inputs = {"annotations": annotations, "metadata": metadata,
              **{f"{split}_split": path for split, path in split_files.items()}}
    report = {"dataset": "pmemo", "purpose": "feature_extraction", "records": len(records),
              "split_counts": dict(Counter(row["split"] for row in records)),
              "input_files": {key: {"path": str(path), "sha256": file_checksum(path)} for key, path in inputs.items()},
              "annotation_scale": "original_0_1", "binary_labels_created": False,
              "mixture_root": str(mixture_root.resolve()), "htdemucs_root": str(htdemucs_root.resolve()),
              "audio_files_checked": False, "dry_run": dry_run}
    if dry_run:
        return report
    sidecar = output.with_suffix(".report.json")
    if not overwrite and any(path.exists() for path in (output, sidecar)):
        raise ValueError(f"PMEmo feature manifest exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, report)
    return report
