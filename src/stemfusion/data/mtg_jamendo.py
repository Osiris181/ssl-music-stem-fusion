"""Preserve provided mood/theme splits when selecting Jamendo audio for separation."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS
from stemfusion.training.reporting import file_checksum, write_json

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class JamendoSeparationManifestConfig:
    split_files: dict[str, Path]
    mixture_root: Path
    output_root: Path
    output_path: Path
    track_list: Path | None = None
    all_tracks: bool = False
    overwrite: bool = False
    dry_run: bool = False


def read_mood_splits(split_files: dict[str, Path]) -> list[dict]:
    if set(split_files) != set(SPLITS):
        raise ValueError("Provide Jamendo train, validation, and test TSVs.")
    rows = []
    seen_ids, seen_paths, tags = set(), set(), set()
    for split in SPLITS:
        with split_files[split].open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader, None) != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS"]:
                raise ValueError(f"Unexpected Jamendo TSV header: {split_files[split]}")
            count = 0
            for number, row in enumerate(reader, 2):
                if not row:
                    continue
                if len(row) < 6 or not re.fullmatch(r"track_\d{7}", row[0]):
                    raise ValueError(f"Invalid Jamendo row: {split_files[split]}:{number}")
                relative = PurePosixPath(row[3])
                if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.suffix != ".mp3":
                    raise ValueError(f"Unsafe Jamendo audio path: {row[3]}")
                numeric_id = str(int(row[0].removeprefix("track_")))
                if relative.stem != numeric_id or relative.parts[0] != numeric_id.zfill(2)[-2:]:
                    raise ValueError(f"Jamendo path does not match its track ID: {row[0]}, {relative}")
                if row[0] in seen_ids or str(relative) in seen_paths:
                    raise ValueError(f"Duplicate Jamendo track within or across splits: {row[0]}")
                duration = float(row[4])
                if not math.isfinite(duration) or duration <= 0:
                    raise ValueError(f"Invalid Jamendo duration: {row[0]}")
                if any(not tag.startswith("mood/theme---") for tag in row[5:]):
                    raise ValueError(f"Expected mood/theme tags: {row[0]}")
                seen_ids.add(row[0]); seen_paths.add(str(relative)); tags.update(row[5:])
                rows.append({"track_id": row[0], "split": split, "relative_path": str(relative),
                             "duration_seconds": duration, "tags": row[5:]})
                count += 1
            if not count:
                raise ValueError(f"Empty Jamendo split: {split}")
    if len(tags) != 56:
        raise ValueError(f"Expected the provided 56-tag mood/theme setting, found {len(tags)} tags.")
    return rows


def build_separation_manifest(config: JamendoSeparationManifestConfig) -> dict:
    """Bind selected metadata records to new stem destinations; never scan audio."""
    rows = read_mood_splits(config.split_files)
    if config.all_tracks:
        selected = rows
    else:
        if config.track_list is None:
            raise ValueError("Provide --track-list, or use --all-tracks explicitly.")
        requested = config.track_list.read_text().split()
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Track list must contain distinct canonical track IDs.")
        by_id = {row["track_id"]: row for row in rows}
        if set(requested) - by_id.keys():
            raise ValueError("Track list contains IDs outside the provided mood/theme splits.")
        selected = [by_id[track_id] for track_id in requested]
    records = []
    for row in selected:
        numeric_id = str(int(row["track_id"].removeprefix("track_")))
        target = config.output_root / "htdemucs_6s" / numeric_id
        paths = {"mixture": str(config.mixture_root / row["relative_path"]),
                 **{stem: str(target / f"{stem}.mp3") for stem in HTDEMUCS_6S_STEMS}}
        records.append({**row, "paths": paths})
    report = {"dataset": "MTG-Jamendo", "purpose": "source_separation", "number_of_tags": 56,
              "input_split_counts": dict(Counter(row["split"] for row in rows)),
              "selected_split_counts": {split: sum(row["split"] == split for row in selected) for split in SPLITS},
              "records": len(records), "all_tracks": config.all_tracks,
              "track_list": str(config.track_list) if config.track_list is not None and not config.all_tracks else None,
              "track_list_sha256": file_checksum(config.track_list) if config.track_list is not None and not config.all_tracks else None,
              "split_files": {split: {"path": str(path), "sha256": file_checksum(path)} for split, path in config.split_files.items()},
              "output_root": str(config.output_root), "dry_run": config.dry_run,
              "audio_files_checked": False}
    if config.dry_run:
        return report
    sidecar = config.output_path.with_suffix(".report.json")
    if not config.overwrite and any(path.exists() for path in (config.output_path, sidecar)):
        raise ValueError(f"Manifest already exists; use --overwrite explicitly: {config.output_path}")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_name(f".{config.output_path.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        temporary.replace(config.output_path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, report)
    return report


def build_feature_manifest(*, split_files: dict[str, Path], mixture_root: Path,
                           htdemucs_root: Path, output: Path,
                           audio_overrides: Path | None = None,
                           overwrite: bool = False, dry_run: bool = False) -> dict:
    """Bind all provided splits to existing audio, overlaying validated repairs.

    Only metadata is read. Availability and decoded segment counts are checked
    during extraction; no track is removed from the supplied splits here.
    """
    rows = read_mood_splits(split_files)
    by_id = {row["track_id"]: row for row in rows}
    overrides = {}
    if audio_overrides is not None:
        with audio_overrides.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                track_id = row["track_id"]
                original = by_id.get(track_id)
                if original is None or track_id in overrides:
                    raise ValueError(f"Unknown or repeated override track: {track_id}")
                if any(row.get(key) != original[key] for key in ("split", "relative_path", "tags")):
                    raise ValueError(f"Audio override changes split/path/tags: {track_id}")
                paths = row.get("paths")
                if not isinstance(paths, dict) or set(paths) != {"mixture", *HTDEMUCS_6S_STEMS} or any(
                    not isinstance(value, str) or not Path(value).is_absolute() for value in paths.values()
                ):
                    raise ValueError(f"Audio override requires absolute paths for all native sources: {track_id}")
                overrides[track_id] = paths
    records = []
    for row in rows:
        numeric_id = str(int(row["track_id"].removeprefix("track_")))
        paths = {"mixture": str(mixture_root / row["relative_path"]),
                 **{stem: str(htdemucs_root / numeric_id / f"{stem}.mp3") for stem in HTDEMUCS_6S_STEMS}}
        paths.update(overrides.get(row["track_id"], {}))
        records.append({**row, "paths": paths})
    report = {"dataset": "mtg_jamendo", "purpose": "feature_extraction", "records": len(records),
              "split_counts": dict(Counter(row["split"] for row in rows)), "number_of_tags": 56,
              "audio_overrides": len(overrides), "audio_files_checked": False, "dry_run": dry_run,
              "split_files": {split: {"path": str(path), "sha256": file_checksum(path)} for split, path in split_files.items()},
              "audio_override_file": str(audio_overrides) if audio_overrides is not None else None,
              "audio_override_sha256": file_checksum(audio_overrides) if audio_overrides is not None else None}
    if dry_run:
        return report
    sidecar = output.with_suffix(".report.json")
    if not overwrite and any(path.exists() for path in (output, sidecar)):
        raise ValueError(f"Feature manifest exists; use --overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, report)
    return report
