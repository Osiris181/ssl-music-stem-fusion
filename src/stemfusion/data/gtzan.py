"""GTZAN split parsing and manifests for the shared feature pipeline."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from stemfusion.data.fma import FMAManifestRecord as ManifestRecord, read_manifest
from stemfusion.training.reporting import file_checksum, write_json

SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")
SPLITS = ("train", "validation", "test")
LABEL_MAPPING = {genre: index for index, genre in enumerate(
    ("blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"))}


@dataclass(frozen=True)
class GTZANManifestConfig:
    split_files: dict[str, Path]
    mixture_root: Path
    htdemucs_root: Path
    output_path: Path
    excluded_track_ids: tuple[str, ...] = ()
    mixture_extension: str = ".wav"
    stem_extension: str = ".mp3"
    check_audio: bool = False
    overwrite: bool = False
    dry_run: bool = False


def read_split_files(split_files: dict[str, Path]) -> dict[str, list[tuple[str, str]]]:
    """Preserve provided assignments and ordering; reject duplicate/unsafe IDs."""
    if set(split_files) != set(SPLITS):
        raise ValueError("Provide GTZAN train, validation, and test split files.")
    grouped = {}
    seen = set()
    for split in SPLITS:
        rows = []
        for line_number, line in enumerate(split_files[split].read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            relative = PurePosixPath(line.strip())
            if relative.is_absolute() or len(relative.parts) != 2 or relative.suffix != ".wav":
                raise ValueError(f"Expected genre/genre.00000.wav: {split_files[split]}:{line_number}")
            genre, filename = relative.parts
            track_id = Path(filename).stem
            if genre not in LABEL_MAPPING or not re.fullmatch(re.escape(genre) + r"\.\d{5}", track_id):
                raise ValueError(f"Invalid GTZAN genre/track ID: {relative}")
            if track_id in seen:
                raise ValueError(f"Duplicate GTZAN track within or across splits: {track_id}")
            seen.add(track_id)
            rows.append((track_id, genre))
        if not rows:
            raise ValueError(f"Empty GTZAN split: {split}")
        grouped[split] = rows
    return grouped


def inspect_manifest(manifest: Path, number_of_classes: int = 10) -> tuple[list, dict]:
    records = read_manifest(manifest)
    if number_of_classes != len(LABEL_MAPPING):
        raise ValueError("GTZAN requires ten genre classes.")
    for record in records:
        if record.genre not in LABEL_MAPPING or record.label != LABEL_MAPPING[record.genre]:
            raise ValueError(f"GTZAN label mapping mismatch: {record.track_id}")
        if not re.fullmatch(re.escape(record.genre) + r"\.\d{5}", record.track_id):
            raise ValueError(f"Invalid GTZAN track ID: {record.track_id}")
        if set(record.paths) != set(SOURCES):
            raise ValueError(f"GTZAN manifest must contain all six selected sources: {record.track_id}")
    for split in SPLITS:
        if {record.label for record in records if record.split == split} != set(LABEL_MAPPING.values()):
            raise ValueError(f"Every genre must be present in GTZAN {split}.")
    return records, dict(LABEL_MAPPING)


def build_gtzan_manifest(config: GTZANManifestConfig) -> dict:
    """Bind the fixed split IDs to local audio paths without scanning audio."""
    grouped = read_split_files(config.split_files)
    excluded = set(config.excluded_track_ids)
    known = {track for rows in grouped.values() for track, _ in rows}
    if len(excluded) != len(config.excluded_track_ids) or excluded - known:
        raise ValueError("Excluded GTZAN IDs must be distinct IDs present in the split files.")
    for extension in (config.mixture_extension, config.stem_extension):
        if not re.fullmatch(r"\.[a-zA-Z0-9]+", extension):
            raise ValueError(f"Invalid audio extension: {extension}")
    records = []
    for split in SPLITS:
        for track_id, genre in grouped[split]:
            if track_id in excluded:
                continue
            paths = {"mixture": str(config.mixture_root / genre / f"{track_id}{config.mixture_extension}")}
            paths.update({source: str(config.htdemucs_root / track_id / f"{source}{config.stem_extension}")
                          for source in SOURCES[1:]})
            if config.check_audio:
                for path in paths.values():
                    if not Path(path).is_file() or not Path(path).stat().st_size:
                        raise ValueError(f"Missing or empty configured audio: {path}")
            records.append(ManifestRecord(track_id, split, genre, LABEL_MAPPING[genre], paths))
    for split in SPLITS:
        if {record.label for record in records if record.split == split} != set(LABEL_MAPPING.values()):
            raise ValueError(f"Every genre must remain in GTZAN {split} after exclusions.")
    report = {
        "dataset": "GTZAN", "split_description": "provided stratified splits",
        "input_split_counts": {split: len(rows) for split, rows in grouped.items()},
        "split_counts": dict(Counter(record.split for record in records)),
        "excluded_track_ids": sorted(excluded), "exclusion_reason": "Unavailable tracks explicitly listed in the configuration.",
        "label_mapping": LABEL_MAPPING, "records": len(records), "split_ids_disjoint": True,
        "audio_files_revalidated": config.check_audio, "dry_run": config.dry_run,
        "split_files": {split: {"path": str(path), "sha256": file_checksum(path)} for split, path in config.split_files.items()},
    }
    if config.dry_run:
        return report
    sidecars = [config.output_path.with_suffix(f".{kind}.json") for kind in ("metadata", "report")]
    if not config.overwrite and any(path.exists() for path in (config.output_path, *sidecars)):
        raise ValueError(f"Manifest or sidecars already exist; use --overwrite explicitly: {config.output_path}")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_name(f".{config.output_path.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records), encoding="utf-8")
        temporary.replace(config.output_path)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecars[0], {"dataset": "GTZAN", "label_mapping": LABEL_MAPPING, "sources": list(SOURCES), "split_names": list(SPLITS)})
    write_json(sidecars[1], report)
    return report
