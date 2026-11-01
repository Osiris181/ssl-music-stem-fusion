"""CSV-authoritative ISMIR04 partitions, duplicate exclusions, and unique IDs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

from stemfusion.data.audio_duplicates import audit_audio_files
from stemfusion.data.fma import FMAManifestRecord as ManifestRecord, read_manifest
from stemfusion.training.reporting import file_checksum, write_json

LABEL_MAPPING = {genre: index for index, genre in enumerate(
    ("classical", "electronic", "jazz_blues", "metal_punk", "rock_pop", "world"))}
SPLITS = {"training": "train", "development": "validation", "evaluation": "test"}
SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")


def inspect_manifest(manifest: Path, number_of_classes: int = 6) -> tuple[list, dict]:
    records = read_manifest(manifest)
    if number_of_classes != len(LABEL_MAPPING):
        raise ValueError("ISMIR04 requires six genre classes.")
    for record in records:
        if record.genre not in LABEL_MAPPING or record.label != LABEL_MAPPING[record.genre]:
            raise ValueError(f"ISMIR04 label mapping mismatch: {record.track_id}")
        if not record.track_id.startswith(f"{record.split}__"):
            raise ValueError(f"ISMIR04 ID must preserve its provided split: {record.track_id}")
        if set(record.paths) != set(SOURCES):
            raise ValueError(f"ISMIR04 manifest must contain exactly the six selected sources: {record.track_id}")
    for split in SPLITS.values():
        if {record.label for record in records if record.split == split} != set(LABEL_MAPPING.values()):
            raise ValueError(f"All six genres must be present in ISMIR04 {split}.")
    return records, dict(LABEL_MAPPING)


@dataclass(frozen=True)
class ISMIR04Track:
    track_id: str
    split: str
    genre: str
    source_split: str
    relative_path: str
    original_path: Path
    legacy_stem_name: str
    csv_row: int
    resolved_relative_path: str = ""


def read_ismir04_metadata(metadata_root: Path, audio_root: Path,
                         filename_overrides: dict[str, str] | None = None) -> list[ISMIR04Track]:
    filename_overrides = filename_overrides or {}
    tracks = []
    seen = set()
    for source_split, split in SPLITS.items():
        csv_path = metadata_root / source_split / "tracklist.csv"
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            for line_number, row in enumerate(csv.reader(stream), 1):
                if not row:
                    continue
                if len(row) not in (6, 7) or row[0] not in LABEL_MAPPING:
                    raise ValueError(f"Invalid ISMIR04 CSV row: {csv_path}:{line_number}")
                # Evaluation has an extra field after the audio filename.
                relative = PurePosixPath(row[5])
                if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() not in (".mp3", ".wav", ".au"):
                    raise ValueError(f"Invalid relative audio path: {row[5]}")
                identity = f"{source_split}/{relative}"
                if identity in seen:
                    raise ValueError(f"Duplicate CSV path: {identity}")
                seen.add(identity)
                resolved = PurePosixPath(filename_overrides.get(identity, str(relative)))
                if resolved.is_absolute() or ".." in resolved.parts or resolved.suffix.lower() not in (".mp3", ".wav", ".au"):
                    raise ValueError(f"Invalid corrected relative audio path: {resolved}")
                short_name = re.sub(r"[^A-Za-z0-9_-]", "_", relative.stem)[:64]
                track_id = f"{split}__{short_name}__{hashlib.sha256(identity.encode()).hexdigest()[:16]}"
                tracks.append(ISMIR04Track(track_id, split, row[0], source_split, str(relative),
                    audio_root / source_split / str(resolved), resolved.stem, line_number, str(resolved)))
    if set(filename_overrides) - seen:
        raise ValueError("Filename overrides must refer to source paths present in the CSVs.")
    return tracks


def select_unique_tracks(tracks: list[ISMIR04Track], audit: dict) -> tuple[list, list]:
    """Keep training before validation before test; never reshuffle partitions."""
    if not audit.get("complete"):
        raise ValueError("A complete audio duplicate audit is required.")
    rank = {"train": 0, "validation": 1, "test": 2}
    groups = defaultdict(list)
    for track in tracks:
        item = audit["files"].get(str(track.original_path))
        if item is None:
            raise ValueError(f"Audio audit does not contain {track.original_path}")
        groups[item["decoded_sha256"]].append(track)
    excluded = []
    excluded_ids = set()
    for digest, members in groups.items():
        if len(members) < 2:
            continue
        if len({track.genre for track in members}) != 1:
            raise ValueError(f"Identical audio has conflicting genre labels: {[str(track.original_path) for track in members]}")
        ordered = sorted(members, key=lambda track: (rank[track.split], track.relative_path))
        keeper = ordered[0]
        for track in ordered[1:]:
            excluded_ids.add(track.track_id)
            excluded.append({"track_id": track.track_id, "split": track.split, "path": str(track.original_path),
                "reason": "Identical decoded audio input", "decoded_sha256": digest,
                "retained_track_id": keeper.track_id, "retained_split": keeper.split,
                "retained_path": str(keeper.original_path)})
    return [track for track in tracks if track.track_id not in excluded_ids], excluded


def prepare_ismir04_manifest(*, metadata_root: Path, audio_root: Path, htdemucs_root: Path,
                             repair_root: Path, output: Path, audit_path: Path,
                             audit_workers: int = 2, overwrite: bool = False,
                             audit_only: bool = False,
                             verified_legacy_stems: tuple[str, ...] = (),
                             filename_overrides: dict[str, str] | None = None) -> dict:
    report_path = output.with_suffix(".report.json")
    metadata_path = output.with_suffix(".metadata.json")
    if not audit_only and not overwrite and any(path.exists() for path in (output, report_path, metadata_path)):
        raise ValueError(f"Manifest or sidecars already exist; use --overwrite explicitly: {output}")
    tracks = read_ismir04_metadata(metadata_root, audio_root, filename_overrides)
    audit = audit_audio_files([track.original_path for track in tracks], audit_path, workers=audit_workers)
    retained, excluded = select_unique_tracks(tracks, audit)
    known_paths = {f"{track.source_split}/{track.relative_path}" for track in tracks}
    if set(verified_legacy_stems) - known_paths:
        raise ValueError("Verified stem records must be full relative audio paths present in the CSVs.")
    for split in SPLITS.values():
        if {track.genre for track in retained if track.split == split} != set(LABEL_MAPPING):
            raise ValueError(f"All six genres must remain in {split}.")
    by_basename = defaultdict(list)
    for track in retained:
        by_basename[track.legacy_stem_name].append(track)
    records = []
    repairs = []
    for track in retained:
        stem_root = htdemucs_root / track.legacy_stem_name
        if len(by_basename[track.legacy_stem_name]) > 1:
            verified = f"{track.source_split}/{track.relative_path}" in verified_legacy_stems
            if not verified:
                stem_root = repair_root / track.track_id
                repairs.append({"track_id": track.track_id, "original_path": str(track.original_path),
                    "output_directory": str(stem_root), "reason": "Ambiguous legacy basename; needs its own stem set."})
        paths = {"mixture": str(track.original_path)}
        paths.update({source: str(stem_root / f"{source}.mp3") for source in SOURCES[1:]})
        records.append(ManifestRecord(track.track_id, track.split, track.genre, LABEL_MAPPING[track.genre], paths))
    report = {"dataset": "ISMIR04", "partition_policy": "Retain provided splits; remove confirmed decoded-audio duplicates.",
        "duplicate_priority": ["train", "validation", "test"],
        "input_split_counts": dict(Counter(track.split for track in tracks)),
        "split_counts": dict(Counter(track.split for track in retained)),
        "label_mapping": LABEL_MAPPING, "excluded_duplicates": excluded,
        "required_stem_repairs": repairs, "audio_duplicate_audit": str(audit_path),
        "verified_legacy_stems": list(verified_legacy_stems),
        "filename_overrides": filename_overrides or {},
        "audio_duplicate_audit_sha256": file_checksum(audit_path),
        "csv_files": {source_split: {"path": str(metadata_root / source_split / "tracklist.csv"),
            "sha256": file_checksum(metadata_root / source_split / "tracklist.csv")} for source_split in SPLITS},
        "track_identity": "Provided split plus full relative audio path; filename and SHA256 suffix.",
        "audit_limit": "Exact file/model-input duplicates checked; re-encodings, near duplicates, and artist overlap are not comprehensively audited.",
        "audio_modified": False}
    if audit_only:
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in records))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(report_path, report)
    write_json(metadata_path, {"dataset": "ISMIR04", "label_mapping": LABEL_MAPPING,
        "sources": list(SOURCES), "split_names": list(SPLITS.values()),
        "records": [{**asdict(track), "original_path": str(track.original_path)} for track in retained]})
    return report
