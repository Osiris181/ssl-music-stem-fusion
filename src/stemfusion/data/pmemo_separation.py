"""Label-independent separation jobs from PMEmo's supplied fixed split IDs.

Keep this preparation code separate from the fingerprinted feature-manifest
reader so adding separation support does not invalidate extracted features.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from stemfusion.data.pmemo import SPLITS, _track_id
from stemfusion.separation.htdemucs import HTDEMUCS_6S_STEMS
from stemfusion.training.reporting import file_checksum, write_json


@dataclass(frozen=True)
class PMEmoSeparationManifestConfig:
    split_files: dict[str, Path]
    mixture_root: Path
    output_root: Path
    output_path: Path
    track_list: Path | None = None
    all_tracks: bool = False
    overwrite: bool = False
    dry_run: bool = False


def build_separation_manifest(config: PMEmoSeparationManifestConfig) -> dict:
    """Bind split IDs to exact stem destinations without scanning/decoding audio."""
    if set(config.split_files) != set(SPLITS):
        raise ValueError("Provide PMEmo train, validation and test split files.")
    rows, seen = [], set()
    for split in SPLITS:
        ids = config.split_files[split].read_text(encoding="utf-8-sig").split()
        if not ids:
            raise ValueError(f"Empty PMEmo split: {split}")
        for value in ids:
            track_id = _track_id(value)
            if track_id in seen:
                raise ValueError(f"Repeated PMEmo ID within/across splits: {track_id}")
            seen.add(track_id)
            rows.append({"dataset": "pmemo", "track_id": track_id, "split": split,
                         "relative_path": f"{track_id}.mp3"})
    if config.all_tracks:
        selected = rows
    else:
        if config.track_list is None:
            raise ValueError("Provide --track-list, or use --all-tracks explicitly.")
        requested = [_track_id(value) for value in config.track_list.read_text(encoding="utf-8-sig").split()]
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("Track list must contain distinct canonical PMEmo IDs.")
        by_id = {row["track_id"]: row for row in rows}
        if set(requested) - by_id.keys():
            raise ValueError("Track list contains IDs outside the supplied PMEmo splits.")
        selected = [by_id[track_id] for track_id in requested]
    mixture_root, output_root = config.mixture_root.resolve(), config.output_root.resolve()
    records = []
    for row in selected:
        target = output_root / "htdemucs_6s" / row["track_id"]
        records.append({**row, "paths": {
            "mixture": str(mixture_root / row["relative_path"]),
            **{stem: str(target / f"{stem}.mp3") for stem in HTDEMUCS_6S_STEMS},
        }})
    report = {
        "dataset": "pmemo", "purpose": "source_separation", "records": len(records),
        "input_split_counts": dict(Counter(row["split"] for row in rows)),
        "selected_split_counts": {split: sum(row["split"] == split for row in selected) for split in SPLITS},
        "split_files": {split: {"path": str(path.resolve()), "sha256": file_checksum(path)}
                        for split, path in config.split_files.items()},
        "all_tracks": config.all_tracks,
        "track_list": str(config.track_list.resolve()) if config.track_list is not None and not config.all_tracks else None,
        "track_list_sha256": file_checksum(config.track_list) if config.track_list is not None and not config.all_tracks else None,
        "mixture_root": str(mixture_root), "output_root": str(output_root),
        "audio_files_checked": False, "binary_labels_created": False, "dry_run": config.dry_run,
    }
    output = config.output_path.resolve()
    sidecar = output.with_suffix(".report.json")
    inputs = {path.resolve() for path in config.split_files.values()}
    if config.track_list is not None:
        inputs.add(config.track_list.resolve())
    if {output, sidecar} & inputs:
        raise ValueError("Separation manifest/report must not replace input split or track-list files.")
    if config.dry_run:
        return report
    if not config.overwrite and any(path.exists() for path in (output, sidecar)):
        raise ValueError(f"Manifest already exists; use --overwrite explicitly: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(sidecar, report)
    return report
