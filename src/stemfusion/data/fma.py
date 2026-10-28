"""Build aligned FMA-Medium manifests for mixtures and separated stems."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

FMA_MEDIUM_SUBSETS = frozenset({"small", "medium"})
FMA_SPLITS = {
    "training": "train",
    "validation": "validation",
    "test": "test",
}
HTDEMUCS_SOURCES = frozenset(
    {"bass", "drums", "guitar", "piano", "vocals", "other"}
)
HTDEMUCS_MODEL_NAME = "htdemucs_6s"


class FMAManifestError(ValueError):
    """Raised when an FMA manifest cannot be constructed safely."""


@dataclass(frozen=True)
class FMAManifestConfig:
    """Settings for one aligned FMA-Medium manifest."""

    tracks_csv: Path
    output_path: Path
    mixture_root: Path | None = None
    htdemucs_roots: tuple[Path, ...] = ()
    htdemucs_model_name: str = HTDEMUCS_MODEL_NAME
    track_ids: tuple[str, ...] = ()
    mixture_extension: str = ".mp3"
    allow_missing: bool = False
    overwrite: bool = False
    dry_run: bool = False


@dataclass(frozen=True)
class FMAManifestRecord:
    """One labeled FMA track and its aligned audio paths."""

    track_id: str
    split: str
    genre: str
    label: int
    paths: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FMAManifestSummary:
    """Counts and diagnostics from manifest construction."""

    metadata_tracks: int
    written_tracks: int
    dropped_tracks: int
    split_counts: dict[str, int]
    label_mapping: dict[str, int]
    missing_by_source: dict[str, int] = field(default_factory=dict)
    missing_examples: list[dict[str, Any]] = field(default_factory=list)
    dry_run: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _MetadataTrack:
    track_id: str
    split: str
    genre: str


def _normalize_extension(extension: str) -> str:
    return extension.lower() if extension.startswith(".") else f".{extension.lower()}"


def _column_index(
    groups: Sequence[str],
    fields: Sequence[str],
    group: str,
    field_name: str,
) -> int:
    for index, pair in enumerate(zip(groups, fields, strict=True)):
        if pair == (group, field_name):
            return index
    raise FMAManifestError(
        f"FMA tracks.csv is missing the ({group}, {field_name}) column."
    )


def read_fma_medium_metadata(tracks_csv: Path) -> list[_MetadataTrack]:
    """Read labeled FMA-Medium rows without requiring pandas."""

    try:
        handle = tracks_csv.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise FMAManifestError(
            f"Could not open FMA metadata: {tracks_csv}: {error}"
        ) from error

    with handle:
        reader = csv.reader(handle)
        try:
            groups = next(reader)
            fields = next(reader)
            index_row = next(reader)
        except StopIteration as error:
            raise FMAManifestError(
                "FMA tracks.csv must contain its three-row header."
            ) from error

        if len(groups) != len(fields):
            raise FMAManifestError(
                "FMA tracks.csv has inconsistent header widths."
            )
        if not index_row or index_row[0] != "track_id":
            raise FMAManifestError(
                "FMA tracks.csv does not contain the expected track_id header."
            )

        split_index = _column_index(groups, fields, "set", "split")
        subset_index = _column_index(groups, fields, "set", "subset")
        genre_index = _column_index(groups, fields, "track", "genre_top")
        required_width = max(split_index, subset_index, genre_index) + 1

        tracks: list[_MetadataTrack] = []
        seen_ids: set[str] = set()
        for line_number, row in enumerate(reader, start=4):
            if not row:
                continue
            if len(row) < required_width:
                raise FMAManifestError(
                    f"FMA metadata row {line_number} has too few columns."
                )
            subset = row[subset_index].strip()
            genre = row[genre_index].strip()
            legacy_split = row[split_index].strip()
            if subset not in FMA_MEDIUM_SUBSETS or not genre:
                continue
            if legacy_split not in FMA_SPLITS:
                raise FMAManifestError(
                    f"Unknown FMA split {legacy_split!r} on row {line_number}."
                )
            try:
                track_id = f"{int(row[0]):06d}"
            except ValueError as error:
                raise FMAManifestError(
                    f"Invalid FMA track ID on row {line_number}: {row[0]!r}."
                ) from error
            if track_id in seen_ids:
                raise FMAManifestError(
                    f"Duplicate FMA track ID in metadata: {track_id}."
                )
            seen_ids.add(track_id)
            tracks.append(
                _MetadataTrack(
                    track_id=track_id,
                    split=FMA_SPLITS[legacy_split],
                    genre=genre,
                )
            )

    if not tracks:
        raise FMAManifestError(
            "No labeled FMA-Medium tracks were found in the metadata."
        )
    return sorted(tracks, key=lambda track: int(track.track_id))


def _validate_config(config: FMAManifestConfig) -> None:
    if not config.tracks_csv.is_file():
        raise FMAManifestError(
            f"FMA tracks.csv does not exist: {config.tracks_csv}"
        )
    if any(
        len(track_id) != 6 or not track_id.isdigit()
        for track_id in config.track_ids
    ):
        raise FMAManifestError(
            "Selected FMA track IDs must be zero-padded six-digit strings."
        )
    if len(config.track_ids) != len(set(config.track_ids)):
        raise FMAManifestError(
            "Each selected FMA track ID may be provided only once."
        )
    if (
        config.mixture_root is None
        and not config.htdemucs_roots
    ):
        raise FMAManifestError(
            "At least one mixture or HT-Demucs root is required."
        )
    if config.mixture_root is not None and not config.mixture_root.is_dir():
        raise FMAManifestError(
            f"Mixture root is not a directory: {config.mixture_root}"
        )
    if config.htdemucs_model_name != HTDEMUCS_MODEL_NAME:
        raise FMAManifestError(
            "The FMA feature workflow currently supports only htdemucs_6s."
        )
    for root in config.htdemucs_roots:
        if not root.is_dir():
            raise FMAManifestError(
                f"HT-Demucs output root is not a directory: {root}"
            )
    if (
        config.output_path.exists()
        and not config.overwrite
        and not config.dry_run
    ):
        raise FMAManifestError(
            f"Manifest already exists; use --overwrite: {config.output_path}"
        )


def _audio_paths(
    config: FMAManifestConfig,
    track_id: str,
) -> dict[str, Path]:
    folder = track_id[:3]
    paths: dict[str, Path] = {}
    if config.mixture_root is not None:
        paths["mixture"] = (
            config.mixture_root
            / folder
            / f"{track_id}{_normalize_extension(config.mixture_extension)}"
        )
    if config.htdemucs_roots:
        candidates: list[dict[str, Path]] = []
        for root in config.htdemucs_roots:
            track_root = root / config.htdemucs_model_name / track_id
            stem_paths = {
                source: track_root / f"{source}.mp3"
                for source in sorted(HTDEMUCS_SOURCES)
            }
            candidates.append(stem_paths)
            if all(
                path.is_file() and path.stat().st_size > 0
                for path in stem_paths.values()
            ):
                paths.update(stem_paths)
                break
        else:
            paths.update(candidates[0])
    return paths


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _sidecar_path(manifest_path: Path, suffix: str) -> Path:
    stem = manifest_path.name.removesuffix(manifest_path.suffix)
    return manifest_path.with_name(f"{stem}.{suffix}.json")


def build_fma_manifest(
    config: FMAManifestConfig,
) -> tuple[list[FMAManifestRecord], FMAManifestSummary]:
    """Build and optionally write one aligned FMA-Medium manifest."""

    _validate_config(config)
    metadata = read_fma_medium_metadata(config.tracks_csv)
    genres = sorted({track.genre for track in metadata})
    label_mapping = {genre: index for index, genre in enumerate(genres)}
    if config.track_ids:
        metadata_by_id = {track.track_id: track for track in metadata}
        unknown_ids = set(config.track_ids) - set(metadata_by_id)
        if unknown_ids:
            names = ", ".join(sorted(unknown_ids))
            raise FMAManifestError(
                f"Selected IDs are not labeled FMA-Medium tracks: {names}."
            )
        metadata = [
            metadata_by_id[track_id] for track_id in config.track_ids
        ]

    records: list[FMAManifestRecord] = []
    missing_counts: Counter[str] = Counter()
    missing_examples: list[dict[str, Any]] = []
    for track in metadata:
        candidate_paths = _audio_paths(config, track.track_id)
        missing = [
            name
            for name, path in candidate_paths.items()
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing:
            missing_counts.update(missing)
            if len(missing_examples) < 25:
                missing_examples.append(
                    {"track_id": track.track_id, "sources": missing}
                )
            if config.allow_missing:
                continue
        records.append(
            FMAManifestRecord(
                track_id=track.track_id,
                split=track.split,
                genre=track.genre,
                label=label_mapping[track.genre],
                paths={
                    name: str(path.resolve())
                    for name, path in sorted(candidate_paths.items())
                },
            )
        )

    if missing_counts and not config.allow_missing:
        details = ", ".join(
            f"{source}={count}"
            for source, count in sorted(missing_counts.items())
        )
        raise FMAManifestError(
            "Configured audio is incomplete. Use --allow-missing to create "
            f"the aligned intersection. Missing counts: {details}."
        )
    if not records:
        raise FMAManifestError(
            "No complete FMA-Medium tracks remain after path validation."
        )

    split_counts = Counter(record.split for record in records)
    summary = FMAManifestSummary(
        metadata_tracks=len(metadata),
        written_tracks=len(records),
        dropped_tracks=len(metadata) - len(records),
        split_counts={
            split: split_counts.get(split, 0)
            for split in ("train", "validation", "test")
        },
        label_mapping=label_mapping,
        missing_by_source=dict(sorted(missing_counts.items())),
        missing_examples=missing_examples,
        dry_run=config.dry_run,
    )
    if config.dry_run:
        return records, summary

    manifest_text = "".join(
        json.dumps(record.to_dict(), sort_keys=True) + "\n"
        for record in records
    )
    _atomic_write_text(config.output_path, manifest_text)
    metadata_path = _sidecar_path(config.output_path, "metadata")
    report_path = _sidecar_path(config.output_path, "report")
    _atomic_write_text(
        metadata_path,
        json.dumps(
            {
                "dataset": "FMA-Medium",
                "label_mapping": label_mapping,
                "sources": sorted(records[0].paths),
                "split_names": ["train", "validation", "test"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    _atomic_write_text(
        report_path,
        json.dumps(summary.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    return records, summary


def read_manifest(path: Path) -> list[FMAManifestRecord]:
    """Read and validate a manifest produced by this module."""

    if not path.is_file():
        raise FMAManifestError(f"Manifest does not exist: {path}")
    records: list[FMAManifestRecord] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                record = FMAManifestRecord(
                    track_id=str(data["track_id"]),
                    split=str(data["split"]),
                    genre=str(data["genre"]),
                    label=int(data["label"]),
                    paths={
                        str(name): str(value)
                        for name, value in data["paths"].items()
                    },
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise FMAManifestError(
                    f"Invalid manifest record on line {line_number}: {error}"
                ) from error
            if record.track_id in seen:
                raise FMAManifestError(
                    f"Duplicate manifest track ID: {record.track_id}."
                )
            if record.split not in {"train", "validation", "test"}:
                raise FMAManifestError(
                    f"Invalid split for track {record.track_id}: {record.split}."
                )
            if not record.paths:
                raise FMAManifestError(
                    f"Track {record.track_id} has no configured audio paths."
                )
            seen.add(record.track_id)
            records.append(record)
    if not records:
        raise FMAManifestError(f"Manifest is empty: {path}")
    return records


def _parse_track_id(value: str) -> str:
    if not value.isdigit():
        raise argparse.ArgumentTypeError(
            "FMA track IDs must contain digits only."
        )
    number = int(value)
    if number > 999_999:
        raise argparse.ArgumentTypeError("FMA track IDs must fit six digits.")
    return f"{number:06d}"


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Build an aligned FMA-Medium manifest for mixtures and separated "
        "stems."
    )
    parser.add_argument("--tracks-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mixture-root", type=Path)
    parser.add_argument("--mixture-extension", default=".mp3")
    parser.add_argument(
        "--htdemucs-root",
        action="append",
        default=[],
        type=Path,
        help=(
            "Demucs output root containing htdemucs_6s; repeat when completed "
            "tracks are distributed across storage devices."
        ),
    )
    parser.add_argument(
        "--track-id",
        action="append",
        default=[],
        type=_parse_track_id,
        metavar="TRACK_ID",
        help="Restrict the manifest to this track; may be repeated.",
    )
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args: argparse.Namespace) -> FMAManifestConfig:
    mixture_root = (
        args.mixture_root.expanduser().resolve()
        if args.mixture_root is not None
        else None
    )
    return FMAManifestConfig(
        tracks_csv=args.tracks_csv.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        mixture_root=mixture_root,
        htdemucs_roots=tuple(
            path.expanduser().resolve() for path in args.htdemucs_root
        ),
        track_ids=tuple(args.track_id),
        mixture_extension=args.mixture_extension,
        allow_missing=args.allow_missing,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        config = config_from_args(args)
        _, summary = build_fma_manifest(config)
    except FMAManifestError as error:
        print(f"FMA manifest error: {error}")
        return 2
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0
