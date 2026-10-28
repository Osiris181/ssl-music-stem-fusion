"""Extract HT-Demucs mean RMS energy from sources in an aligned FMA manifest."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import numbers
import pickle
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from stemfusion.data.fma import (
    FMAManifestError,
    FMAManifestRecord,
    HTDEMUCS_SOURCES,
    read_manifest,
)

LOGGER = logging.getLogger(__name__)
RMS_SOURCES = frozenset({"mixture", *HTDEMUCS_SOURCES})
DEFAULT_SPLITS = ("train", "validation", "test")


class RMSExtractionError(ValueError):
    """Raised when RMS extraction cannot be configured or completed."""


@dataclass(frozen=True)
class RMSExtractionConfig:
    """Configuration for paper-aligned HT-Demucs per-source RMS extraction."""

    manifest_path: Path
    output_dir: Path
    source: str
    center: bool = True
    pad_mode: str = "constant"
    sample_rate: int = 24_000
    frame_length: int = 2_048
    hop_length: int = 512
    scale: float = 20.0
    max_seconds: float | None = None
    splits: tuple[str, ...] = DEFAULT_SPLITS
    limit_per_split: int | None = None
    overwrite: bool = False
    continue_on_error: bool = True
    dry_run: bool = False
    report_path: Path | None = None


@dataclass(frozen=True)
class RMSExtractionFailure:
    track_id: str
    split: str
    audio_path: str
    reason: str


@dataclass
class RMSSplitSummary:
    split: str
    selected: int
    succeeded: int = 0
    failed: int = 0
    skipped_existing_cache: bool = False
    output_path: str | None = None
    partial_output_path: str | None = None


@dataclass
class RMSExtractionSummary:
    source: str
    manifest_records: int
    manifest_path: str
    sample_rate: int
    frame_length: int
    hop_length: int
    center: bool
    pad_mode: str
    scale: float
    max_seconds: float | None
    planned: int = 0
    succeeded: int = 0
    failed: int = 0
    split_summaries: list[RMSSplitSummary] = field(default_factory=list)
    failures: list[RMSExtractionFailure] = field(default_factory=list)
    dry_run: bool = False

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exit_code"] = self.exit_code
        return data


def _validate_config(config: RMSExtractionConfig) -> None:
    if not config.manifest_path.is_file():
        raise RMSExtractionError(
            f"Manifest does not exist: {config.manifest_path}"
        )
    if config.source not in RMS_SOURCES:
        allowed = ", ".join(sorted(RMS_SOURCES))
        raise RMSExtractionError(
            f"Unsupported source {config.source!r}; choose {allowed}."
        )
    if config.sample_rate <= 0:
        raise RMSExtractionError("Sample rate must be positive.")
    if config.frame_length <= 0:
        raise RMSExtractionError("RMS frame length must be positive.")
    if not config.center:
        raise RMSExtractionError(
            "The paper reproduction requires centered RMS frames."
        )
    if config.pad_mode != "constant":
        raise RMSExtractionError("RMS padding mode must be 'constant'.")
    if config.hop_length <= 0:
        raise RMSExtractionError("RMS hop length must be positive.")
    if config.scale <= 0:
        raise RMSExtractionError("RMS scaling factor must be positive.")
    if config.max_seconds is not None and config.max_seconds <= 0:
        raise RMSExtractionError(
            "Maximum duration must be positive when provided."
        )
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise RMSExtractionError(
            "Per-split limit must be positive when provided."
        )
    if not config.splits:
        raise RMSExtractionError("At least one split must be selected.")
    unknown_splits = set(config.splits) - set(DEFAULT_SPLITS)
    if unknown_splits:
        names = ", ".join(sorted(unknown_splits))
        raise RMSExtractionError(f"Unknown manifest splits: {names}.")
    if len(config.splits) != len(set(config.splits)):
        raise RMSExtractionError("Each split may be selected only once.")


def _load_module(name: str, installation_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as error:
        raise RMSExtractionError(
            f"Could not import {name}: {error}. {installation_hint}"
        ) from error


class _RMSRuntime:
    """Lazy audio and RMS runtime so unit tests need no audio dependencies."""

    def __init__(self, config: RMSExtractionConfig) -> None:
        self.config = config
        hint = 'Install the project with the "musicfm" optional dependencies.'
        self.numpy = _load_module("numpy", hint)
        self.librosa = _load_module("librosa", hint)
        try:
            self.torch = importlib.import_module("torch")
            self.torchaudio = importlib.import_module("torchaudio")
        except Exception:
            self.torch = None
            self.torchaudio = None

    def _load_audio(self, path: Path) -> Any:
        errors: list[str] = []
        if self.torchaudio is not None and self.torch is not None:
            try:
                waveform, sample_rate = self.torchaudio.load(str(path))
                if waveform.ndim > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sample_rate != self.config.sample_rate:
                    waveform = self.torchaudio.functional.resample(
                        waveform,
                        sample_rate,
                        self.config.sample_rate,
                    )
                return (
                    waveform.squeeze(0)
                    .to(self.torch.float32)
                    .cpu()
                    .numpy()
                )
            except Exception as error:
                errors.append(f"torchaudio: {error}")

        try:
            waveform, _ = self.librosa.load(
                str(path),
                sr=self.config.sample_rate,
                mono=True,
            )
            return self.numpy.asarray(waveform, dtype=self.numpy.float32)
        except Exception as error:
            errors.append(f"librosa: {error}")
        raise RMSExtractionError(
            f"Could not load audio {path}: {'; '.join(errors)}"
        )

    def __call__(self, path: Path) -> Any:
        waveform = self._load_audio(path)
        if self.config.max_seconds is not None:
            maximum_samples = int(
                self.config.sample_rate * self.config.max_seconds
            )
            waveform = waveform[:maximum_samples]
        if waveform.size == 0:
            raise RMSExtractionError(f"Audio is empty: {path}")

        rms_frames = self.librosa.feature.rms(
            y=waveform,
            frame_length=self.config.frame_length,
            hop_length=self.config.hop_length,
            center=self.config.center,
            pad_mode=self.config.pad_mode,
        )[0]
        scaled_rms = float(self.numpy.mean(rms_frames)) * self.config.scale
        if not self.numpy.isfinite(scaled_rms):
            raise RMSExtractionError(f"RMS is not finite: {path}")
        return self.numpy.asarray([scaled_rms], dtype=self.numpy.float32)


def _feature_size(feature: Any) -> int | None:
    if isinstance(feature, numbers.Real):
        return 1
    shape = getattr(feature, "shape", None)
    if shape is not None:
        dimensions = tuple(int(size) for size in shape)
        if dimensions == ():
            return 1
        if len(dimensions) == 1:
            return dimensions[0]
        return None
    if isinstance(feature, (list, tuple)):
        return len(feature)
    return None


def _cache_path(config: RMSExtractionConfig, split: str) -> Path:
    return config.output_dir / f"fma_rms_{config.source}_{split}.pkl"


def _partial_cache_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _atomic_write_pickle(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _records_by_split(
    records: list[FMAManifestRecord],
    config: RMSExtractionConfig,
) -> dict[str, list[FMAManifestRecord]]:
    grouped = {split: [] for split in config.splits}
    for record in records:
        if record.split in grouped:
            grouped[record.split].append(record)
    for split in config.splits:
        if config.limit_per_split is not None:
            grouped[split] = grouped[split][: config.limit_per_split]
        if not grouped[split]:
            raise RMSExtractionError(
                f"Manifest has no records for selected split: {split}."
            )
    return grouped


def extract_rms(
    config: RMSExtractionConfig,
    feature_extractor: Callable[[Path], Any] | None = None,
) -> RMSExtractionSummary:
    """Extract one RMS value per track and source for selected splits."""

    _validate_config(config)
    try:
        records = read_manifest(config.manifest_path)
    except FMAManifestError as error:
        raise RMSExtractionError(str(error)) from error

    grouped = _records_by_split(records, config)
    summary = RMSExtractionSummary(
        source=config.source,
        manifest_records=len(records),
        manifest_path=str(config.manifest_path),
        sample_rate=config.sample_rate,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
        center=config.center,
        pad_mode=config.pad_mode,
        scale=config.scale,
        max_seconds=config.max_seconds,
        dry_run=config.dry_run,
    )
    pending: list[tuple[str, list[FMAManifestRecord], Path]] = []
    for split in config.splits:
        split_records = grouped[split]
        output_path = _cache_path(config, split)
        split_summary = RMSSplitSummary(
            split=split,
            selected=len(split_records),
            output_path=str(output_path),
        )
        summary.split_summaries.append(split_summary)
        if output_path.is_file() and output_path.stat().st_size > 0:
            if not config.overwrite:
                split_summary.skipped_existing_cache = True
                continue
        summary.planned += len(split_records)
        pending.append((split, split_records, output_path))

    if config.dry_run or not pending:
        return summary

    extractor = feature_extractor or _RMSRuntime(config)
    for split, split_records, output_path in pending:
        split_summary = next(
            item for item in summary.split_summaries if item.split == split
        )
        features: dict[str, Any] = {}
        split_failures: list[RMSExtractionFailure] = []
        for index, record in enumerate(split_records, start=1):
            source_path = record.paths.get(config.source)
            if source_path is None:
                reason = (
                    f"Manifest does not contain source {config.source!r}."
                )
                audio_path = ""
            else:
                audio_path = source_path
                path = Path(source_path)
                if not path.is_file() or path.stat().st_size == 0:
                    reason = "Audio file is missing or empty."
                else:
                    try:
                        feature = extractor(path)
                        size = _feature_size(feature)
                        if size != 1:
                            raise RMSExtractionError(
                                f"Expected one RMS value, received {size}."
                            )
                    except Exception as error:
                        reason = str(error)
                    else:
                        features[record.track_id] = feature
                        split_summary.succeeded += 1
                        summary.succeeded += 1
                        if index % 100 == 0:
                            LOGGER.info(
                                "Extracted %d/%d %s RMS values.",
                                index,
                                len(split_records),
                                split,
                            )
                        continue

            failure = RMSExtractionFailure(
                track_id=record.track_id,
                split=split,
                audio_path=audio_path,
                reason=reason,
            )
            split_failures.append(failure)
            summary.failures.append(failure)
            split_summary.failed += 1
            summary.failed += 1
            LOGGER.error(
                "RMS extraction failed for %s: %s",
                record.track_id,
                reason,
            )
            if not config.continue_on_error:
                break

        if split_failures:
            if features:
                partial_path = _partial_cache_path(output_path)
                _atomic_write_pickle(partial_path, features)
                split_summary.partial_output_path = str(partial_path)
            if not config.continue_on_error:
                break
            continue

        _atomic_write_pickle(output_path, features)
        _partial_cache_path(output_path).unlink(missing_ok=True)

    report_path = config.report_path or (
        config.output_dir / f"extraction_report_rms_{config.source}.json"
    )
    _atomic_write_json(report_path, summary.to_dict())
    return summary


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Extract full-duration, mean HT-Demucs RMS energy from an aligned manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        choices=tuple(sorted(RMS_SOURCES)),
        required=True,
    )
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--frame-length", type=int, default=2_048)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("--scale", type=float, default=20.0)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--split",
        action="append",
        choices=DEFAULT_SPLITS,
        dest="splits",
    )
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.set_defaults(command_handler=run_from_args)


def config_from_args(args: argparse.Namespace) -> RMSExtractionConfig:
    return RMSExtractionConfig(
        manifest_path=args.manifest.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        source=args.source,
        sample_rate=args.sample_rate,
        frame_length=args.frame_length,
        hop_length=args.hop_length,
        scale=args.scale,
        max_seconds=args.max_seconds,
        splits=tuple(args.splits or DEFAULT_SPLITS),
        limit_per_split=args.limit_per_split,
        overwrite=args.overwrite,
        continue_on_error=not args.fail_fast,
        dry_run=args.dry_run,
        report_path=(
            args.report.expanduser().resolve()
            if args.report is not None
            else None
        ),
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        config = config_from_args(args)
        summary = extract_rms(config)
    except RMSExtractionError as error:
        LOGGER.error("%s", error)
        return 2
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return summary.exit_code
