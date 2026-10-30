"""Shared MusicFM implementation used by dataset-specific extraction tasks."""

from __future__ import annotations

import importlib
import json
import logging
import pickle
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stemfusion.configuration import configuration_dict

from stemfusion.data.fma import (
    HTDEMUCS_SOURCES,
    FMAManifestError,
    FMAManifestRecord,
    read_manifest,
)

LOGGER = logging.getLogger(__name__)
MUSICFM_SOURCES = frozenset({"mixture", *HTDEMUCS_SOURCES})
DEFAULT_SPLITS = ("train", "validation", "test")


class MusicFMExtractionError(ValueError):
    """Raised when MusicFM extraction cannot be configured or completed."""


@dataclass(frozen=True)
class MusicFMExtractionConfig:
    """Configuration for the paper-aligned MusicFM extraction stage."""

    manifest_path: Path
    output_dir: Path
    source: str
    musicfm_root: Path
    stats_path: Path | None = None
    checkpoint_path: Path | None = None
    device: str = "cuda"
    sample_rate: int = 24_000
    layer_index: int = 7
    expected_dimension: int = 1_024
    temporal_pooling: str = "mean"
    is_flash: bool = False
    max_seconds: float | None = None
    splits: tuple[str, ...] = DEFAULT_SPLITS
    limit_per_split: int | None = None
    overwrite: bool = False
    continue_on_error: bool = True
    dry_run: bool = False
    report_path: Path | None = None
    config_provenance: dict[str, Any] = field(default_factory=dict)
    dataset: str = "fma_medium"

    @property
    def effective_stats_path(self) -> Path:
        return self.stats_path or self.musicfm_root / "data" / "msd_stats.json"

    @property
    def effective_checkpoint_path(self) -> Path:
        return (
            self.checkpoint_path
            or self.musicfm_root / "data" / "pretrained_msd.pt"
        )


@dataclass(frozen=True)
class MusicFMExtractionFailure:
    track_id: str
    split: str
    audio_path: str
    reason: str


@dataclass
class MusicFMSplitSummary:
    split: str
    selected: int
    succeeded: int = 0
    failed: int = 0
    skipped_existing_cache: bool = False
    output_path: str | None = None
    partial_output_path: str | None = None


@dataclass
class MusicFMExtractionSummary:
    source: str
    manifest_records: int
    manifest_path: str
    checkpoint_path: str
    stats_path: str
    device: str
    sample_rate: int
    layer_index: int
    temporal_pooling: str
    expected_dimension: int
    max_seconds: float | None
    planned: int = 0
    succeeded: int = 0
    failed: int = 0
    split_summaries: list[MusicFMSplitSummary] = field(default_factory=list)
    failures: list[MusicFMExtractionFailure] = field(default_factory=list)
    dry_run: bool = False
    resolved_config: dict[str, Any] = field(default_factory=dict)

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["exit_code"] = self.exit_code
        return data


def _validate_config(config: MusicFMExtractionConfig) -> None:
    if config.dataset not in {"fma_medium", "gtzan"}:
        raise MusicFMExtractionError(f"Unsupported extraction dataset: {config.dataset}")
    if not config.manifest_path.is_file():
        raise MusicFMExtractionError(
            f"Manifest does not exist: {config.manifest_path}"
        )
    if config.source not in MUSICFM_SOURCES:
        allowed = ", ".join(sorted(MUSICFM_SOURCES))
        raise MusicFMExtractionError(
            f"Unsupported source {config.source!r}; choose {allowed}."
        )
    if not config.musicfm_root.is_dir():
        raise MusicFMExtractionError(
            f"MusicFM repository does not exist: {config.musicfm_root}"
        )
    if not config.effective_stats_path.is_file():
        raise MusicFMExtractionError(
            f"MusicFM statistics file does not exist: "
            f"{config.effective_stats_path}"
        )
    if not config.effective_checkpoint_path.is_file():
        raise MusicFMExtractionError(
            f"MusicFM checkpoint does not exist: "
            f"{config.effective_checkpoint_path}"
        )
    if config.sample_rate <= 0:
        raise MusicFMExtractionError("Sample rate must be positive.")
    if not 0 <= config.layer_index <= 12:
        raise MusicFMExtractionError(
            "MusicFM layer index must be between 0 and 12."
        )
    if config.expected_dimension <= 0:
        raise MusicFMExtractionError(
            "Expected feature dimension must be positive."
        )
    if config.temporal_pooling != "mean":
        raise MusicFMExtractionError(
            "The paper reproduction supports only temporal mean pooling."
        )
    if config.max_seconds is not None and config.max_seconds <= 0:
        raise MusicFMExtractionError(
            "Maximum duration must be positive when provided."
        )
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise MusicFMExtractionError(
            "Per-split limit must be positive when provided."
        )
    if not config.splits:
        raise MusicFMExtractionError("At least one split must be selected.")
    unknown_splits = set(config.splits) - set(DEFAULT_SPLITS)
    if unknown_splits:
        names = ", ".join(sorted(unknown_splits))
        raise MusicFMExtractionError(f"Unknown manifest splits: {names}.")
    if len(config.splits) != len(set(config.splits)):
        raise MusicFMExtractionError("Each split may be selected only once.")


def _load_module(name: str, installation_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception as error:
        raise MusicFMExtractionError(
            f"Could not import {name}: {error}. {installation_hint}"
        ) from error


class _MusicFMRuntime:
    """Lazy runtime wrapper so manifest tests do not require ML packages."""

    def __init__(self, config: MusicFMExtractionConfig) -> None:
        self.config = config
        hint = 'Install the project with the "musicfm" optional dependencies.'
        self.torch = _load_module("torch", hint)
        self.numpy = _load_module("numpy", hint)
        try:
            self.torchaudio = importlib.import_module("torchaudio")
        except Exception:
            self.torchaudio = None

        if config.device.startswith("cuda") and not self.torch.cuda.is_available():
            raise MusicFMExtractionError(
                f"CUDA was requested but is unavailable: {config.device}."
            )

        parent = str(config.musicfm_root.parent)
        if parent not in sys.path:
            sys.path.insert(0, parent)
        try:
            module = importlib.import_module("musicfm.model.musicfm_25hz")
            model_class = module.MusicFM25Hz
        except Exception as error:
            raise MusicFMExtractionError(
                "Could not import MusicFM25Hz from the configured MusicFM "
                f"repository: {error}"
            ) from error

        self.model = model_class(
            is_flash=config.is_flash,
            stat_path=str(config.effective_stats_path),
            model_path=str(config.effective_checkpoint_path),
        )
        self.model = self.model.to(config.device)
        self.model.eval()

    def _load_audio(self, path: Path) -> Any:
        errors: list[str] = []
        if self.torchaudio is not None:
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
                return waveform.squeeze(0).to(self.torch.float32)
            except Exception as error:
                errors.append(f"torchaudio: {error}")

        try:
            librosa = importlib.import_module("librosa")
            waveform, _ = librosa.load(
                str(path),
                sr=self.config.sample_rate,
                mono=True,
            )
            return self.torch.as_tensor(waveform, dtype=self.torch.float32)
        except Exception as error:
            errors.append(f"librosa: {error}")
        raise MusicFMExtractionError(
            f"Could not load audio {path}: {'; '.join(errors)}"
        )

    def __call__(self, path: Path) -> Any:
        waveform = self._load_audio(path)
        if self.config.max_seconds is not None:
            maximum_samples = int(
                self.config.sample_rate * self.config.max_seconds
            )
            waveform = waveform[:maximum_samples]
        if waveform.numel() == 0:
            raise MusicFMExtractionError(f"Audio is empty: {path}")

        batched = waveform.unsqueeze(0).to(self.config.device)
        with self.torch.inference_mode():
            embedding = self.model.get_latent(
                batched,
                layer_ix=self.config.layer_index,
            )
            feature = embedding.mean(dim=1).squeeze(0)
        feature = feature.detach().to(dtype=self.torch.float32).cpu()
        shape = tuple(int(size) for size in feature.shape)
        if shape != (self.config.expected_dimension,):
            raise MusicFMExtractionError(
                f"Expected a ({self.config.expected_dimension},) feature for "
                f"{path}, received {shape}."
            )
        return feature


def _feature_dimension(feature: Any) -> int | None:
    shape = getattr(feature, "shape", None)
    if shape is not None:
        dimensions = tuple(int(size) for size in shape)
        if len(dimensions) == 1:
            return dimensions[0]
        return None
    if isinstance(feature, (list, tuple)):
        return len(feature)
    return None


def _cache_path(config: MusicFMExtractionConfig, split: str) -> Path:
    prefix = "fma" if config.dataset == "fma_medium" else "gtzan"
    source = "mix" if config.dataset == "gtzan" and config.source == "mixture" else config.source
    return config.output_dir / f"{prefix}_musicfm_{source}_{split}.pkl"


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
    config: MusicFMExtractionConfig,
) -> dict[str, list[FMAManifestRecord]]:
    grouped = {split: [] for split in config.splits}
    for record in records:
        if record.split in grouped:
            grouped[record.split].append(record)
    for split in config.splits:
        if config.limit_per_split is not None:
            grouped[split] = grouped[split][: config.limit_per_split]
        if not grouped[split]:
            raise MusicFMExtractionError(
                f"Manifest has no records for selected split: {split}."
            )
    return grouped


def extract_musicfm(
    config: MusicFMExtractionConfig,
    feature_extractor: Callable[[Path], Any] | None = None,
) -> MusicFMExtractionSummary:
    """Extract one source for the selected splits and write atomic caches."""

    _validate_config(config)
    try:
        records = read_manifest(config.manifest_path)
    except FMAManifestError as error:
        raise MusicFMExtractionError(str(error)) from error

    grouped = _records_by_split(records, config)
    summary = MusicFMExtractionSummary(
        source=config.source,
        manifest_records=len(records),
        manifest_path=str(config.manifest_path),
        checkpoint_path=str(config.effective_checkpoint_path),
        stats_path=str(config.effective_stats_path),
        device=config.device,
        sample_rate=config.sample_rate,
        layer_index=config.layer_index,
        temporal_pooling=config.temporal_pooling,
        expected_dimension=config.expected_dimension,
        max_seconds=config.max_seconds,
        dry_run=config.dry_run,
        resolved_config=configuration_dict(config),
    )
    pending: list[tuple[str, list[FMAManifestRecord], Path]] = []
    for split in config.splits:
        split_records = grouped[split]
        output_path = _cache_path(config, split)
        split_summary = MusicFMSplitSummary(
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

    extractor = feature_extractor or _MusicFMRuntime(config)
    for split, split_records, output_path in pending:
        split_summary = next(
            item for item in summary.split_summaries if item.split == split
        )
        features: dict[str, Any] = {}
        split_failures: list[MusicFMExtractionFailure] = []
        with logging_redirect_tqdm(loggers=[LOGGER]), tqdm(
            total=len(split_records),
            desc=f"{config.source} / {split}",
            unit="track",
            dynamic_ncols=True,
        ) as progress:
            for record in split_records:
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
                            dimension = _feature_dimension(feature)
                            if dimension != config.expected_dimension:
                                raise MusicFMExtractionError(
                                    f"Expected {config.expected_dimension} values, "
                                    f"received {dimension}."
                                )
                        except Exception as error:
                            reason = str(error)
                        else:
                            features[record.track_id] = feature
                            split_summary.succeeded += 1
                            summary.succeeded += 1
                            progress.update(1)
                            continue

                failure = MusicFMExtractionFailure(
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
                    "MusicFM extraction failed for %s: %s",
                    record.track_id,
                    reason,
                )
                progress.update(1)
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
        config.output_dir
        / f"extraction_report_musicfm_{config.source}.json"
    )
    _atomic_write_json(report_path, summary.to_dict())
    return summary


def extract_musicfm_sources(
    configs: Sequence[MusicFMExtractionConfig],
    feature_extractor: Callable[[Path], Any] | None = None,
) -> list[MusicFMExtractionSummary]:
    """Extract sources sequentially with one shared encoder and audio policy.

    Dry runs and already-complete caches never initialize the model. Stop
    after a failed source, retaining its partial cache and individual report.
    """

    if not configs:
        raise MusicFMExtractionError("At least one extraction source is required.")
    if len({config.source for config in configs}) != len(configs):
        raise MusicFMExtractionError("Each extraction source must be selected only once.")

    def shared_settings(config: MusicFMExtractionConfig) -> tuple[Any, ...]:
        return (
            config.manifest_path, config.musicfm_root,
            config.effective_checkpoint_path, config.effective_stats_path,
            config.device, config.sample_rate, config.layer_index,
            config.expected_dimension, config.temporal_pooling, config.is_flash,
            config.max_seconds, config.splits, config.limit_per_split,
            config.output_dir, config.dry_run,
        )

    expected = shared_settings(configs[0])
    for config in configs:
        _validate_config(config)
        if shared_settings(config) != expected:
            raise MusicFMExtractionError(
                "All sources in one job must share the manifest, encoder, "
                "audio settings, splits, and output directory."
            )

    extractor = feature_extractor
    summaries = []
    for index, config in enumerate(configs, start=1):
        LOGGER.info("Source %d/%d: %s", index, len(configs), config.source)
        if extractor is None and not config.dry_run:
            plan = extract_musicfm(replace(config, dry_run=True))
            if plan.planned:
                # Initialize outside the per-track failure handler so a model
                # initialization error is not retried for every audio file.
                extractor = _MusicFMRuntime(config)
        summary = extract_musicfm(config, feature_extractor=extractor)
        summaries.append(summary)
        if summary.exit_code:
            break
    return summaries
