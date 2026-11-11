"""Music2Emo-style segment caches with the shared frozen MusicFM runtime."""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stemfusion.configuration import configuration_dict
from stemfusion.data.segments import segment_files
from stemfusion.features._musicfm import MusicFMExtractionConfig, _MusicFMRuntime
from stemfusion.training.reporting import (
    file_checksum, finish_artifacts, fingerprint, reusable_artifacts, write_json,
)

LOGGER = logging.getLogger(__name__)
SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals", "other")
MINIMUM_INPUT_SAMPLES = 1025  # MusicFM's 2048-point STFT uses reflection padding.


@dataclass(frozen=True)
class SegmentExtractionConfig(MusicFMExtractionConfig):
    segment_seconds: float = 30.0
    stem_alignment: str = "mixture_duration"
    max_alignment_seconds: float = 0.25
    short_segment_policy: str = "pad_to_minimum"
    minimum_input_samples: int = MINIMUM_INPUT_SAMPLES


def segment_ranges(samples: int, sample_rate: int, seconds: float) -> list[tuple[int, int]]:
    """Keep every sample in fixed-width, unpadded, non-overlapping segments."""
    if samples <= 0 or sample_rate <= 0 or not math.isfinite(seconds) or seconds * sample_rate < 1:
        raise ValueError("Nonempty audio and a positive segment duration are required.")
    width = int(seconds * sample_rate)
    return [(start, min(start + width, samples)) for start in range(0, samples, width)]


def align_stem(waveform, reference_samples: int, max_difference_samples: int):
    """Match only the end of a stem to the decoded mixture's duration."""
    import torch

    samples = waveform.numel()
    if samples <= 0 or reference_samples <= 0 or waveform.ndim != 1:
        raise ValueError("Alignment requires nonempty mono audio and a positive reference length.")
    difference = samples - reference_samples
    if abs(difference) > max_difference_samples:
        raise ValueError(f"Stem differs from mixture by {difference} samples; allowed absolute difference is {max_difference_samples}.")
    if difference > 0:
        waveform = waveform[:reference_samples]
    elif difference < 0:
        waveform = torch.nn.functional.pad(waveform, (0, -difference))
    return waveform, {"decoded_samples": samples, "aligned_samples": reference_samples,
                      "trimmed_end_samples": max(0, difference), "padded_end_samples": max(0, -difference)}


def prepare_encoder_segment(waveform, minimum_samples: int, policy: str):
    """Preserve valid segments exactly; right-pad only unsupported tiny inputs."""
    import torch

    if policy not in {"pad_to_minimum", "error"} or minimum_samples <= 0:
        raise ValueError("Unknown short-segment policy or invalid minimum input length.")
    samples = waveform.numel()
    if waveform.ndim != 1 or samples <= 0:
        raise ValueError("Encoder segments must contain nonempty mono audio.")
    padding = max(0, minimum_samples - samples)
    if padding and policy == "error":
        raise ValueError(f"Segment has {samples} samples; MusicFM requires at least {minimum_samples}. Select pad_to_minimum to retain this tail.")
    if padding:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform, padding


class _SegmentRuntime(_MusicFMRuntime):
    def __init__(self, config):
        super().__init__(config)
        frontend = self.model.preprocessor_melspec_2048.mel_stft.spectrogram
        if frontend.n_fft // 2 + 1 != config.minimum_input_samples:
            raise ValueError("Configured minimum input length does not match the MusicFM frontend.")
        self.reference_path = None
        self.reference_waveform = None
        self.last_audio_metadata = None

    def set_reference(self, path: Path) -> None:
        if path != getattr(self, "reference_path", None):
            self.reference_path = path
            self.reference_waveform = None

    def __call__(self, path: Path):
        reference_path = getattr(self, "reference_path", None)
        if self.config.stem_alignment == "mixture_duration" and reference_path is not None:
            if getattr(self, "reference_waveform", None) is None:
                reference = self._load_audio(reference_path)
                if reference.ndim != 1 or not reference.numel() or not self.torch.isfinite(reference).all():
                    raise ValueError(f"Expected finite nonempty mixture audio: {reference_path}")
                self.reference_waveform = reference
            reference = self.reference_waveform
            if path == reference_path:
                waveform = reference
                adjustment = {"decoded_samples": reference.numel(), "aligned_samples": reference.numel(),
                              "trimmed_end_samples": 0, "padded_end_samples": 0}
            else:
                raw = self._load_audio(path)
                if not self.torch.isfinite(raw).all():
                    raise ValueError(f"Non-finite stem audio: {path}")
                waveform, adjustment = align_stem(raw, reference.numel(),
                    int(self.config.max_alignment_seconds * self.config.sample_rate))
        else:
            waveform = self._load_audio(path)
            adjustment = {"decoded_samples": waveform.numel(), "aligned_samples": waveform.numel(),
                          "trimmed_end_samples": 0, "padded_end_samples": 0}
        if waveform.ndim != 1 or not self.torch.isfinite(waveform).all():
            raise ValueError(f"Expected finite mono audio: {path}")
        self.last_audio_metadata = {**adjustment,
            "reference_audio_path": str(reference_path) if reference_path is not None else None}
        ranges = segment_ranges(waveform.numel(), self.config.sample_rate, self.config.segment_seconds)
        with self.torch.inference_mode():
            for start, end in ranges:
                segment, _ = prepare_encoder_segment(waveform[start:end], self.config.minimum_input_samples,
                                                     self.config.short_segment_policy)
                batch = segment.unsqueeze(0).to(self.config.device)
                latent = self.model.get_latent(batch, layer_ix=self.config.layer_index)
                if latent.ndim != 3 or latent.shape[1] == 0:
                    raise ValueError(f"Encoder returned no usable frames for {path}, samples {start}:{end}")
                feature = latent.mean(dim=1).squeeze(0).to(self.torch.float32).cpu().numpy()
                yield start, end, feature


def read_segment_manifest(path: Path) -> list[dict]:
    """Read canonical Jamendo IDs without discarding multi-label metadata."""
    records, seen, seen_paths = [], set(), set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            track_id, relative = row["track_id"], PurePosixPath(row["relative_path"])
            if not isinstance(track_id, str) or not re.fullmatch(r"track_\d{7}", track_id):
                raise ValueError("Segment manifests require canonical Jamendo track IDs.")
            numeric = str(int(track_id.removeprefix("track_")))
            if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2 or relative.suffix != ".mp3" or (
                relative.stem != numeric or relative.parts[0] != numeric.zfill(2)[-2:]
            ):
                raise ValueError(f"Invalid relative audio path: {relative}")
            if track_id in seen or str(relative) in seen_paths or row["split"] not in ("train", "validation", "test"):
                raise ValueError(f"Repeated track/path or invalid split: {track_id}")
            if not isinstance(row.get("paths"), dict) or any(
                not isinstance(source, str) or not isinstance(value, str) or not Path(value).is_absolute()
                for source, value in row["paths"].items()
            ):
                raise ValueError(f"Expected absolute audio paths: {track_id}")
            seen.add(track_id); seen_paths.add(str(relative)); records.append(row)
    if not records:
        raise ValueError(f"Empty segment manifest: {path}")
    return records


def _write_track(directory: Path, config: SegmentExtractionConfig, row: dict,
                 source: str, settings: dict, extractor, audio_metadata=None) -> dict:
    path = Path(row["paths"][source])
    info = path.stat()
    if not path.is_file() or info.st_size == 0:
        raise ValueError(f"Audio is missing or empty: {path}")
    reference_identity = None
    if config.stem_alignment == "mixture_duration":
        reference_path = Path(row["paths"]["mixture"])
        reference_stat = reference_path.stat()
        if not reference_path.is_file() or reference_stat.st_size == 0:
            raise ValueError(f"Reference mixture is missing or empty: {reference_path}")
        reference_identity = {"path": str(reference_path), "size": reference_stat.st_size,
                              "mtime_ns": reference_stat.st_mtime_ns}
    signature = fingerprint({"settings": settings, "source": source, "track_id": row["track_id"],
                             "split": row["split"], "audio_path": str(path),
                             "audio_size": info.st_size, "audio_mtime_ns": info.st_mtime_ns,
                             "reference_mixture": reference_identity})
    if directory.exists() and not config.overwrite:
        if not reusable_artifacts(directory, signature):
            raise ValueError(f"Unverified existing segment folder; use a new output directory or --overwrite: {directory}")
        metadata = json.loads((directory / "segments.json").read_text())
        files = segment_files(directory)
        if len(files) != len(metadata["segments"]):
            raise ValueError(f"Unexpected segment files: {directory}")
        return {"status": "reused", "segments": len(files), "output": str(directory),
                "audio_alignment": metadata.get("audio_alignment"),
                "minimum_padded_segments": sum(segment["minimum_padding_samples"] > 0 for segment in metadata["segments"])}

    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    backup = None
    try:
        segments = []
        previous_end = 0
        width = int(config.sample_rate * config.segment_seconds)
        for index, (start, end, feature) in enumerate(extractor(path)):
            feature = np.asarray(feature, dtype=np.float32)
            if feature.shape != (config.expected_dimension,) or not np.isfinite(feature).all():
                raise ValueError(f"Expected a finite ({config.expected_dimension},) embedding: {path}")
            if start != previous_end or not start < end <= start + width or (index and segments[-1]["end_sample"] - segments[-1]["start_sample"] != width):
                raise ValueError(f"Noncontiguous or incorrectly partitioned audio segments: {path}")
            name = f"segment_{index}.npy"
            np.save(staging / name, feature, allow_pickle=False)
            padding = max(0, config.minimum_input_samples - (end - start)) if config.short_segment_policy == "pad_to_minimum" else 0
            segments.append({"file": name, "start_sample": start, "end_sample": end,
                             "audio_samples": end - start, "encoder_input_samples": end - start + padding,
                             "minimum_padding_samples": padding})
            previous_end = end
        if not segments:
            raise ValueError(f"No segments extracted: {path}")
        # Input changes during extraction must never be marked reusable.
        after = path.stat()
        if (info.st_size, info.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Audio changed during extraction: {path}")
        if reference_identity is not None:
            after_reference = reference_path.stat()
            if (after_reference.st_size, after_reference.st_mtime_ns) != (reference_stat.st_size, reference_stat.st_mtime_ns):
                raise ValueError(f"Reference mixture changed during extraction: {reference_path}")
        alignment = audio_metadata() if audio_metadata is not None else None
        if alignment is not None and previous_end != alignment["aligned_samples"]:
            raise ValueError("Encoded segments do not cover the entire aligned waveform.")
        write_json(staging / "segments.json", {"track_id": row["track_id"], "split": row["split"],
                   "source": source, "audio_path": str(path), "settings": settings,
                   "reference_mixture": reference_identity, "audio_alignment": alignment, "segments": segments})
        finish_artifacts(staging, signature, [segment["file"] for segment in segments] + ["segments.json"])
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{directory.name}.backup.", dir=directory.parent))
            backup.rmdir()
            directory.rename(backup)
        try:
            staging.rename(directory)
        except BaseException:
            if backup is not None:
                backup.rename(directory)
                backup = None
            raise
        return {"status": "extracted", "segments": len(segments), "output": str(directory),
                "audio_alignment": alignment,
                "minimum_padded_segments": sum(segment["minimum_padding_samples"] > 0 for segment in segments)}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None:
            shutil.rmtree(backup)


def extract_segment_features(config: SegmentExtractionConfig, sources: tuple[str, ...], *,
                             extractor_factory=None) -> dict:
    """Extract all requested views with one model; commit complete tracks atomically."""
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("Select distinct supported mixture/stem sources.")
    if config.chunk_seconds is not None or config.max_seconds is not None or config.temporal_pooling != "mean":
        raise ValueError("Segment extraction requires full audio and per-segment temporal mean pooling.")
    if config.minimum_input_samples != MINIMUM_INPUT_SAMPLES:
        raise ValueError(f"MusicFM25Hz requires minimum_input_samples={MINIMUM_INPUT_SAMPLES}.")
    if config.short_segment_policy not in {"pad_to_minimum", "error"}:
        raise ValueError("Unknown short-segment policy.")
    if config.stem_alignment not in {"mixture_duration", "none"}:
        raise ValueError("Unknown stem-alignment policy.")
    if not math.isfinite(config.max_alignment_seconds) or config.max_alignment_seconds < 0:
        raise ValueError("Alignment tolerance must be finite and nonnegative.")
    segment_ranges(1, config.sample_rate, config.segment_seconds)
    if not 0 <= config.layer_index <= 12 or config.expected_dimension <= 0:
        raise ValueError("Invalid MusicFM layer or feature dimension.")
    if not config.splits or len(set(config.splits)) != len(config.splits) or set(config.splits) - {"train", "validation", "test"}:
        raise ValueError("Select distinct known splits.")
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise ValueError("Per-split limit must be positive.")
    for path in (config.musicfm_root, config.effective_stats_path, config.effective_checkpoint_path):
        if not path.exists():
            raise ValueError(f"Missing MusicFM dependency: {path}")
    records = read_segment_manifest(config.manifest_path)
    selected = []
    for split in config.splits:
        rows = [row for row in records if row["split"] == split]
        if config.limit_per_split is not None:
            rows = rows[:config.limit_per_split]
        selected.extend(rows)
    if not selected:
        raise ValueError("No tracks match the selected splits.")
    settings = {"schema": 2, "architecture": "MusicFM25Hz", "sample_rate": config.sample_rate,
                "layer_index": config.layer_index, "dimension": config.expected_dimension,
                "segment_seconds": config.segment_seconds, "tail_policy": "retain",
                "short_segment_policy": config.short_segment_policy, "minimum_input_samples": config.minimum_input_samples,
                "stem_alignment": config.stem_alignment, "max_alignment_seconds": config.max_alignment_seconds,
                "temporal_pooling": "mean", "dtype": "float32", "is_flash": config.is_flash}
    report = {"dataset": "mtg_jamendo", "sources": list(sources), "tracks": len(selected),
              "planned": len(selected) * len(sources), "extracted": 0, "reused": 0, "failed": 0,
              "dry_run": config.dry_run, "settings": settings, "resolved_config": configuration_dict(config),
              "manifest_sha256": file_checksum(config.manifest_path), "records": [],
              "segment_count_mismatches": [], "minimum_padded_segments": 0, "adjusted_stem_views": 0, "exit_code": 0}
    if config.dry_run:
        return report
    settings.update({"checkpoint_sha256": file_checksum(config.effective_checkpoint_path),
                     "statistics_sha256": file_checksum(config.effective_stats_path)})
    runtime = None

    def extract(path):
        nonlocal runtime
        if runtime is None:
            runtime = extractor_factory(config) if extractor_factory else _SegmentRuntime(config)
        if isinstance(runtime, _SegmentRuntime) and config.stem_alignment == "mixture_duration":
            runtime.set_reference(Path(row["paths"]["mixture"]))
        return runtime(path)

    def runtime_metadata():
        return runtime.last_audio_metadata if isinstance(runtime, _SegmentRuntime) else None

    stop = False
    with logging_redirect_tqdm(loggers=[LOGGER]), tqdm(total=report["planned"], unit="view", desc="MusicFM segments", dynamic_ncols=True) as progress:
        for row in selected:
            counts = {}
            for source in sources:
                relative = PurePosixPath(row["relative_path"]).with_suffix("")
                prefix = "musicfm_30s" if source == "mixture" else f"musicfm_30s_{source}"
                if config.segment_seconds != 30:
                    prefix = prefix.replace("30s", f"{config.segment_seconds:g}s")
                directory = config.output_dir / prefix / str(relative)
                result = {"track_id": row["track_id"], "split": row["split"], "source": source}
                try:
                    result.update(_write_track(directory, config, row, source, settings, extract, runtime_metadata))
                    report[result["status"]] += 1
                    counts[source] = result["segments"]
                    report["minimum_padded_segments"] += result["minimum_padded_segments"]
                    alignment = result.get("audio_alignment")
                    if source != "mixture" and alignment is not None:
                        report["adjusted_stem_views"] += int(bool(alignment["trimmed_end_samples"] or alignment["padded_end_samples"]))
                except Exception as error:
                    result.update({"status": "failed", "reason": str(error)})
                    report["failed"] += 1
                    LOGGER.error("%s / %s: %s", row["track_id"], source, error)
                    stop = not config.continue_on_error
                report["records"].append(result)
                progress.update(1)
                if stop:
                    break
            if len(set(counts.values())) > 1:
                report["segment_count_mismatches"].append({"track_id": row["track_id"], "counts": counts})
            if stop:
                break
    report["exit_code"] = int(bool(report["failed"] or report["segment_count_mismatches"]))
    report["unprocessed"] = report["planned"] - len(report["records"])
    write_json(config.report_path or config.output_dir / "extraction_report_musicfm_segments.json", report)
    return report
