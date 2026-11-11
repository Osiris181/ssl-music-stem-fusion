"""Frozen MERT segment extraction with Music2Emotion-compatible saved tensors."""

from __future__ import annotations

import importlib.metadata
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
from stemfusion.features.musicfm_segments import align_stem, read_segment_manifest, segment_ranges
from stemfusion.training.reporting import file_checksum, finish_artifacts, fingerprint, reusable_artifacts, write_json

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "m-a-p/MERT-v1-95M"
MODEL_REVISION = "12af15fef9d0ac838c3f475bfbbf26d2060dd4f5"
MODEL_FILES = ("config.json", "preprocessor_config.json", "configuration_MERT.py", "modeling_MERT.py", "pytorch_model.bin")
SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals", "other")
CONV_KERNELS = (10, 3, 3, 3, 3, 2, 2)
CONV_STRIDES = (5, 2, 2, 2, 2, 2, 2)
MINIMUM_INPUT_SAMPLES = 400


@dataclass(frozen=True)
class MERTSegmentConfig:
    manifest_path: Path
    output_dir: Path
    model_name: str = MODEL_NAME
    revision: str = MODEL_REVISION
    model_path: Path | None = None
    local_files_only: bool = True
    device: str = "cuda"
    sample_rate: int = 24000
    segment_seconds: float = 30.0
    minimum_input_samples: int = MINIMUM_INPUT_SAMPLES
    short_segment_policy: str = "pad_to_minimum"
    stem_alignment: str = "mixture_duration"
    max_alignment_seconds: float = 0.25
    splits: tuple[str, ...] = ("train", "validation", "test")
    limit_per_split: int | None = None
    overwrite: bool = False
    continue_on_error: bool = True
    dry_run: bool = False
    report_path: Path | None = None
    config_provenance: dict | None = None


def minimum_conv_samples(kernels, strides) -> int:
    """Reverse the unpadded convolution stack to find one output frame."""
    if len(kernels) != len(strides) or not kernels or any(v <= 0 for v in (*kernels, *strides)):
        raise ValueError("Convolution kernels and strides must be positive, equally sized sequences.")
    size = 1
    for kernel, stride in reversed(tuple(zip(kernels, strides))):
        size = (size - 1) * stride + kernel
    return size


def prepare_mert_segment(waveform, policy: str):
    """Keep valid audio unchanged and pad only tails below MERT's 400 samples."""
    import torch

    if waveform.ndim != 1 or waveform.numel() <= 0 or not torch.isfinite(waveform).all():
        raise ValueError("MERT requires finite, nonempty mono segments.")
    if policy not in {"pad_to_minimum", "error"}:
        raise ValueError("Unknown short-segment policy.")
    padding = max(0, MINIMUM_INPUT_SAMPLES - waveform.numel())
    if padding and policy == "error":
        raise ValueError(f"MERT requires at least {MINIMUM_INPUT_SAMPLES} samples; received {waveform.numel()}.")
    return (torch.nn.functional.pad(waveform, (0, padding)) if padding else waveform), padding


def pool_hidden_states(hidden_states):
    """Exclude the embedding state; mean each transformer layer over time."""
    import torch

    if len(hidden_states) != 13:
        raise ValueError("MERT-v1-95M must return an embedding state and 12 transformer states.")
    layers = hidden_states[1:]
    shape = layers[0].shape
    if len(shape) != 3 or shape[0] != 1 or shape[1] < 1 or shape[2] != 768 or any(x.shape != shape for x in layers):
        raise ValueError("Expected MERT hidden states with shape (1, frames, 768).")
    pooled = torch.stack(layers, dim=1).mean(dim=2).to(torch.float32)
    if not torch.isfinite(pooled).all():
        raise ValueError("MERT returned non-finite embeddings.")
    return pooled


def resolve_model(config: MERTSegmentConfig) -> Path:
    if config.model_path is not None:
        root = config.model_path.expanduser().resolve()
    else:
        from huggingface_hub import snapshot_download

        root = Path(snapshot_download(config.model_name, revision=config.revision,
                    local_files_only=config.local_files_only, allow_patterns=list(MODEL_FILES)))
    for name in MODEL_FILES:
        if not (root / name).is_file():
            raise ValueError(f"Missing MERT model asset: {root / name}")
    return root


class _MERTRuntime:
    def __init__(self, config: MERTSegmentConfig, model_root: Path):
        import torch
        import torchaudio
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.config, self.torch, self.torchaudio = config, torch, torchaudio
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_root, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_root, trust_remote_code=True, local_files_only=True)
        architecture = self.model.config
        if (tuple(architecture.conv_kernel) != CONV_KERNELS or tuple(architecture.conv_stride) != CONV_STRIDES
                or architecture.hidden_size != 768 or architecture.num_hidden_layers != 12
                or minimum_conv_samples(architecture.conv_kernel, architecture.conv_stride) != config.minimum_input_samples
                or self.processor.sampling_rate != config.sample_rate
                or not self.processor.do_normalize):
            raise ValueError("Model/processor does not match the configured MERT-v1-95M architecture.")
        self.model.requires_grad_(False).float().to(config.device).eval()
        self.reference_path = None
        self.reference_waveform = None
        self.last_audio_metadata = None

    def _load_audio(self, path: Path):
        waveform, rate = self.torchaudio.load(str(path))
        waveform = waveform.mean(dim=0)
        if rate != self.config.sample_rate:
            waveform = self.torchaudio.transforms.Resample(rate, self.config.sample_rate)(waveform)
        if waveform.ndim != 1 or not waveform.numel() or not self.torch.isfinite(waveform).all():
            raise ValueError(f"Expected finite nonempty audio: {path}")
        return waveform

    def set_reference(self, path: Path):
        if self.reference_path != path:
            self.reference_path, self.reference_waveform = path, None

    def __call__(self, path: Path):
        if self.config.stem_alignment == "mixture_duration":
            if self.reference_path is None:
                raise ValueError("A mixture reference must be selected before extracting aligned stems.")
            if self.reference_waveform is None:
                self.reference_waveform = self._load_audio(self.reference_path)
            waveform = self.reference_waveform if path == self.reference_path else self._load_audio(path)
            waveform, adjustment = align_stem(waveform, self.reference_waveform.numel(),
                                              int(self.config.max_alignment_seconds * self.config.sample_rate))
        else:
            waveform = self._load_audio(path)
            adjustment = {"decoded_samples": waveform.numel(), "aligned_samples": waveform.numel(),
                          "trimmed_end_samples": 0, "padded_end_samples": 0}
        self.last_audio_metadata = {**adjustment, "reference_audio_path":
            str(self.reference_path) if self.config.stem_alignment == "mixture_duration" else None}
        with self.torch.inference_mode():
            for start, end in segment_ranges(waveform.numel(), self.config.sample_rate, self.config.segment_seconds):
                segment, _ = prepare_mert_segment(waveform[start:end], self.config.short_segment_policy)
                inputs = self.processor(segment.cpu().numpy(), sampling_rate=self.config.sample_rate, return_tensors="pt")
                inputs = {name: value.to(self.config.device) for name, value in inputs.items()}
                output = self.model(**inputs, output_hidden_states=True)
                yield start, end, pool_hidden_states(output.hidden_states).cpu().numpy()


def _audio_identity(path: Path) -> dict:
    stat = path.stat()
    if not path.is_file() or not stat.st_size:
        raise ValueError(f"Audio is missing or empty: {path}")
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write_track(directory, config, row, source, settings, extract, metadata):
    path = Path(row["paths"][source])
    identity = _audio_identity(path)
    reference = Path(row["paths"]["mixture"]) if config.stem_alignment == "mixture_duration" else None
    reference_identity = _audio_identity(reference) if reference is not None else None
    signature = fingerprint({"settings": settings, "track_id": row["track_id"], "split": row["split"],
                             "source": source, "audio": identity, "reference": reference_identity})
    if directory.exists() and not config.overwrite:
        if not reusable_artifacts(directory, signature):
            raise ValueError(f"Unverified existing MERT folder; use a new directory or --overwrite: {directory}")
        marker = json.loads((directory / "segments.json").read_text())
        if len(segment_files(directory)) != len(marker["segments"]):
            raise ValueError(f"Unexpected segment files: {directory}")
        return _track_result("reused", directory, marker)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    backup = None
    published = False
    try:
        segments, previous_end = [], 0
        width = int(config.sample_rate * config.segment_seconds)
        for index, (start, end, feature) in enumerate(extract(path)):
            feature = np.asarray(feature, dtype=np.float32)
            if feature.shape != (1, 12, 768) or not np.isfinite(feature).all():
                raise ValueError("Expected a finite (1, 12, 768) MERT embedding.")
            if start != previous_end or not start < end <= start + width or (index and segments[-1]["audio_samples"] != width):
                raise ValueError("Noncontiguous or incorrectly partitioned audio segments.")
            padding = max(0, config.minimum_input_samples - (end - start))
            if padding and config.short_segment_policy == "error":
                raise ValueError("Short MERT tail requires minimum padding under the configured policy.")
            name = f"segment_{index}.npy"
            np.save(staging / name, feature, allow_pickle=False)
            segments.append({"file": name, "start_sample": start, "end_sample": end,
                             "audio_samples": end - start, "encoder_input_samples": end - start + padding,
                             "minimum_padding_samples": padding})
            previous_end = end
        if not segments:
            raise ValueError("No MERT segments extracted.")
        if identity != _audio_identity(path) or (reference is not None and reference_identity != _audio_identity(reference)):
            raise ValueError("Audio or reference changed during extraction.")
        alignment = metadata()
        if alignment is not None and previous_end != alignment["aligned_samples"]:
            raise ValueError("Segments do not cover the entire aligned waveform.")
        marker = {"track_id": row["track_id"], "split": row["split"], "source": source,
                  "audio": identity, "reference_mixture": reference_identity, "audio_alignment": alignment,
                  "settings": settings, "segments": segments}
        write_json(staging / "segments.json", marker)
        finish_artifacts(staging, signature, [s["file"] for s in segments] + ["segments.json"])
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{directory.name}.backup.", dir=directory.parent))
            backup.rmdir()
            directory.rename(backup)
        try:
            staging.rename(directory)
        except BaseException:
            if backup is not None:
                try:
                    backup.rename(directory)
                    backup = None
                except OSError:
                    LOGGER.critical("Could not restore old MERT cache; recover it from %s", backup)
            raise
        published = True
        return _track_result("extracted", directory, marker)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # Never erase a backup when publication and restoration both failed.
        if backup is not None and published:
            shutil.rmtree(backup)


def _track_result(status, directory, marker):
    return {"status": status, "output": str(directory), "segments": len(marker["segments"]),
            "audio_alignment": marker["audio_alignment"],
            "minimum_padded_segments": sum(s["minimum_padding_samples"] > 0 for s in marker["segments"])}


def extract_mert_segments(config: MERTSegmentConfig, sources: tuple[str, ...], *, extractor_factory=None) -> dict:
    """Plan without audio I/O, or extract/reuse complete, checksummed track views."""
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("Select distinct supported sources.")
    if config.model_name != MODEL_NAME or not re.fullmatch(r"[0-9a-f]{40}", config.revision):
        raise ValueError("Use MERT-v1-95M with an immutable 40-character model revision.")
    if config.sample_rate != 24000 or config.minimum_input_samples != MINIMUM_INPUT_SAMPLES:
        raise ValueError("MERT-v1-95M requires 24 kHz audio and a 400-sample minimum.")
    if config.short_segment_policy not in {"pad_to_minimum", "error"} or config.stem_alignment not in {"mixture_duration", "none"}:
        raise ValueError("Unknown short-segment or stem-alignment policy.")
    if not math.isfinite(config.max_alignment_seconds) or config.max_alignment_seconds < 0:
        raise ValueError("Alignment tolerance must be finite and nonnegative.")
    segment_ranges(1, config.sample_rate, config.segment_seconds)
    if not config.splits or len(set(config.splits)) != len(config.splits) or set(config.splits) - {"train", "validation", "test"}:
        raise ValueError("Select distinct known splits.")
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise ValueError("Per-split limit must be positive.")
    rows = read_segment_manifest(config.manifest_path)
    selected = []
    for split in config.splits:
        split_rows = [row for row in rows if row["split"] == split]
        selected.extend(split_rows[:config.limit_per_split])
    if not selected:
        raise ValueError("No tracks match the selected splits.")
    settings = {"schema": 1, "architecture": "MERT-v1-95M", "model_name": config.model_name,
                "reference_implementation_commit": "c5f7984f78d9928b841f25bd05a47377c12c58ca",
                "revision": config.revision, "sample_rate": config.sample_rate, "channels": "mono",
                "segment_seconds": config.segment_seconds, "tail_policy": "retain",
                "short_segment_policy": config.short_segment_policy, "minimum_input_samples": config.minimum_input_samples,
                "stem_alignment": config.stem_alignment, "max_alignment_seconds": config.max_alignment_seconds,
                "temporal_pooling": "mean", "saved_hidden_states": list(range(1, 13)),
                "shape": [1, 12, 768], "dtype": "float32", "device": config.device,
                "audio_loader": "torchaudio.load/channel_mean/transforms.Resample", "processor_normalize": True}
    report = {"dataset": "mtg_jamendo", "sources": list(sources), "tracks": len(selected),
              "planned": len(selected) * len(sources), "extracted": 0, "reused": 0, "failed": 0,
              "unprocessed": len(selected) * len(sources), "dry_run": config.dry_run,
              "settings": settings, "resolved_config": configuration_dict(config),
              "manifest_sha256": file_checksum(config.manifest_path), "records": [],
              "segment_count_mismatches": [], "minimum_padded_segments": 0, "adjusted_stem_views": 0, "exit_code": 0}
    if config.dry_run:
        return report
    model_root = resolve_model(config)
    versions = {}
    for package in ("torch", "torchaudio", "torchcodec", "soundfile", "transformers", "numpy", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    settings.update({"model_assets_sha256": {name: file_checksum(model_root / name) for name in MODEL_FILES},
                     "software_versions": versions,
                     "implementation_sha256": {name: file_checksum(Path(__file__).parent / name)
                                               for name in ("mert_segments.py", "musicfm_segments.py")}})
    runtime = None

    def extract(path):
        nonlocal runtime
        if runtime is None:
            runtime = extractor_factory(config, model_root) if extractor_factory else _MERTRuntime(config, model_root)
        if hasattr(runtime, "set_reference"):
            runtime.set_reference(Path(row["paths"]["mixture"]))
        return runtime(path)

    stop = False
    with logging_redirect_tqdm(loggers=[LOGGER]), tqdm(total=report["planned"], unit="view", desc="MERT segments", dynamic_ncols=True) as progress:
        for row in selected:
            counts = {}
            for source in sources:
                prefix = f"mert_{config.segment_seconds:g}s" + (f"_{source}" if source != "mixture" else "")
                directory = config.output_dir / prefix / str(PurePosixPath(row["relative_path"]).with_suffix(""))
                result = {"track_id": row["track_id"], "split": row["split"], "source": source}
                try:
                    result.update(_write_track(directory, config, row, source, settings, extract,
                                  lambda: getattr(runtime, "last_audio_metadata", None)))
                    report[result["status"]] += 1
                    counts[source] = result["segments"]
                    report["minimum_padded_segments"] += result["minimum_padded_segments"]
                    alignment = result["audio_alignment"]
                    if source != "mixture" and alignment is not None:
                        report["adjusted_stem_views"] += int(bool(alignment["trimmed_end_samples"] or alignment["padded_end_samples"]))
                except Exception as error:
                    result.update(status="failed", reason=str(error))
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
    report["unprocessed"] = report["planned"] - len(report["records"])
    report["exit_code"] = int(bool(report["failed"] or report["unprocessed"] or report["segment_count_mismatches"]))
    write_json(config.report_path or config.output_dir / "mert_extraction_report.json", report)
    return report
