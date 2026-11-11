"""Dataset-neutral scheduling through the unchanged MERT/BTC feature backends.

The earlier PMEmo adapter remains frozen while its extraction is running. New
emotion datasets supply their own manifest reader here. Neural inference,
alignment, pooling, artifact writers and CQT repair all use existing backends.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stemfusion.configuration import configuration_dict
from stemfusion.features import mert_segments as mert_backend
from stemfusion.features import chords_keys as harmony_backend
from stemfusion.training.reporting import file_checksum, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmotionDatasetAdapter:
    dataset: str
    manifest_reader: Callable[[Path], list[dict]]

    def __post_init__(self):
        if not re.fullmatch(r"[a-z][a-z_]*", self.dataset) or not callable(self.manifest_reader):
            raise ValueError("Provide a canonical dataset name and manifest reader.")

    def read_manifest(self, path):
        rows = self.manifest_reader(path)
        if not rows or any(row.get("dataset") != self.dataset for row in rows):
            raise ValueError("Feature manifest does not match the requested dataset.")
        return rows

    def identity(self):
        source = inspect.getsourcefile(self.manifest_reader)
        if source is None:
            raise ValueError("Dataset manifest reader must have a fingerprintable source file.")
        return {"dataset": self.dataset,
                "dataset_adapter_sha256": file_checksum(Path(__file__)),
                "manifest_reader_sha256": file_checksum(Path(source))}


def extract_mert_segments(config: mert_backend.MERTSegmentConfig, sources: tuple[str, ...], *, adapter: EmotionDatasetAdapter, extractor_factory=None) -> dict:
    """Plan without audio I/O, or extract/reuse complete, checksummed track views."""
    if not sources or len(set(sources)) != len(sources) or set(sources) - set(mert_backend.SOURCES):
        raise ValueError("Select distinct supported sources.")
    if config.model_name != mert_backend.MODEL_NAME or not re.fullmatch(r"[0-9a-f]{40}", config.revision):
        raise ValueError("Use MERT-v1-95M with an immutable 40-character model revision.")
    if config.sample_rate != 24000 or config.minimum_input_samples != mert_backend.MINIMUM_INPUT_SAMPLES:
        raise ValueError("MERT-v1-95M requires 24 kHz audio and a 400-sample minimum.")
    if config.short_segment_policy not in {"pad_to_minimum", "error"} or config.stem_alignment not in {"mixture_duration", "none"}:
        raise ValueError("Unknown short-segment or stem-alignment policy.")
    if not math.isfinite(config.max_alignment_seconds) or config.max_alignment_seconds < 0:
        raise ValueError("Alignment tolerance must be finite and nonnegative.")
    mert_backend.segment_ranges(1, config.sample_rate, config.segment_seconds)
    if not config.splits or len(set(config.splits)) != len(config.splits) or set(config.splits) - {"train", "validation", "test"}:
        raise ValueError("Select distinct known splits.")
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise ValueError("Per-split limit must be positive.")
    rows = adapter.read_manifest(config.manifest_path)
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
    report = {"dataset": adapter.dataset, "sources": list(sources), "tracks": len(selected),
              "planned": len(selected) * len(sources), "extracted": 0, "reused": 0, "failed": 0,
              "unprocessed": len(selected) * len(sources), "dry_run": config.dry_run,
              "settings": settings, "resolved_config": configuration_dict(config),
              "manifest_sha256": file_checksum(config.manifest_path), "records": [],
              "segment_count_mismatches": [], "minimum_padded_segments": 0, "adjusted_stem_views": 0, "exit_code": 0}
    settings.update(adapter.identity())
    if config.dry_run:
        return report
    model_root = mert_backend.resolve_model(config)
    versions = {}
    for package in ("torch", "torchaudio", "torchcodec", "soundfile", "transformers", "numpy", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    settings.update({"model_assets_sha256": {name: file_checksum(model_root / name) for name in mert_backend.MODEL_FILES},
                     "software_versions": versions,
                     "implementation_sha256": {name: file_checksum(Path(__file__).parent / name)
                                               for name in ("mert_segments.py", "musicfm_segments.py")}})
    runtime = None

    def extract(path):
        nonlocal runtime
        if runtime is None:
            runtime = extractor_factory(config, model_root) if extractor_factory else mert_backend._MERTRuntime(config, model_root)
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
                    result.update(mert_backend._write_track(directory, config, row, source, settings, extract,
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


def extract_chords_keys(config: harmony_backend.ChordKeyExtractionConfig, *, adapter: EmotionDatasetAdapter, runtime_factory=None) -> dict:
    """Verify caches, run normal extraction, then optionally retry supported tails once."""
    if not config.sources or len(set(config.sources)) != len(config.sources) or set(config.sources) - set(harmony_backend.SOURCES):
        raise ValueError("Choose distinct mixture/five-stem sources.")
    if not config.splits or len(set(config.splits)) != len(config.splits) or set(config.splits) - {"train", "validation", "test"}:
        raise ValueError("Choose distinct train/validation/test splits.")
    if config.midi_policy not in {"music2emotion_released", "complete_intervals"}:
        raise ValueError("Unknown MIDI event policy.")
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise ValueError("Per-split limit must be positive.")
    if not isinstance(config.retry_failures, bool):
        raise ValueError("retry_failures must be a boolean.")
    records = adapter.read_manifest(config.manifest)
    selected = []
    for split in config.splits:
        rows = [row for row in records if row["split"] == split]
        selected.extend(rows[:config.limit_per_split] if config.limit_per_split else rows)
    if not selected:
        raise ValueError("No tracks match the selected splits.")
    if any(source not in row["paths"] for row in selected for source in config.sources):
        raise ValueError("Manifest is missing a requested source audio path.")
    report = {"dataset": adapter.dataset, "stage": "chords_keys", "tracks": len(selected),
              "sources": list(config.sources), "planned": len(selected) * len(config.sources),
              "extracted": 0, "reused": 0, "repaired": 0, "failed": 0, "key_unavailable": 0, "unprocessed": 0,
              "legacy_reused": 0, "initial_failed": 0, "retry_candidates": 0, "retry_attempted": 0,
              "retry_enabled": config.retry_failures, "initial_report": None,
              "dry_run": config.dry_run, "manifest_sha256": file_checksum(config.manifest),
              "resolved_config": configuration_dict(config), "records": [], "exit_code": 0}
    # Planning does not touch source audio, initialize models, or create outputs.
    if config.dry_run:
        return report
    report["settings"] = {**harmony_backend._settings(config), **adapter.identity()}
    if runtime_factory is None:
        missing = [name for name, version in report["settings"]["package_versions"].items() if version is None]
        if missing:
            raise ValueError(f"Missing optional chord/key dependencies: {', '.join(missing)}. Install the project's chords-keys extra.")
        import torch

        if torch.device(config.device).type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; select --device cpu or use a CUDA-enabled environment.")
    runtime = None

    def extract(audio, staging):
        nonlocal runtime
        if runtime is None:
            runtime = (runtime_factory or harmony_backend._ChordKeyRuntime)(config)
        return runtime(audio, staging)

    def retry(audio, staging):
        if runtime is None or not callable(getattr(runtime, "retry_with_cqt_padding", None)):
            raise ValueError("The extraction runtime does not support CQT-tail retries.")
        return runtime.retry_with_cqt_padding(audio, staging)

    stop = False
    retry_jobs = []
    with logging_redirect_tqdm(loggers=[LOGGER]), tqdm(total=report["planned"], desc="BTC chords/key", unit="view", dynamic_ncols=True) as progress:
        for row in selected:
            for source in config.sources:
                record = {"track_id": row["track_id"], "split": row["split"], "source": source, "attempts": 1}
                try:
                    record.update(harmony_backend._write_view(config, row, source, report["settings"], extract))
                    report[record["status"]] += 1
                    report["key_unavailable"] += int(record["key_status"] == "unavailable")
                    report["legacy_reused"] += int(record.get("reused_compatible_legacy", False))
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                  retry_eligible=harmony_backend._supports_cqt_retry(error))
                    report["failed"] += 1
                    report["exit_code"] = 1
                    if record["retry_eligible"]:
                        retry_jobs.append((row, source, record))
                    LOGGER.error("%s / %s: %s", row["track_id"], source, error)
                    stop = config.fail_fast
                report["records"].append(record)
                progress.update()
                if stop:
                    break
            if stop:
                break
    report["unprocessed"] = report["planned"] - len(report["records"])
    report["initial_failed"] = report["failed"]
    report["retry_candidates"] = len(retry_jobs)
    if config.retry_failures and retry_jobs and not stop:
        initial_report = config.report.with_name(f"{config.report.stem}.initial.json")
        report["initial_report"] = str(initial_report)
        # Preserve first-pass failures before any records are updated in memory.
        write_json(initial_report, report)
        LOGGER.info("Normal pass finished: %d failed; retrying %d supported CQT tails once.",
                    report["initial_failed"], len(retry_jobs))
        with logging_redirect_tqdm(loggers=[LOGGER]), tqdm(total=len(retry_jobs), desc="BTC CQT retry", unit="view", dynamic_ncols=True) as progress:
            for row, source, record in retry_jobs:
                record["initial_error"] = record["error"]
                record["attempts"] = 2
                report["retry_attempted"] += 1
                try:
                    result = harmony_backend._write_view(config, row, source, report["settings"], retry)
                    status = "repaired" if result["status"] == "extracted" else result["status"]
                    record.update(result)
                    record["status"] = status
                    record.pop("error", None)
                    report[status] += 1
                    report["failed"] -= 1
                    report["key_unavailable"] += int(record["key_status"] == "unavailable")
                    report["legacy_reused"] += int(record.get("reused_compatible_legacy", False))
                except Exception as error:
                    record["error"] = f"{type(error).__name__}: {error}"
                    record["retry_error"] = record["error"]
                    LOGGER.error("Retry %s / %s: %s", row["track_id"], source, error)
                progress.update()
    elif stop and retry_jobs:
        report["retry_skipped_reason"] = "fail_fast"
    report["exit_code"] = int(bool(report["failed"] or report["unprocessed"]))
    write_json(config.report, report)
    return report
