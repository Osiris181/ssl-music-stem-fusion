"""Manifest-driven BTC chords and MIDI-derived global key, following Music2Emotion.

Algorithms are adapted from AMAAI-Lab/Music2Emotion's music2emo.py at
c5f7984f78d9928b841f25bd05a47377c12c58ca. No Music2Emotion UI or classifier is imported.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import logging
import math
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from stemfusion.configuration import configuration_dict
from stemfusion.features._chord_cqt import CQT_MINIMUM_INPUT_SAMPLES, cqt_features_with_tail_padding
from stemfusion.features.musicfm_segments import read_segment_manifest
from stemfusion.training.reporting import file_checksum, finish_artifacts, fingerprint, reusable_artifacts, write_json

LOGGER = logging.getLogger(__name__)
AUTHOR_COMMIT = "c5f7984f78d9928b841f25bd05a47377c12c58ca"
AUTHOR_URL = f"https://github.com/AMAAI-Lab/Music2Emotion/blob/{AUTHOR_COMMIT}/music2emo.py"
AUTHOR_FILES = {
    "utils/btc_model.py": "b8cc0b66f3e7bc92e9703ea408bce21d8a39b10c3133cb2ab569c285fceb3a27",
    "utils/transformer_modules.py": "af25b6fdc04573bbc7577b619ac837e0106dc1d3ed0b9fa3743521c9b8273643",
    "utils/hparams.py": "c5c25f3f886a137eccc6761df845cc87a1f02023734d0c2d88698f3a9af3c47c",
    "utils/mir_eval_modules.py": "7fef796a2899e72cea726582c3418cda4e562df6cd5fb62b44f431cdaeb888cb",
}
BTC_CONFIG_SHA256 = "f1d924d79c781ffdae800dce7818fcac49d9d4887d09541424b93c80d49be2df"
# Only this verified previous implementation is eligible for unchanged-cache
# reuse. Never ignore arbitrary implementation, checkpoint, or package changes.
LEGACY_IMPLEMENTATION_SHA256 = "ced7d0b000c1ddcf60a026f2bbb6bad32068b7df8aa65983f2886cdb95ed6830"
NORMAL_PIPELINE_REVISION = "music2emotion_released_v1"
SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")
PITCHES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
FLATS = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#"}


@dataclass(frozen=True)
class ChordKeyExtractionConfig:
    manifest: Path
    output_dir: Path
    author_root: Path
    btc_config: Path
    btc_checkpoint: Path
    report: Path
    sources: tuple[str, ...] = SOURCES
    splits: tuple[str, ...] = ("train", "validation", "test")
    device: str = "cuda"
    midi_policy: str = "music2emotion_released"
    keep_midi: bool = True
    limit_per_split: int | None = None
    overwrite: bool = False
    fail_fast: bool = False
    dry_run: bool = False
    config_provenance: dict | None = None
    retry_failures: bool = True


def prediction_intervals(predictions, frame_count: int, timestep: int, time_unit: float, vocabulary: dict) -> list[str]:
    """Match the released block-padding, transition, and final-frame handling."""
    if frame_count <= 0 or timestep <= 0 or not math.isfinite(time_unit) or time_unit <= 0:
        raise ValueError("Expected nonempty predictions and positive timing settings.")
    num_pad = timestep - frame_count % timestep
    padded_count = frame_count + num_pad
    if len(predictions) != padded_count:
        raise ValueError("Predictions must include the released BTC block padding.")
    start_time, lines = 0.0, []
    previous = int(predictions[0])
    for index, value in enumerate(predictions):
        if index == 0:
            continue
        chord = int(value)
        boundary = time_unit * index
        if chord != previous:
            lines.append(f"{start_time:.3f} {boundary:.3f} {vocabulary[previous]}\n")
            start_time, previous = boundary, chord
        block, within = divmod(index, timestep)
        if block == padded_count // timestep - 1 and within + num_pad == timestep:
            if start_time != boundary:
                lines.append(f"{start_time:.3f} {boundary:.3f} {vocabulary[previous]}\n")
            break
    return lines


def parse_intervals(lines: list[str]) -> list[tuple[float, float, str]]:
    intervals, previous_end = [], 0.0
    for line in lines:
        start, end, chord = line.split()
        start, end = float(start), float(end)
        if not math.isfinite(start + end) or start < 0 or end <= start or start != previous_end:
            raise ValueError("Expected finite, contiguous, positive chord intervals starting at zero.")
        intervals.append((start, end, chord))
        previous_end = end
    if not intervals:
        raise ValueError("BTC produced no usable chord intervals.")
    return intervals


def normalize_chords(lines: list[str], key_signature: str) -> list[str]:
    """Transpose chords to C major or A minor as in the released normalizer."""
    parts = key_signature.replace("-", "b").split()
    if parts == ["None"]:
        shift = 0
    elif len(parts) == 2 and parts[1] in ("major", "minor"):
        tonic = parts[0][0].upper() + parts[0][1:]
        tonic = FLATS.get(tonic, tonic)
        shift = PITCHES.index(tonic)
        if parts[1] == "minor":
            shift = (shift - 9) % 12
    else:
        raise ValueError(f"Unsupported key signature: {key_signature}")
    result = []
    for line in lines:
        start, end, chord = line.split()
        if chord not in {"N", "X"}:
            root, separator, quality = chord.partition(":")
            chord = PITCHES[(PITCHES.index(root) - shift) % 12] + separator + quality
        result.append(f"{start} {end} {chord}\n")
    return result


def pitch_events(intervals, bitmaps, policy: str = "music2emotion_released") -> list[tuple[float, float, int]]:
    """Convert chord pitch classes into MIDI notes in octave 3, velocity set later.

    The default deliberately preserves the released single-interval edge case.
    complete_intervals fixes its missing final flush; it is an explicit opt-in.
    """
    if policy not in {"music2emotion_released", "complete_intervals"}:
        raise ValueError("Unknown MIDI event policy.")
    if len(intervals) != len(bitmaps) or not intervals:
        raise ValueError("Chord intervals and bitmaps must be nonempty and aligned.")
    notes = []
    for pitch in range(12):
        start, label = intervals[0][0], bitmaps[0][pitch]
        for index in range(1, len(intervals)):
            current = bitmaps[index][pitch]
            if current != label:
                if label == 1:
                    notes.append((start, intervals[index][0], pitch + 48))
                start, label = intervals[index][0], current
        if label == 1 and (len(intervals) > 1 or policy == "complete_intervals"):
            notes.append((start, intervals[-1][1], pitch + 48))
    return notes


def _import_author_tools(root: Path):
    """Load only BTC utility modules; never import the UI-heavy music2emo module."""
    loaded = sys.modules.get("utils")
    if loaded is not None and root / "utils" not in [Path(p).resolve() for p in getattr(loaded, "__path__", ())]:
        raise ValueError("Another 'utils' package is already imported; run chord/key extraction in a fresh process.")
    sys.path.insert(0, str(root))
    try:
        model = importlib.import_module("utils.btc_model")
        hparams = importlib.import_module("utils.hparams")
        features = importlib.import_module("utils.mir_eval_modules")
    finally:
        sys.path.pop(0)
    for module in (model, hparams, features, sys.modules["utils.transformer_modules"]):
        path = Path(module.__file__).resolve()
        if path.parent != root / "utils" or file_checksum(path) != AUTHOR_FILES[f"utils/{path.name}"]:
            raise ValueError(f"Imported BTC utility does not match the pinned author source: {path}")
    return model.BTC_model, hparams.HParams, features


class _ChordKeyRuntime:
    def __init__(self, config: ChordKeyExtractionConfig):
        import torch
        import mir_eval
        import pretty_midi
        from music21 import converter

        self.torch, self.mir_eval, self.pretty_midi, self.converter = torch, mir_eval, pretty_midi, converter
        self.config = config
        model_class, hparams, self.tools = _import_author_tools(config.author_root)
        self.parameters = hparams.load(str(config.btc_config))
        self.parameters.feature["large_voca"] = True
        self.parameters.model["num_chords"] = 170
        self.model = model_class(config=self.parameters.model).to(config.device)
        # This is an explicitly supplied, trusted external research checkpoint.
        checkpoint = torch.load(config.btc_checkpoint, map_location="cpu", weights_only=False)
        self.mean, self.std = np.asarray(checkpoint["mean"]), np.asarray(checkpoint["std"])
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std <= 0).any():
            raise ValueError("Invalid BTC checkpoint normalization statistics.")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.model.requires_grad_(False)

    def __call__(self, audio: Path, staging: Path) -> dict:
        feature, time_unit, duration = self.tools.audio_file_to_features(str(audio), self.parameters)
        return self._features_to_harmony(feature, time_unit, duration, staging)

    def retry_with_cqt_padding(self, audio: Path, staging: Path) -> dict:
        feature, time_unit, duration, repair = cqt_features_with_tail_padding(audio, self.parameters)
        if not repair["padded_blocks"]:
            raise ValueError("No unsupported tiny CQT tail was found; refusing an unrelated repair.")
        result = self._features_to_harmony(feature, time_unit, duration, staging)
        return {**result, "cqt_repair": repair}

    def _features_to_harmony(self, feature, time_unit, duration, staging) -> dict:
        feature = (feature.T - self.mean) / self.std
        if feature.ndim != 2 or feature.shape[1] != self.parameters.model["feature_size"] or not np.isfinite(feature).all():
            raise ValueError("BTC features must be finite frame-by-feature values.")
        frames, timestep = len(feature), self.parameters.model["timestep"]
        if frames == 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("BTC input has no audio/features.")
        padding = timestep - frames % timestep
        feature = np.pad(feature, ((0, padding), (0, 0)), mode="constant")
        predictions = []
        with self.torch.inference_mode():
            for start in range(0, len(feature), timestep):
                batch = self.torch.as_tensor(feature[start:start + timestep], dtype=self.torch.float32,
                                             device=self.config.device).unsqueeze(0)
                latent, _ = self.model.self_attn_layers(batch)
                prediction, _ = self.model.output_layer(latent)
                predictions.extend(prediction.reshape(-1).cpu().tolist())
        lines = prediction_intervals(predictions, frames, timestep, time_unit, self.tools.idx2voca_chord())
        intervals = parse_intervals(lines)
        (staging / "chords.lab").write_text("".join(lines), encoding="utf-8")
        bitmaps = []
        for _, _, chord in intervals:
            root, bitmap, _ = self.mir_eval.chord.encode(chord)
            bitmaps.append(self.mir_eval.chord.rotate_bitmap_to_root(bitmap, root))
        events = pitch_events(intervals, bitmaps, self.config.midi_policy)
        midi = self.pretty_midi.PrettyMIDI()
        instrument = self.pretty_midi.Instrument(program=0)
        for start, end, pitch in events:
            instrument.notes.append(self.pretty_midi.Note(velocity=120, pitch=pitch, start=start, end=end))
        midi.instruments.append(instrument)
        midi_path = staging / "chords.mid"
        midi.write(str(midi_path))
        key, key_error = "None", None
        try:
            key = str(self.converter.parse(str(midi_path)).analyze("key")).replace("-", "b")
        except Exception as error:
            key_error = f"{type(error).__name__}: {error}"
        # None means no key estimate; it is never written as a fabricated major key.
        normalized = normalize_chords(lines, key)
        (staging / "normalized_chords.lab").write_text("".join(normalized), encoding="utf-8")
        (staging / "key.lab").write_text(key + "\n", encoding="utf-8")
        if not self.config.keep_midi:
            midi_path.unlink()
        return {"audio_duration_seconds": duration, "btc_frames": frames, "btc_padding_frames": padding,
                "btc_frame_time_seconds": time_unit, "chord_intervals": len(intervals), "midi_notes": len(events),
                "key": key, "key_error": key_error, "key_status": "unavailable" if key == "None" else "estimated"}


def _settings(config: ChordKeyExtractionConfig) -> dict:
    for name, expected in AUTHOR_FILES.items():
        if file_checksum(config.author_root / name) != expected:
            raise ValueError(f"Author utility differs from the pinned release: {name}")
    if file_checksum(config.btc_config) != BTC_CONFIG_SHA256:
        raise ValueError("BTC configuration differs from the verified author release.")
    import yaml

    parameters = yaml.safe_load(config.btc_config.read_text())
    versions = {}
    for package in ("numpy", "torch", "librosa", "mir_eval", "pretty_midi", "music21"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    dependency_sources = {}
    for package, version in versions.items():
        if version is not None:
            direct_url = importlib.metadata.distribution(package).read_text("direct_url.json")
            if direct_url:
                dependency_sources[package] = json.loads(direct_url)
    return {"schema": 2, "author_commit": AUTHOR_COMMIT, "author_url": AUTHOR_URL,
            "implementation_sha256": file_checksum(Path(__file__)), "package_versions": versions,
            "normal_pipeline_revision": NORMAL_PIPELINE_REVISION,
            "cqt_retry": {"policy": "right_zero_pad_to_minimum", "minimum_input_samples": CQT_MINIMUM_INPUT_SAMPLES,
                          "implementation_sha256": file_checksum(Path(__file__).with_name("_chord_cqt.py"))},
            "package_sources": dependency_sources,
            "author_files": AUTHOR_FILES, "btc_config_sha256": BTC_CONFIG_SHA256,
            "checkpoint_sha256": file_checksum(config.btc_checkpoint),
            "btc_parameters": {k: parameters[k] for k in ("mp3", "feature", "model")},
            "midi_policy": config.midi_policy, "midi_program": 0, "midi_velocity": 120, "midi_pitch_base": 48,
            "keep_midi": config.keep_midi, "key_failure_policy": "record_none",
            "normalization": "C_major_A_minor", "scope": "independent_per_source", "device": config.device}


def _reuse_view(directory, payload, keep_midi):
    """Verify current outputs or the single known, numerically compatible release.

    Legacy files, metadata, and completion markers are never migrated/rewritten.
    The exemption permits only the unchanged successful normal pipeline; every
    scientific setting, package version, input identity, and checksum must match.
    """
    if not (directory / "metadata.json").is_file() or not (directory / "run_manifest.json").is_file():
        raise ValueError(f"Unverified existing chord/key output; choose a new directory or --overwrite: {directory}")
    metadata = json.loads((directory / "metadata.json").read_text())
    settings = payload["settings"]
    cached_settings = metadata.get("settings")
    legacy = False
    if cached_settings != settings:
        expected_legacy = {key: value for key, value in settings.items()
                           if key not in {"normal_pipeline_revision", "cqt_retry"}}
        expected_legacy.update(schema=1, implementation_sha256=LEGACY_IMPLEMENTATION_SHA256)
        if (settings.get("schema") != 2 or settings.get("normal_pipeline_revision") != NORMAL_PIPELINE_REVISION
                or cached_settings != expected_legacy):
            raise ValueError(f"Existing run has incompatible settings or inputs: {directory}")
        legacy = True
    for key in ("audio", "track_id", "split", "source"):
        if metadata.get(key) != payload[key]:
            raise ValueError(f"Existing run has incompatible settings or inputs: {directory}")
    signature = fingerprint({**payload, "settings": cached_settings})
    required = {"chords.lab", "normalized_chords.lab", "key.lab", "metadata.json"}
    if keep_midi:
        required.add("chords.mid")
    marker = json.loads((directory / "run_manifest.json").read_text())
    if set(marker.get("files", {})) != required or not reusable_artifacts(directory, signature):
        raise ValueError(f"Unverified existing chord/key artifacts: {directory}")
    return {"status": "reused", "output": str(directory), **metadata["extraction"],
            "reused_compatible_legacy": legacy,
            "cache_implementation_sha256": cached_settings.get("implementation_sha256")}


def _write_view(config, row, source, settings, extract):
    audio = Path(row["paths"][source])
    before = audio.stat()
    if not audio.is_file() or before.st_size <= 0:
        raise ValueError(f"Missing or empty audio: {audio}")
    directory = config.output_dir / source / PurePosixPath(row["relative_path"]).with_suffix("")
    identity = {"path": str(audio), "size": before.st_size, "mtime_ns": before.st_mtime_ns}
    payload = {"settings": settings, "audio": identity, "track_id": row["track_id"],
               "split": row["split"], "source": source}
    signature = fingerprint(payload)
    if directory.exists() and not config.overwrite:
        return _reuse_view(directory, payload, config.keep_midi)
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}.", dir=directory.parent))
    backup = None
    published = False
    try:
        result = extract(audio, staging)
        filenames = ["chords.lab", "normalized_chords.lab", "key.lab", "metadata.json"]
        if config.keep_midi:
            filenames.append("chords.mid")
        parse_intervals((staging / "chords.lab").read_text().splitlines())
        parse_intervals((staging / "normalized_chords.lab").read_text().splitlines())
        after = audio.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input audio changed during extraction: {audio}")
        write_json(staging / "metadata.json", {"track_id": row["track_id"], "split": row["split"],
                   "source": source, "audio": identity, "settings": settings, "extraction": result})
        finish_artifacts(staging, signature, filenames)
        if directory.exists():
            backup = Path(tempfile.mkdtemp(prefix=f".{directory.name}.previous.", dir=directory.parent))
            backup.rmdir()
            directory.replace(backup)
        try:
            staging.replace(directory)
            published = True
        except Exception:
            if backup is not None:
                backup.replace(directory)
                backup = None
            raise
        return {"status": "extracted", "output": str(directory), **result}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        # Never delete the recoverable previous cache if promotion/restoration failed.
        if published and backup is not None and backup.exists():
            shutil.rmtree(backup)


def _supports_cqt_retry(error) -> bool:
    match = re.fullmatch(r"Input signal length=(\d+) is too short for (\d+)-octave CQT", str(error))
    return bool(type(error).__name__ == "ParameterError" and match
                and 0 < int(match[1]) < CQT_MINIMUM_INPUT_SAMPLES and int(match[2]) == 6)


def extract_chords_keys(config: ChordKeyExtractionConfig, *, runtime_factory=None) -> dict:
    """Verify caches, run normal extraction, then optionally retry supported tails once."""
    if not config.sources or len(set(config.sources)) != len(config.sources) or set(config.sources) - set(SOURCES):
        raise ValueError("Choose distinct mixture/five-stem sources.")
    if not config.splits or len(set(config.splits)) != len(config.splits) or set(config.splits) - {"train", "validation", "test"}:
        raise ValueError("Choose distinct train/validation/test splits.")
    if config.midi_policy not in {"music2emotion_released", "complete_intervals"}:
        raise ValueError("Unknown MIDI event policy.")
    if config.limit_per_split is not None and config.limit_per_split <= 0:
        raise ValueError("Per-split limit must be positive.")
    if not isinstance(config.retry_failures, bool):
        raise ValueError("retry_failures must be a boolean.")
    records = read_segment_manifest(config.manifest)
    selected = []
    for split in config.splits:
        rows = [row for row in records if row["split"] == split]
        selected.extend(rows[:config.limit_per_split] if config.limit_per_split else rows)
    if not selected:
        raise ValueError("No tracks match the selected splits.")
    if any(source not in row["paths"] for row in selected for source in config.sources):
        raise ValueError("Manifest is missing a requested source audio path.")
    report = {"dataset": "mtg_jamendo", "stage": "chords_keys", "tracks": len(selected),
              "sources": list(config.sources), "planned": len(selected) * len(config.sources),
              "extracted": 0, "reused": 0, "repaired": 0, "failed": 0, "key_unavailable": 0, "unprocessed": 0,
              "legacy_reused": 0, "initial_failed": 0, "retry_candidates": 0, "retry_attempted": 0,
              "retry_enabled": config.retry_failures, "initial_report": None,
              "dry_run": config.dry_run, "manifest_sha256": file_checksum(config.manifest),
              "resolved_config": configuration_dict(config), "records": [], "exit_code": 0}
    # Planning does not touch source audio, initialize models, or create outputs.
    if config.dry_run:
        return report
    report["settings"] = _settings(config)
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
            runtime = (runtime_factory or _ChordKeyRuntime)(config)
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
                    record.update(_write_view(config, row, source, report["settings"], extract))
                    report[record["status"]] += 1
                    report["key_unavailable"] += int(record["key_status"] == "unavailable")
                    report["legacy_reused"] += int(record.get("reused_compatible_legacy", False))
                except Exception as error:
                    record.update(status="failed", error=f"{type(error).__name__}: {error}",
                                  retry_eligible=_supports_cqt_retry(error))
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
                    result = _write_view(config, row, source, report["settings"], retry)
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
