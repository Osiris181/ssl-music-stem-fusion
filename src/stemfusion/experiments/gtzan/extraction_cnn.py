"""GTZAN official frozen CNN extraction, using the unchanged Phase 2 manifest."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import site
import sys


def prepare_runtime():
    """Set official musicnn compatibility before imports; restart once if needed.

    The dynamic linker reads LD_LIBRARY_PATH at process startup, so merely
    setting it immediately before importing TensorFlow is insufficient.
    """
    packages = Path(site.getsitepackages()[0])
    libraries = sorted(str(path) for path in (packages / "nvidia").glob("*/lib") if path.is_dir())
    environment = os.environ.copy()
    previous = environment.get("LD_LIBRARY_PATH", "").split(os.pathsep)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        libraries + [path for path in previous if path and path not in libraries]
    )
    environment["TF_USE_LEGACY_KERAS"] = "1"
    environment.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    cuda_bin = packages / "nvidia/cuda_nvcc/bin"
    if cuda_bin.is_dir():
        previous_path = environment.get("PATH", "").split(os.pathsep)
        environment["PATH"] = os.pathsep.join(
            [str(cuda_bin)] + [path for path in previous_path if path != str(cuda_bin)]
        )
    if environment != dict(os.environ):
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], environment)


if __name__ == "__main__":
    prepare_runtime()

if __name__ == "__main__" and __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import yaml
from tqdm.auto import tqdm

from stemfusion.features.musicnn import (
    SOURCES, SAMPLE_RATE, VERSION, OfficialMusicNN, align_audio, digest,
    log_mel, pool_track, save_cache,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/experiments/gtzan/extraction_cnn.yaml"


def read_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for record in records:
        track_id = record["track_id"]
        if track_id in seen or Path(track_id).name != track_id or track_id in (".", ".."):
            raise ValueError(f"Duplicate or unsafe track ID: {track_id}")
        seen.add(track_id)
        if record["split"] not in ("train", "validation", "test"):
            raise ValueError("Unsupported split")
        if not isinstance(record["label"], int) or not 0 <= record["label"] < 10:
            raise ValueError("Expected GTZAN integer label in [0, 9]")
        if not all(source in record["paths"] for source in SOURCES):
            raise ValueError(f"Missing source mapping: {track_id}")
    if not records:
        raise ValueError("Empty manifest")
    return records


def cache_valid(path, expected):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        vector = cache["embedding"]
        valid = vector.shape == (200,) and vector.dtype == np.float32 and np.isfinite(vector).all()
        valid = valid and all(metadata.get(key) == value for key, value in expected.items())
    if not valid:
        raise ValueError(f"Incompatible existing cache (use --overwrite explicitly): {path}")
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("/GPU:0", "/CPU:0"))
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--limit-per-split", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)
    if args.limit_per_split is not None and args.limit_per_split < 1:
        parser.error("--limit-per-split must be positive")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    if set(config) != {"dataset", "paths", "runtime", "audio"} or config["dataset"] != "gtzan":
        parser.error("Expected GTZAN CNN extraction configuration")
    paths = {key: (config_path.parent / value).resolve() for key, value in config["paths"].items()}
    for key in ("manifest", "output_dir", "report"):
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key).resolve()
    runtime = dict(config["runtime"])
    if args.device:
        runtime["device"] = args.device
    tolerance = config["audio"]["maximum_source_duration_difference_seconds"]
    if runtime["batch_size"] < 1 or runtime["gpu_memory_mb"] < 1 or not 0 <= tolerance <= 0.25:
        parser.error("Invalid runtime or duration tolerance")
    records = read_records(paths["manifest"])
    selected = []
    counts = {}
    for record in records:
        split = record["split"]
        if args.split and split != args.split:
            continue
        if args.limit_per_split and counts.get(split, 0) >= args.limit_per_split:
            continue
        counts[split] = counts.get(split, 0) + 1
        selected.append(record)
    print(f"GTZAN CNN: {counts}; {len(selected) * len(SOURCES)} views; output={paths['output_dir']}")
    if args.dry_run:
        return 0
    import librosa
    model = OfficialMusicNN(**{key: runtime[key] for key in ("device", "gpu_memory_mb")})
    identity = {
        "dataset": "gtzan", "extraction_version": VERSION,
        "manifest_sha256": digest(paths["manifest"]),
        "implementation_sha256": digest(Path(__file__).resolve().parents[2] / "features/musicnn.py"),
        "model": model.provenance, "runtime": runtime,
        "duration_tolerance_seconds": tolerance,
        "window_frames": 187, "hop_frames": 187, "sample_rate": SAMPLE_RATE,
        "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean",
    }
    report = {"identity": identity, "selected_splits": counts, "extracted": 0, "reused": 0, "failed": []}
    try:
        with tqdm(total=len(selected) * len(SOURCES), desc="MSD_musicnn", unit="view") as progress:
            for record in selected:
                mixture = None
                for source in SOURCES:
                    path = Path(record["paths"][source])
                    output = paths["output_dir"] / record["split"] / source / f"{record['track_id']}.npz"
                    try:
                        stat = path.stat()
                        reference_path = Path(record["paths"]["mixture"])
                        reference_stat = reference_path.stat()
                        expected = {**identity, "track_id": record["track_id"], "split": record["split"],
                                    "label": record["label"], "source": source,
                                    "audio_path": str(path), "audio_bytes": stat.st_size,
                                    "audio_mtime_ns": stat.st_mtime_ns,
                                    "mixture_path": str(reference_path), "mixture_bytes": reference_stat.st_size,
                                    "mixture_mtime_ns": reference_stat.st_mtime_ns}
                        if not args.overwrite and cache_valid(output, expected):
                            report["reused"] += 1
                            continue
                        if mixture is None:
                            mixture, _ = librosa.load(reference_path, sr=SAMPLE_RATE, mono=True)
                            align_audio(mixture, len(mixture), tolerance)
                        audio = mixture if source == "mixture" else librosa.load(path, sr=SAMPLE_RATE, mono=True)[0]
                        original_samples = len(audio)
                        audio = align_audio(audio, len(mixture), tolerance)
                        spectrogram = log_mel(audio)
                        vector, windows = pool_track(model, spectrogram, runtime["batch_size"])
                        metadata = {**expected, "audio_sha256": digest(path), "original_samples": original_samples,
                                    "aligned_samples": len(audio), "spectrogram_frames": len(spectrogram),
                                    "windows": windows, "padded_frames": windows * 187 - len(spectrogram)}
                        save_cache(output, vector, metadata)
                        report["extracted"] += 1
                    except Exception as error:
                        report["failed"].append({"track_id": record["track_id"], "split": record["split"],
                                                 "source": source, "error": str(error)})
                        tqdm.write(f"{record['track_id']}/{source}: {error}")
                        if args.fail_fast:
                            raise
                    finally:
                        progress.update(1)
    finally:
        model.close()
        paths["report"].parent.mkdir(parents=True, exist_ok=True)
        paths["report"].write_text(json.dumps(report, indent=2) + "\n")
        print(f"CNN extraction: extracted={report['extracted']}, reused={report['reused']}, "
              f"failed={len(report['failed'])}; report={paths['report']}")
    return 1 if report["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
