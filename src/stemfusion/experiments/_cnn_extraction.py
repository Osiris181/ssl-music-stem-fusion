"""Shared official frozen CNN extraction over existing dataset manifests."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import yaml
from tqdm.auto import tqdm

from stemfusion.features import _cnn_audio
from stemfusion.features.musicnn import (
    SOURCES, SAMPLE_RATE, VERSION, OfficialMusicNN, align_audio, digest,
    log_mel, pool_track, save_cache,
)


def read_records(path, *, dataset="gtzan"):
    classes = {"gtzan": 10, "ismir04": 6, "fma_medium": 16, "mtg_jamendo": None,
               "pmemo": None, "deam": None, "emomusic": None}[dataset]
    if dataset == "mtg_jamendo":
        from stemfusion.features.musicfm_segments import read_segment_manifest
        records = read_segment_manifest(path)
    elif dataset == "pmemo":
        from stemfusion.data.pmemo import read_feature_manifest
        records = read_feature_manifest(path)
    elif dataset == "deam":
        from stemfusion.data.deam import read_feature_manifest
        records = read_feature_manifest(path)
    elif dataset == "emomusic":
        from stemfusion.data.emomusic import read_feature_manifest
        records = read_feature_manifest(path)
    else:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    seen = set()
    for record in records:
        track_id = record["track_id"]
        if track_id in seen or Path(track_id).name != track_id or track_id in (".", ".."):
            raise ValueError(f"Duplicate or unsafe track ID: {track_id}")
        seen.add(track_id)
        if record["split"] not in ("train", "validation", "test"):
            raise ValueError("Unsupported split")
        if dataset == "mtg_jamendo":
            tags = record.get("tags")
            if not isinstance(tags, list) or not tags or any(
                not isinstance(tag, str) or not tag.startswith("mood/theme---") for tag in tags
            ) or len(tags) != len(set(tags)):
                raise ValueError(f"Expected distinct mood/theme tags: {track_id}")
        elif dataset not in ("pmemo", "deam", "emomusic") and (not isinstance(record["label"], int) or not 0 <= record["label"] < classes):
            raise ValueError(f"Expected {dataset} integer label in [0, {classes - 1}]")
        if not all(source in record["paths"] for source in SOURCES):
            raise ValueError(f"Missing source mapping: {track_id}")
    if not records:
        raise ValueError("Empty manifest")
    if dataset == "ismir04":
        from stemfusion.data.ismir04 import inspect_manifest
        inspect_manifest(path, classes)
    elif dataset == "fma_medium":
        from stemfusion.data.fma_features import inspect_manifest
        inspect_manifest(path, classes)
    elif dataset == "mtg_jamendo":
        tags = {tag for record in records for tag in record["tags"]}
        if len(tags) != 56:
            raise ValueError(f"Expected the existing 56-tag Jamendo mood/theme setting, found {len(tags)}")
        if {record["split"] for record in records} != {"train", "validation", "test"}:
            raise ValueError("Jamendo manifest must retain all three existing splits")
    elif dataset in ("pmemo", "deam", "emomusic"):
        if {record["split"] for record in records} != {"train", "validation", "test"}:
            name = {"pmemo": "PMEmo", "deam": "DEAM", "emomusic": "EmoMusic"}[dataset]
            raise ValueError(f"{name} manifest must retain all three existing splits")
    return records


def cache_valid(path, expected, *, reference_samples=None):
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
        vector = cache["embedding"]
        valid = vector.shape == (200,) and vector.dtype == np.float32 and np.isfinite(vector).all()
        valid = valid and all(metadata.get(key) == value for key, value in expected.items())
    if not valid:
        raise ValueError(f"Incompatible existing cache (use --overwrite explicitly): {path}")
    if reference_samples is not None and metadata.get("aligned_samples") != reference_samples:
        # Only an independently decoded reference justifies automatic repair.
        # Settings/model/source identity mismatches still require --overwrite.
        return False
    return True


def _reference_needs_validation(record, output_dir, overwrite, tolerance):
    if overwrite or any(not (output_dir / record["split"] / source / f"{record['track_id']}.npz").is_file()
                        for source in SOURCES):
        return True
    mixture_cache = output_dir / record["split"] / "mixture" / f"{record['track_id']}.npz"
    with np.load(mixture_cache, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"]))
    # A very short cached reference is checked, not discarded or padded to a
    # minimum song length. Legitimately short inputs remain valid and reusable.
    samples = metadata.get("aligned_samples", 0)
    return samples <= round(tolerance * SAMPLE_RATE)


def _backup_repaired_cache(path):
    checksum = digest(path)
    backup = path.with_name(path.name + ".before_audio_repair")
    if backup.exists():
        if digest(backup) != checksum:
            raise ValueError(f"An earlier repair backup already exists; preserve it before retrying: {backup}")
    else:
        shutil.copy2(path, backup)
    return {"backup": str(backup), "previous_cache_sha256": checksum}


def main(argv=None, *, dataset="gtzan"):
    if dataset not in {"gtzan", "ismir04", "fma_medium", "mtg_jamendo", "pmemo", "deam", "emomusic"}:
        raise ValueError(f"Unsupported CNN extraction dataset: {dataset}")
    default_config = Path(__file__).resolve().parents[3] / f"configs/experiments/{dataset}/extraction_cnn.yaml"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--device", choices=("/GPU:0", "/CPU:0"))
    parser.add_argument("--gpu-memory-mb", type=int,
                        help="TensorFlow GPU memory cap in MB; overrides YAML. Use the original value to resume older caches.")
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
    if set(config) != {"dataset", "paths", "runtime", "audio"} or config["dataset"] != dataset:
        parser.error(f"Expected {dataset} CNN extraction configuration")
    paths = {key: (config_path.parent / value).resolve() for key, value in config["paths"].items()}
    for key in ("manifest", "output_dir", "report"):
        if getattr(args, key) is not None:
            paths[key] = getattr(args, key).resolve()
    runtime = dict(config["runtime"])
    if args.device:
        runtime["device"] = args.device
    if args.gpu_memory_mb is not None:
        runtime["gpu_memory_mb"] = args.gpu_memory_mb
    tolerance = config["audio"]["maximum_source_duration_difference_seconds"]
    if runtime["batch_size"] < 1 or runtime["gpu_memory_mb"] < 1 or not 0 <= tolerance <= 0.25:
        parser.error("Invalid runtime or duration tolerance")
    records = read_records(paths["manifest"], dataset=dataset)
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
    print(f"{dataset.upper()} CNN: {counts}; {len(selected) * len(SOURCES)} views; output={paths['output_dir']}")
    if args.dry_run:
        return 0
    model = OfficialMusicNN(**{key: runtime[key] for key in ("device", "gpu_memory_mb")})
    identity = {
        "dataset": dataset, "extraction_version": VERSION,
        "manifest_sha256": digest(paths["manifest"]),
        "implementation_sha256": digest(Path(__file__).resolve().parents[1] / "features/musicnn.py"),
        "model": model.provenance, "runtime": runtime,
        "duration_tolerance_seconds": tolerance,
        "window_frames": 187, "hop_frames": 187, "sample_rate": SAMPLE_RATE,
        "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean",
    }
    audio_loading = {"policy": _cnn_audio.POLICY, "implementation_sha256": digest(Path(_cnn_audio.__file__))}
    report = {"identity": identity, "audio_loading": audio_loading, "selected_splits": counts,
              "extracted": 0, "reused": 0, "decoder_fallbacks": [], "repaired": [], "failed": []}
    if dataset == "mtg_jamendo":
        report["number_of_tags"] = len({tag for record in records for tag in record["tags"]})
    elif dataset in ("pmemo", "deam", "emomusic"):
        report.update(targets=["valence", "arousal"],
                      annotation_scale="original_0_1" if dataset == "pmemo" else "original_1_9",
                      binary_labels_created=False)
    try:
        with tqdm(total=len(selected) * len(SOURCES), desc="MSD_musicnn", unit="view") as progress:
            for record in selected:
                mixture = None
                mixture_decoding = None
                reference_error = None
                try:
                    validate_reference = _reference_needs_validation(record, paths["output_dir"], args.overwrite, tolerance)
                except Exception as error:
                    validate_reference = True
                    reference_error = error

                def reference_audio():
                    nonlocal mixture, mixture_decoding, reference_error
                    if reference_error is not None:
                        raise reference_error
                    if mixture is None:
                        try:
                            mixture, mixture_decoding = _cnn_audio.load_audio(
                                Path(record["paths"]["mixture"]), tolerance)
                        except Exception as error:
                            reference_error = error
                            raise
                    return mixture

                for source in SOURCES:
                    path = Path(record["paths"][source])
                    output = paths["output_dir"] / record["split"] / source / f"{record['track_id']}.npz"
                    try:
                        stat = path.stat()
                        reference_path = Path(record["paths"]["mixture"])
                        reference_stat = reference_path.stat()
                        if dataset == "mtg_jamendo":
                            target_metadata = {"tags": record["tags"], "relative_path": record["relative_path"]}
                        elif dataset in ("pmemo", "deam", "emomusic"):
                            target_metadata = {"annotations": record["annotations"], "relative_path": record["relative_path"]}
                        else:
                            target_metadata = {"label": record["label"]}
                        expected = {**identity, "track_id": record["track_id"], "split": record["split"],
                                    **target_metadata, "source": source,
                                    "audio_path": str(path), "audio_bytes": stat.st_size,
                                    "audio_mtime_ns": stat.st_mtime_ns,
                                    "mixture_path": str(reference_path), "mixture_bytes": reference_stat.st_size,
                                    "mixture_mtime_ns": reference_stat.st_mtime_ns}
                        if validate_reference:
                            reference_audio()
                        if not args.overwrite and cache_valid(
                            output, expected, reference_samples=len(mixture) if mixture is not None else None,
                        ):
                            report["reused"] += 1
                            continue
                        reference_audio()
                        if source == "mixture":
                            audio, decoding = mixture, mixture_decoding
                        else:
                            audio, decoding = _cnn_audio.load_audio(path, tolerance, reference_samples=len(mixture))
                        original_samples = len(audio)
                        audio = align_audio(audio, len(mixture), tolerance)
                        spectrogram = log_mel(audio)
                        vector, windows = pool_track(model, spectrogram, runtime["batch_size"])
                        metadata = {**expected, "audio_sha256": digest(path), "original_samples": original_samples,
                                    "aligned_samples": len(audio), "spectrogram_frames": len(spectrogram),
                                    "windows": windows, "padded_frames": windows * 187 - len(spectrogram),
                                    "audio_decoding": {**audio_loading, **decoding},
                                    "mixture_decoding": mixture_decoding}
                        repair = None
                        if output.exists() and not args.overwrite:
                            repair = _backup_repaired_cache(output)
                            metadata["audio_repair"] = repair
                        save_cache(output, vector, metadata)
                        report["extracted"] += 1
                        if decoding["decoder"] == "ffmpeg":
                            report["decoder_fallbacks"].append({"track_id": record["track_id"],
                                                               "split": record["split"], "source": source,
                                                               "decoding": decoding})
                        if repair is not None:
                            report["repaired"].append({"track_id": record["track_id"], "split": record["split"],
                                                       "source": source, **repair})
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
              f"recovered={len(report['decoder_fallbacks'])}, repaired={len(report['repaired'])}, "
              f"failed={len(report['failed'])}; report={paths['report']}")
    return 1 if report["failed"] else 0
