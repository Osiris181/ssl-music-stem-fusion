"""FMA-Medium CNN cache adapter using the established genre feature contract."""
from pathlib import Path
import json
import logging

import numpy as np

from stemfusion.data.features import FeatureDataset, FeatureSplit, FeatureDataError
from stemfusion.data.fma_features import inspect_manifest
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT
from stemfusion.training.reporting import file_checksum, fingerprint

SPLITS = ("train", "validation", "test")
LOGGER = logging.getLogger(__name__)


def cache_paths(feature_dir):
    paths = {source: {split: feature_dir / split / source for split in SPLITS} for source in SOURCES}
    for values in paths.values():
        for path in values.values():
            if not path.is_dir():
                raise FeatureDataError(f"Missing CNN cache directory: {path}")
    return paths


def load_fma_cnn_features(manifest, feature_dir, dimension=200, number_of_classes=16):
    if dimension != 200 or number_of_classes != 16:
        raise FeatureDataError("FMA-Medium MSD_musicnn requires 200 features and sixteen classes")
    records, mapping = inspect_manifest(manifest, number_of_classes)
    paths = cache_paths(feature_dir)
    manifest_sha = file_checksum(manifest)
    audit = {"manifest_sha256": manifest_sha, "dimension": dimension, "number_of_classes": number_of_classes,
             "feature_backend": "msd_musicnn", "split_counts": {}, "caches": {},
             "normalization": None, "silent_vectors": "retained",
             "manifest_split_counts": {}, "skipped_missing_features": []}
    splits = {}
    extraction_identity = None
    coverage = {}
    for split in SPLITS:
        rows = [row for row in records if row.split == split]
        manifest_ids = {row.track_id for row in rows}
        audit["manifest_split_counts"][split] = len(rows)
        available = {}
        for source in SOURCES:
            directory = paths[source][split]
            available[source] = {path.stem for path in directory.glob("*.npz")}
            if available[source] - manifest_ids:
                raise FeatureDataError(f"Unexpected CNN cache IDs outside the {split} manifest: {directory}")
        complete_ids = manifest_ids.intersection(*available.values())
        skipped = [{"track_id": row.track_id, "split": split,
                    "missing_sources": [source for source in SOURCES if row.track_id not in available[source]]}
                   for row in rows if row.track_id not in complete_ids]
        audit["skipped_missing_features"].extend(skipped)
        if skipped:
            LOGGER.warning("Skipping %d incomplete FMA CNN track(s) in %s: missing source embeddings",
                           len(skipped), split)
        rows = [row for row in rows if row.track_id in complete_ids]
        if not rows:
            raise FeatureDataError(f"No complete FMA CNN tracks remain in {split}")
        if {row.label for row in rows} != set(range(number_of_classes)):
            raise FeatureDataError(f"Complete FMA CNN tracks must retain all {number_of_classes} classes in {split}")
        ids = tuple(row.track_id for row in rows)
        audit["split_counts"][split] = len(rows)
        matrices = {}
        for source in SOURCES:
            directory = paths[source][split]
            matrix = np.empty((len(rows), dimension), dtype=np.float32)
            for index, row in enumerate(rows):
                path = directory / f"{row.track_id}.npz"
                with np.load(path, allow_pickle=False) as cache:
                    vector = cache["embedding"]
                    metadata = json.loads(str(cache["metadata"]))
                expected = {"dataset": "fma_medium", "track_id": row.track_id, "split": split,
                            "source": source, "label": row.label, "manifest_sha256": manifest_sha,
                            "extraction_version": VERSION, "window_frames": 187, "hop_frames": 187,
                            "sample_rate": 16000, "tail_policy": "right_pad_log_mel_zero",
                            "pooling": "unweighted_window_mean"}
                model = metadata.get("model", {})
                if any(metadata.get(key) != value for key, value in expected.items()) or \
                   model.get("checkpoint") != "MSD_musicnn" or model.get("checkpoint_sha256") != CHECKPOINT_HASHES or \
                   model.get("source_commit") != SOURCE_COMMIT:
                    raise FeatureDataError(f"CNN cache provenance mismatch: {path}")
                identity = {key: metadata.get(key) for key in
                            ("model", "runtime", "implementation_sha256", "duration_tolerance_seconds")}
                if extraction_identity is None:
                    extraction_identity = identity
                if identity != extraction_identity:
                    raise FeatureDataError(f"Mixed CNN extraction settings: {path}")
                if vector.shape != (200,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                    raise FeatureDataError(f"Expected finite float32 CNN vector (200,): {path}")
                frames = metadata.get("spectrogram_frames", 0)
                if frames < 1 or metadata.get("windows") != (frames + 186) // 187 or \
                   metadata.get("padded_frames") != metadata["windows"] * 187 - frames:
                    raise FeatureDataError(f"Invalid window coverage: {path}")
                track_coverage = (metadata.get("aligned_samples"), frames, metadata["windows"])
                key = (split, row.track_id)
                if coverage.setdefault(key, track_coverage) != track_coverage:
                    raise FeatureDataError(f"Source windows do not align with mixture: {path}")
                matrix[index] = vector
                audit["caches"][f"{split}/{source}/{row.track_id}"] = file_checksum(path)
            matrices[source] = matrix
        splits[split] = FeatureSplit(ids, np.asarray([row.label for row in rows], dtype=np.int64), matrices)
    audit["extraction_identity"] = extraction_identity
    audit["loader_sha256"] = file_checksum(Path(__file__))
    return FeatureDataset(splits, mapping, dimension, fingerprint(audit), audit)
