"""FMA feature-cache adapter with a common track set for every condition."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path

from stemfusion.data.features import FeatureDataError, FeatureDataset, FeatureSplit
from stemfusion.data.fma import read_manifest
from stemfusion.training.reporting import file_checksum, fingerprint

SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")
SPLITS = ("train", "validation", "test")


def inspect_manifest(manifest: Path, number_of_classes: int = 16) -> tuple[list, dict]:
    records = read_manifest(manifest)
    mapping = {}
    reverse = {}
    for record in records:
        if len(record.track_id) != 6 or not record.track_id.isdigit():
            raise FeatureDataError("FMA track IDs must be six digits.")
        if mapping.setdefault(record.genre, record.label) != record.label or reverse.setdefault(record.label, record.genre) != record.genre:
            raise FeatureDataError("Genre names and class labels are inconsistent.")
    if set(reverse) != set(range(number_of_classes)):
        raise FeatureDataError(f"Expected contiguous labels for {number_of_classes} classes.")
    for split in SPLITS:
        selected = [record for record in records if record.split == split]
        if {record.label for record in selected} != set(reverse):
            raise FeatureDataError(f"Split {split} must contain every class.")
    return records, mapping


def cache_paths(feature_dir: Path) -> dict:
    result = {source: {split: feature_dir / f"fma_musicfm_{source}_{split}.pkl" for split in SPLITS} for source in SOURCES}
    for splits in result.values():
        for path in splits.values():
            if not path.is_file() or not path.stat().st_size:
                raise FeatureDataError(f"Missing or empty feature cache: {path}")
    return result


def load_fma_features(manifest: Path, feature_dir: Path, dimension: int = 1024,
                      number_of_classes: int = 16, require_reports: bool = True) -> FeatureDataset:
    """Read local caches only; never read or scan the original audio paths."""
    import numpy as np

    records, mapping = inspect_manifest(manifest, number_of_classes)
    paths = cache_paths(feature_dir)
    grouped = {split: [record for record in records if record.split == split] for split in SPLITS}
    splits = {split: FeatureSplit(tuple(record.track_id for record in values),
                                  np.asarray([record.label for record in values], dtype=np.int64), {})
              for split, values in grouped.items()}
    audit = {"manifest": str(manifest), "manifest_sha256": file_checksum(manifest), "dimension": dimension,
             "number_of_classes": number_of_classes, "label_mapping": mapping, "sources": list(SOURCES),
             "split_counts": dict(Counter(record.split for record in records)), "split_ids_disjoint": True,
             "normalization": None, "silent_vectors": "retained", "caches": {}, "extraction_reports": {}}
    for source in SOURCES:
        report_path = feature_dir / f"extraction_report_musicfm_{source}.json"
        if require_reports:
            if not report_path.is_file():
                raise FeatureDataError(f"Missing extraction provenance: {report_path}")
            report = json.loads(report_path.read_text())
            expected = {"source": source, "layer_index": 7, "expected_dimension": dimension, "sample_rate": 24000,
                        "temporal_pooling": "mean", "max_seconds": None, "failed": 0, "exit_code": 0, "dry_run": False}
            if any(report.get(key) != value for key, value in expected.items()):
                raise FeatureDataError(f"Extraction report does not describe the agreed MSD/layer-7 features: {report_path}")
            if Path(report.get("checkpoint_path", "")).name != "pretrained_msd.pt" or Path(report.get("stats_path", "")).name != "msd_stats.json":
                raise FeatureDataError(f"Unexpected checkpoint or statistics in {report_path}")
            reported_splits = {item["split"]: item for item in report["split_summaries"]}
            for split, values in grouped.items():
                if split not in reported_splits or reported_splits[split]["selected"] != len(values) or reported_splits[split]["failed"]:
                    raise FeatureDataError(f"Extraction split counts do not match the manifest: {report_path}")
            audit["extraction_reports"][source] = {"path": str(report_path), "sha256": file_checksum(report_path),
                                                   "checkpoint_path": report["checkpoint_path"], "layer_index": report["layer_index"]}
        for split in SPLITS:
            path = paths[source][split]
            # These pickles are the user's own local extraction artifacts.
            with path.open("rb") as handle:
                cache = pickle.load(handle)
            expected_ids = set(splits[split].track_ids)
            if not isinstance(cache, dict) or set(cache) != expected_ids:
                raise FeatureDataError(f"Cache IDs must exactly match {split} manifest IDs: {path}")
            matrix = np.empty((len(expected_ids), dimension), dtype=np.float32)
            for index, track_id in enumerate(splits[split].track_ids):
                vector = cache[track_id]
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu().numpy()
                vector = np.asarray(vector)
                if vector.shape != (dimension,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                    raise FeatureDataError(f"Expected finite float32 ({dimension},) vector: {path}, {track_id}")
                matrix[index] = vector
            splits[split].features[source] = matrix
            audit["caches"][f"{source}/{split}"] = {"path": str(path), "sha256": file_checksum(path),
                "shape": list(matrix.shape), "dtype": str(matrix.dtype), "zero_vectors": int((~matrix.any(axis=1)).sum())}
            del cache
    identity = {"manifest": audit["manifest_sha256"], "caches": {name: item["sha256"] for name, item in audit["caches"].items()}}
    identity["reports"] = {name: item["sha256"] for name, item in audit["extraction_reports"].items()}
    return FeatureDataset(splits, mapping, dimension, fingerprint(identity), audit)
