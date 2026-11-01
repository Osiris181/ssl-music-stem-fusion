"""ISMIR04 caches adapted to the shared genre training feature contract."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from stemfusion.data.features import FeatureDataError, FeatureDataset, FeatureSplit
from stemfusion.data.ismir04 import SOURCES, SPLITS as CSV_SPLITS, inspect_manifest
from stemfusion.training.reporting import file_checksum, fingerprint

SPLITS = tuple(CSV_SPLITS.values())
EXTRACTION_POLICY = {
    "layer_index": 7,
    "sample_rate": 24000,
    "temporal_pooling": "mean",
    "max_seconds": None,
    "chunk_seconds": 30.0,
    "chunk_pooling": "latent_frame_mean",
    "chunk_partition": "balanced_non_overlapping",
}


def cache_paths(feature_dir: Path) -> dict:
    paths = {source: {split: feature_dir / f"ismir04_musicfm_{'mix' if source == 'mixture' else source}_{split}.pkl"
                      for split in SPLITS} for source in SOURCES}
    for files in paths.values():
        for path in files.values():
            if not path.is_file() or not path.stat().st_size:
                raise FeatureDataError(f"Missing or empty feature cache: {path}")
    return paths


def _check_report(report: dict, source: str, dimension: int, counts: dict) -> None:
    expected = {**EXTRACTION_POLICY, "source": source, "expected_dimension": dimension,
                "manifest_records": sum(counts.values()), "failed": 0, "exit_code": 0, "dry_run": False}
    if any(key not in report or report[key] != value for key, value in expected.items()):
        raise FeatureDataError(f"Extraction report does not describe the agreed ISMIR04 MSD/layer-7 chunk policy: {source}")
    if Path(report.get("checkpoint_path", "")).name != "pretrained_msd.pt" or Path(report.get("stats_path", "")).name != "msd_stats.json":
        raise FeatureDataError(f"Unexpected checkpoint or statistics for {source}")
    resolved = report.get("resolved_config", {})
    if resolved.get("dataset") != "ismir04" or resolved.get("limit_per_split") is not None:
        raise FeatureDataError(f"Extraction report must describe full ISMIR04 caches: {source}")
    summaries = report.get("split_summaries", [])
    if len(summaries) != len(SPLITS) or {item["split"] for item in summaries} != set(SPLITS):
        raise FeatureDataError(f"Extraction report must include all three splits: {source}")
    for item in summaries:
        count = counts[item["split"]]
        if item.get("selected") != count or item.get("failed") != 0 or item.get("partial_output_path") is not None:
            raise FeatureDataError(f"Extraction split counts or failures do not match the manifest: {source}")
        if not item.get("skipped_existing_cache", False) and item.get("succeeded") != count:
            raise FeatureDataError(f"Extraction split is incomplete: {source}, {item['split']}")


def load_ismir04_features(manifest: Path, feature_dir: Path, dimension: int = 1024,
                          number_of_classes: int = 6, require_reports: bool = True) -> FeatureDataset:
    """Align NumPy or PyTorch vectors by manifest ID without accessing audio.

    Reports must describe the agreed full-duration, balanced 30 s chunk policy.
    Legacy caches have different IDs, labels, and extraction provenance and are
    not imported through this adapter. Valid zero vectors remain in every view.
    """
    import numpy as np

    records, mapping = inspect_manifest(manifest, number_of_classes)
    paths = cache_paths(feature_dir)
    grouped = {split: [record for record in records if record.split == split] for split in SPLITS}
    splits = {split: FeatureSplit(tuple(record.track_id for record in values),
                                 np.asarray([record.label for record in values], dtype=np.int64), {})
              for split, values in grouped.items()}
    counts = {split: len(values) for split, values in grouped.items()}
    audit = {"manifest": str(manifest), "manifest_sha256": file_checksum(manifest), "dimension": dimension,
             "number_of_classes": number_of_classes, "label_mapping": mapping, "sources": list(SOURCES),
             "split_counts": counts, "split_ids_disjoint": True, "normalization": None,
             "silent_vectors": "retained", "extraction_policy": dict(EXTRACTION_POLICY),
             "caches": {}, "extraction_reports": {}}
    for source in SOURCES:
        report_path = feature_dir / f"extraction_report_musicfm_{source}.json"
        if require_reports:
            if not report_path.is_file():
                raise FeatureDataError(f"Missing ISMIR04 extraction report: {report_path}")
            _check_report(json.loads(report_path.read_text()), source, dimension, counts)
            audit["extraction_reports"][source] = {"path": str(report_path), "sha256": file_checksum(report_path)}
        for split in SPLITS:
            path = paths[source][split]
            checksum = file_checksum(path)
            with path.open("rb") as handle:
                cache = pickle.load(handle)
            expected_ids = splits[split].track_ids
            if not isinstance(cache, dict) or set(cache) != set(expected_ids):
                raise FeatureDataError(f"Cache IDs must exactly match {split} manifest IDs: {path}")
            matrix = np.empty((len(expected_ids), dimension), dtype=np.float32)
            for index, track_id in enumerate(expected_ids):
                vector = cache[track_id]
                if hasattr(vector, "detach"):
                    vector = vector.detach().cpu().numpy()
                vector = np.asarray(vector)
                if vector.shape != (dimension,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                    raise FeatureDataError(f"Expected finite float32 ({dimension},) vector: {path}, {track_id}")
                matrix[index] = vector
            splits[split].features[source] = matrix
            audit["caches"][f"{source}/{split}"] = {"path": str(path), "sha256": checksum,
                "shape": list(matrix.shape), "dtype": str(matrix.dtype), "zero_vectors": int((~matrix.any(axis=1)).sum())}
    identity = {"manifest": audit["manifest_sha256"],
                "caches": {key: value["sha256"] for key, value in audit["caches"].items()},
                "reports": {key: value["sha256"] for key, value in audit["extraction_reports"].items()}}
    return FeatureDataset(splits, mapping, dimension, fingerprint(identity), audit)
