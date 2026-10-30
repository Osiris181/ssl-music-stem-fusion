"""GTZAN caches adapted to the same feature contract as FMA-Medium."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from stemfusion.data.features import FeatureDataError, FeatureDataset, FeatureSplit
from stemfusion.data.gtzan import SOURCES, SPLITS, inspect_manifest
from stemfusion.training.reporting import file_checksum, fingerprint


def cache_paths(feature_dir: Path) -> dict:
    paths = {source: {split: feature_dir / f"gtzan_musicfm_{'mix' if source == 'mixture' else source}_{split}.pkl"
                      for split in SPLITS} for source in SOURCES}
    for files in paths.values():
        for path in files.values():
            if not path.is_file() or not path.stat().st_size:
                raise FeatureDataError(f"Missing or empty feature cache: {path}")
    return paths


def _check_report(report: dict, source: str, dimension: int, counts: dict) -> None:
    expected = {"source": source, "layer_index": 7, "expected_dimension": dimension,
                "sample_rate": 24000, "temporal_pooling": "mean", "max_seconds": None,
                "failed": 0, "exit_code": 0, "dry_run": False}
    if any(report.get(key) != value for key, value in expected.items()):
        raise FeatureDataError(f"Extraction report does not describe the agreed MSD/layer-7 features: {source}")
    if Path(report.get("checkpoint_path", "")).name != "pretrained_msd.pt" or Path(report.get("stats_path", "")).name != "msd_stats.json":
        raise FeatureDataError(f"Unexpected checkpoint or statistics for {source}")
    summaries = {item["split"]: item for item in report.get("split_summaries", [])}
    if any(split not in summaries or summaries[split]["selected"] != count or summaries[split]["failed"]
           for split, count in counts.items()):
        raise FeatureDataError(f"Extraction split counts do not match the manifest: {source}")


def load_gtzan_features(manifest: Path, feature_dir: Path, dimension: int = 1024,
                        number_of_classes: int = 10, require_reports: bool = True) -> FeatureDataset:
    """Load NumPy or PyTorch vectors; validate exact common IDs without audio I/O.

    New caches require the shared extractor's reports. Existing imported caches
    instead require their matching import audit, which records the original
    script's declared settings rather than asserting numerical re-extraction.
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
             "silent_vectors": "retained", "caches": {}, "extraction_reports": {}, "import_provenance": {}}
    imported = None
    if require_reports and any(not (feature_dir / f"extraction_report_musicfm_{source}.json").is_file() for source in SOURCES):
        import_path = feature_dir / "import_audit.json"
        if not import_path.is_file():
            raise FeatureDataError("Missing provenance: provide extraction reports or the original import_audit.json.")
        imported = json.loads(import_path.read_text())
        expected = {"checkpoint": "pretrained_msd.pt", "statistics": "msd_stats.json", "layer_index": 7,
                    "sample_rate": 24000, "channels": "mono", "max_seconds": None,
                    "pooling": "temporal_mean", "dimension": dimension, "energy_features": False}
        settings = imported.get("extraction_source", {}).get("declared_settings", {})
        if any(settings.get(key) != value for key, value in expected.items()):
            raise FeatureDataError("Imported features do not declare the agreed MSD/layer-7 settings.")
        if imported.get("retained_split_counts") != counts or imported.get("label_mapping") != mapping:
            raise FeatureDataError("Imported split counts or label mapping differ from the manifest.")
        audit["import_provenance"] = {"path": str(import_path), "sha256": file_checksum(import_path),
            "extraction_source": imported["extraction_source"], "excluded_track_ids": imported["excluded_track_ids"]}
    for source in SOURCES:
        report_path = feature_dir / f"extraction_report_musicfm_{source}.json"
        if require_reports and report_path.is_file():
            _check_report(json.loads(report_path.read_text()), source, dimension, counts)
            audit["extraction_reports"][source] = {"path": str(report_path), "sha256": file_checksum(report_path)}
        for split in SPLITS:
            path = paths[source][split]
            checksum = file_checksum(path)
            if require_reports and not report_path.is_file():
                audit_source = imported.get("source_mapping", {}).get(source, source)
                if imported.get("caches", {}).get(f"{audit_source}/{split}", {}).get("sha256") != checksum:
                    raise FeatureDataError(f"Imported cache differs from its audit: {path}")
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
    identity = {"manifest": audit["manifest_sha256"], "caches": {key: value["sha256"] for key, value in audit["caches"].items()},
                "reports": {key: value["sha256"] for key, value in audit["extraction_reports"].items()},
                "import_audit": audit["import_provenance"].get("sha256")}
    return FeatureDataset(splits, mapping, dimension, fingerprint(identity), audit)
