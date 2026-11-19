"""Frozen CNN emotion caches shared by separate binary target classifiers.

Dataset readers validate native score scales and supplied split identities.
No audio, MERT segments, or harmonic descriptors are loaded here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from stemfusion.data.jamendo_cnn_features import CNNTrack, JamendoCNNDataset
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT
from stemfusion.training.reporting import file_checksum, fingerprint

TARGETS = ("valence", "arousal")
SPLITS = ("train", "validation", "test")


def manifest_reader(dataset):
    if dataset not in {"pmemo", "deam", "emomusic"}:
        raise ValueError("Unsupported CNN emotion dataset")
    return importlib.import_module(f"stemfusion.data.{dataset}").read_feature_manifest


@dataclass
class EmotionCNNTargetData:
    splits: dict[str, list[CNNTrack]]
    prevalence: np.ndarray
    data_fingerprint: str
    audit: dict
    tag_names: tuple[str, ...]


@dataclass
class EmotionCNNFeatures:
    splits: dict[str, list[CNNTrack]]
    annotations: dict[str, dict]
    label_policy: dict
    data_fingerprint: str
    audit: dict

    def target(self, name):
        if name not in TARGETS:
            raise ValueError("Choose valence or arousal")
        threshold = self.label_policy["threshold"]
        splits, counts = {}, {}
        for split, records in self.splits.items():
            values = [CNNTrack(record.track_id, record.views, np.asarray([
                self.annotations[record.track_id][name] >= threshold,
            ], dtype=np.float32)) for record in records]
            positive = sum(int(record.labels[0]) for record in values)
            if not 0 < positive < len(values):
                raise ValueError(f"Both low/high {name} classes are required in {split}")
            counts[split] = {"low": len(values) - positive, "high": positive}
            splits[split] = values
        prevalence = np.asarray([counts["train"]["high"] / len(splits["train"])], dtype=np.float64)
        identity = fingerprint({"features": self.data_fingerprint, "target": name,
                                "label_policy": self.label_policy})
        audit = {**self.audit, "target": name, "label_policy": dict(self.label_policy),
                 "class_counts": counts, "class_order": ["low", "high"],
                 "training_positive_prevalence": prevalence.tolist()}
        return EmotionCNNTargetData(splits, prevalence, identity, audit, (f"high_{name}",))


def load_emotion_cnn_features(manifest, feature_dir, *, dataset, label_policy):
    reader = manifest_reader(dataset)
    rows = reader(manifest)
    if any(row.get("dataset") != dataset for row in rows):
        raise ValueError("CNN manifest dataset does not match the selected experiment")
    if label_policy.get("positive_comparison") != ">=" or not np.isfinite(label_policy.get("threshold", np.nan)):
        raise ValueError("Use the native-score greater-or-equal emotion label rule")
    grouped = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    for split, records in grouped.items():
        if not records:
            raise ValueError(f"Empty CNN emotion split: {split}")
        expected = {row["track_id"] for row in records}
        for source in SOURCES:
            directory = feature_dir / split / source
            if not directory.is_dir() or {path.stem for path in directory.glob("*.npz")} != expected:
                raise ValueError(f"CNN cache IDs must exactly match the {split} manifest: {directory}")
    manifest_sha = file_checksum(manifest)
    splits = {split: [] for split in SPLITS}
    digests = {source: hashlib.sha256() for source in SOURCES}
    extraction_identity = None
    with tqdm(total=len(rows) * len(SOURCES), desc="Load emotion CNN caches", unit="view", dynamic_ncols=True) as progress:
        for row in rows:
            views, reference = {}, None
            if not all(source in row["paths"] for source in SOURCES):
                raise ValueError(f"Missing source mapping: {row['track_id']}")
            for source in SOURCES:
                path = feature_dir / row["split"] / source / f"{row['track_id']}.npz"
                raw = path.read_bytes()
                with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
                    vector = cache["embedding"]
                    metadata = json.loads(str(cache["metadata"]))
                expected = {"dataset": dataset, "track_id": row["track_id"], "split": row["split"],
                            "source": source, "annotations": row["annotations"], "relative_path": row["relative_path"],
                            "audio_path": row["paths"][source], "manifest_sha256": manifest_sha,
                            "extraction_version": VERSION, "window_frames": 187, "hop_frames": 187,
                            "sample_rate": 16000, "tail_policy": "right_pad_log_mel_zero", "pooling": "unweighted_window_mean"}
                model = metadata.get("model", {})
                if any(metadata.get(key) != value for key, value in expected.items()) or \
                   model.get("checkpoint") != "MSD_musicnn" or model.get("checkpoint_sha256") != CHECKPOINT_HASHES or \
                   model.get("source_commit") != SOURCE_COMMIT:
                    raise ValueError(f"CNN cache provenance mismatch: {path}")
                identity = {key: metadata.get(key) for key in
                            ("model", "runtime", "implementation_sha256", "duration_tolerance_seconds")}
                if extraction_identity is None:
                    extraction_identity = identity
                if identity != extraction_identity:
                    raise ValueError(f"Mixed CNN extraction settings: {path}")
                if vector.shape != (200,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
                    raise ValueError(f"Expected finite float32 CNN vector (200,): {path}")
                frames, samples = metadata.get("spectrogram_frames", 0), metadata.get("aligned_samples", 0)
                if frames < 1 or samples < 1 or metadata.get("windows") != (frames + 186) // 187 or \
                   metadata.get("padded_frames") != metadata["windows"] * 187 - frames:
                    raise ValueError(f"Invalid CNN window coverage: {path}")
                coverage = (samples, frames, metadata["windows"])
                if reference is None:
                    reference = coverage
                if coverage != reference:
                    raise ValueError(f"CNN source windows do not align with mixture: {path}")
                views[source] = vector
                digests[source].update(fingerprint({"id": row["track_id"], "sha256": hashlib.sha256(raw).hexdigest()}).encode())
                progress.update()
            splits[row["split"]].append(CNNTrack(row["track_id"], views, np.empty(0, dtype=np.float32)))
    identity = {"manifest": manifest_sha, "reader": file_checksum(Path(importlib.import_module(reader.__module__).__file__)),
                "loader": file_checksum(Path(__file__)), "caches": {source: digest.hexdigest() for source, digest in digests.items()}}
    audit = {"dataset": dataset, "split_counts": {split: len(records) for split, records in splits.items()},
             "sources": list(SOURCES), "feature_dimension": 200, "harmony_features": False,
             "normalization": None, "silent_vectors": "retained", "extraction_identity": extraction_identity,
             "input_hashes": identity}
    return EmotionCNNFeatures(splits, {row["track_id"]: dict(row["annotations"]) for row in rows},
                              dict(label_policy), fingerprint(identity), audit)
