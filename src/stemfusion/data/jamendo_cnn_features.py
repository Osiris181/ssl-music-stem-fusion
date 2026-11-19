"""Source-aligned track-level CNN caches and the existing Jamendo tag order."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from stemfusion.experiments._cnn_extraction import read_records
from stemfusion.features.musicnn import SOURCES, VERSION, CHECKPOINT_HASHES, SOURCE_COMMIT
from stemfusion.training.reporting import file_checksum, fingerprint

SPLITS = ("train", "validation", "test")


@dataclass
class CNNTrack:
    track_id: str
    views: dict[str, np.ndarray]
    labels: np.ndarray


@dataclass
class JamendoCNNData:
    splits: dict[str, list[CNNTrack]]
    tag_names: tuple[str, ...]
    prevalence: np.ndarray
    data_fingerprint: str
    audit: dict


def load_jamendo_cnn_features(manifest: Path, feature_dir: Path, tag_list: Path,
                              prevalence_path: Path) -> JamendoCNNData:
    rows = read_records(manifest, dataset="mtg_jamendo")
    tags = np.load(tag_list, allow_pickle=False)
    if tags.shape != (183,) or any(not str(tag).startswith("mood/theme---") for tag in tags[127:]):
        raise ValueError("Expected the existing 183-tag list, with mood/theme labels at positions 127 onward")
    names = tuple(str(tag) for tag in tags[127:])
    if len(set(names)) != 56 or set(names) != {tag for row in rows for tag in row["tags"]}:
        raise ValueError("Jamendo CNN labels must match the existing 56-tag vocabulary")
    prevalence = np.load(prevalence_path, allow_pickle=False)
    if prevalence.shape != (56,) or not np.isfinite(prevalence).all() or not ((prevalence > 0) & (prevalence < 1)).all():
        raise ValueError("Expected finite training prevalences for 56 tags")
    grouped = {split: [row for row in rows if row["split"] == split] for split in SPLITS}
    for split, values in grouped.items():
        expected_ids = {row["track_id"] for row in values}
        for source in SOURCES:
            directory = feature_dir / split / source
            if not directory.is_dir() or {path.stem for path in directory.glob("*.npz")} != expected_ids:
                raise ValueError(f"CNN cache IDs must exactly match the {split} manifest: {directory}")
    manifest_sha = file_checksum(manifest)
    splits = {split: [] for split in SPLITS}
    hashes = {source: hashlib.sha256() for source in SOURCES}
    extraction_identity = None
    with tqdm(total=len(rows) * len(SOURCES), desc="Load Jamendo CNN caches", unit="view", dynamic_ncols=True) as progress:
        for row in rows:
            views, reference_coverage = {}, None
            for source in SOURCES:
                path = feature_dir / row["split"] / source / f"{row['track_id']}.npz"
                raw = path.read_bytes()
                with np.load(io.BytesIO(raw), allow_pickle=False) as cache:
                    vector = cache["embedding"]
                    metadata = json.loads(str(cache["metadata"]))
                expected = {"dataset": "mtg_jamendo", "track_id": row["track_id"], "split": row["split"],
                            "source": source, "tags": row["tags"], "relative_path": row["relative_path"],
                            "audio_path": row["paths"][source], "manifest_sha256": manifest_sha,
                            "extraction_version": VERSION, "window_frames": 187, "hop_frames": 187,
                            "sample_rate": 16000, "tail_policy": "right_pad_log_mel_zero",
                            "pooling": "unweighted_window_mean"}
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
                    raise ValueError(f"Expected a finite float32 CNN vector (200,): {path}")
                frames, samples = metadata.get("spectrogram_frames", 0), metadata.get("aligned_samples", 0)
                if frames < 1 or samples < 1 or metadata.get("windows") != (frames + 186) // 187 or \
                   metadata.get("padded_frames") != metadata["windows"] * 187 - frames:
                    raise ValueError(f"Invalid CNN window coverage: {path}")
                coverage = (samples, frames, metadata["windows"])
                if reference_coverage is None:
                    reference_coverage = coverage
                if coverage != reference_coverage:
                    raise ValueError(f"CNN source windows do not align with mixture: {path}")
                views[source] = vector
                hashes[source].update(fingerprint({"id": row["track_id"], "sha256": hashlib.sha256(raw).hexdigest()}).encode())
                progress.update()
            targets = np.asarray([name in row["tags"] for name in names], dtype=np.float32)
            splits[row["split"]].append(CNNTrack(row["track_id"], views, targets))
    for split, records in splits.items():
        sums = np.stack([record.labels for record in records]).sum(axis=0)
        if not ((sums > 0) & (sums < len(records))).all():
            raise ValueError(f"Every tag requires positive and negative examples: {split}")
    empirical = np.stack([record.labels for record in splits["train"]]).mean(axis=0, dtype=np.float64)
    if not np.allclose(prevalence, empirical, rtol=0, atol=1e-12):
        raise ValueError("Existing prevalence weights do not match this training split and tag order")
    identity = {"manifest": manifest_sha, "tag_list": file_checksum(tag_list),
                "prevalence": file_checksum(prevalence_path), "loader": file_checksum(Path(__file__)),
                "caches": {source: value.hexdigest() for source, value in hashes.items()}}
    audit = {"split_counts": {split: len(records) for split, records in splits.items()},
             "sources": list(SOURCES), "feature_dimension": 200, "number_of_tags": 56,
             "normalization": None, "silent_vectors": "retained", "harmony_features": False,
             "prevalence_matches_training_labels": True, "extraction_identity": extraction_identity,
             "input_hashes": identity}
    return JamendoCNNData(splits, names, prevalence.copy(), fingerprint(identity), audit)


class JamendoCNNDataset(torch.utils.data.Dataset):
    """Fixed track vectors; no random segment selection or harmonic features.

    x_mert is the shared fitter's historical audio-input key. Empty harmonic
    tensors satisfy that fitter's batch interface and contain no features;
    the plain CNN classifier consumes only x_mert.
    """

    def __init__(self, records, *, sources, copies=1):
        if not sources or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
            raise ValueError("Choose distinct known CNN sources")
        if copies not in (1, 4, 5, 6) or (copies != 1 and tuple(sources) != ("mixture",)):
            raise ValueError("Only the original mixture may be repeated")
        self.records, self.sources, self.copies = records, tuple(sources), copies

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        vector = np.concatenate([record.views[source] for source in self.sources] * self.copies)
        return {"track_id": record.track_id, "labels": torch.from_numpy(record.labels),
                "x_mert": torch.from_numpy(vector), "x_chord_root": torch.empty(0, dtype=torch.long),
                "x_chord_attr": torch.empty(0, dtype=torch.long), "x_key": torch.empty(0)}
