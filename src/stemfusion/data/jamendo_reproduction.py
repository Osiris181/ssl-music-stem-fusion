"""Verified cached inputs for the mixture-only Music2Emo reproduction."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from tqdm.auto import tqdm

from stemfusion.data.segments import segment_files
from stemfusion.features.chords_keys import AUTHOR_COMMIT, parse_intervals
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.features.musicfm_segments import read_segment_manifest
from stemfusion.training.reporting import file_checksum, fingerprint


@dataclass
class ReproductionTrack:
    track_id: str
    segments: np.ndarray
    roots: np.ndarray
    qualities: np.ndarray
    mode: int
    labels: np.ndarray


@dataclass
class ReproductionData:
    splits: dict[str, list[ReproductionTrack]]
    tag_names: tuple[str, ...]
    prevalence: np.ndarray
    data_fingerprint: str
    audit: dict


def _verified_folder(directory: Path) -> tuple[dict[str, bytes], dict]:
    marker = json.loads((directory / "run_manifest.json").read_text())
    files = marker.get("files", {})
    if not files or not marker.get("signature"):
        raise ValueError(f"Missing completion signature/checksums: {directory}")
    contents = {}
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError(f"Unsafe cached filename: {name}")
        raw = (directory / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"Changed or incomplete feature cache: {directory / name}")
        contents[name] = raw
    return contents, marker


def encode_harmony(lines, key, root_mapping, quality_mapping):
    """Match the released loader: 100 chords, zero padding, and None -> major."""
    intervals = parse_intervals(lines)
    roots, qualities = [], []
    for _, _, label in intervals:
        root, separator, quality = label.partition(":")
        if not separator:
            root = "N" if root == "X" else root
            quality = "N" if root == "N" else "maj"
        if root not in root_mapping or quality not in quality_mapping:
            raise ValueError(f"Chord unsupported by the author root/quality vocabulary: {label}")
        roots.append(root_mapping[root])
        qualities.append(quality_mapping[quality])
    root_tokens, quality_tokens = np.zeros(100, dtype=np.int64), np.zeros(100, dtype=np.int64)
    length = min(len(roots), 100)
    root_tokens[:length], quality_tokens[:length] = roots[:length], qualities[:length]
    if key == "None":
        mode = 0
    else:
        parts = key.split()
        if len(parts) != 2 or parts[-1] not in {"major", "minor"}:
            raise ValueError(f"Invalid estimated key: {key}")
        mode = int(parts[-1] == "minor")
    return root_tokens, quality_tokens, mode


def prepare_reproduction_data(manifest, mert_dir, harmony_dir, tag_list, author_metadata) -> ReproductionData:
    """Read only local feature files; never open audio paths in the manifest."""
    tags = np.load(tag_list, allow_pickle=False)
    if tags.shape != (183,) or any(not str(tag).startswith("mood/theme---") for tag in tags[127:]):
        raise ValueError("Expected the author's 183-tag list with 56 mood/theme tags at indices 127 onward.")
    names = tuple(str(tag) for tag in tags[127:])
    if len(set(names)) != 56:
        raise ValueError("Mood/theme label names must be unique.")
    root_map = json.loads((author_metadata / "chord_root.json").read_text())
    quality_map = json.loads((author_metadata / "chord_attr.json").read_text())
    for mapping in (root_map, quality_map):
        if mapping.get("N") != 0 or any(type(v) is not int or not 0 <= v < 14 for v in mapping.values()):
            raise ValueError("Expected released root/quality vocabularies with N=0 and indices below 14.")
    prevalence = np.load(author_metadata / "probas_train.npy", allow_pickle=False)
    if prevalence.shape != (56,) or not np.isfinite(prevalence).all() or not ((prevalence > 0) & (prevalence < 1)).all():
        raise ValueError("Author training prevalences must be finite probabilities for 56 tags.")
    rows = read_segment_manifest(manifest)
    splits = {name: [] for name in ("train", "validation", "test")}
    digest = hashlib.sha256()
    segment_count, missing_keys, repaired_views = 0, 0, 0
    encoder_identity = None
    for row in tqdm(rows, desc="Verify MERT + harmony", unit="track", dynamic_ncols=True):
        relative = str(PurePosixPath(row["relative_path"]).with_suffix(""))
        directory = mert_dir / "mert_30s" / relative
        arrays, marker = _verified_folder(directory)
        metadata = json.loads(arrays["segments.json"])
        harmony, harmony_marker = _verified_folder(harmony_dir / "mixture" / relative)
        harmonic_metadata = json.loads(harmony["metadata.json"])
        for entry in (metadata, harmonic_metadata):
            if any(entry.get(key) != value for key, value in (("track_id", row["track_id"]), ("split", row["split"]), ("source", "mixture"))):
                raise ValueError(f"Feature/manifest ID, split, or source mismatch: {row['track_id']}")
        settings = metadata["settings"]
        if (settings.get("model_name"), settings.get("revision"), settings.get("shape"),
                settings.get("saved_hidden_states"), settings.get("sample_rate"), settings.get("segment_seconds"),
                settings.get("temporal_pooling")) != (MODEL_NAME, MODEL_REVISION, [1, 12, 768], list(range(1, 13)), 24000, 30.0, "mean"):
            raise ValueError("Expected pinned MERT-v1-95M all-layer 30-second features.")
        identity = fingerprint({k: settings.get(k) for k in ("model_name", "revision", "model_assets_sha256", "processor_normalize", "tail_policy", "short_segment_policy", "minimum_input_samples")})
        if encoder_identity is not None and identity != encoder_identity:
            raise ValueError("Mixed encoder checkpoints or preprocessing rules in the baseline features.")
        encoder_identity = identity
        hsettings = harmonic_metadata["settings"]
        if (hsettings.get("author_commit"), hsettings.get("midi_policy"), hsettings.get("normalization")) != (
                AUTHOR_COMMIT, "music2emotion_released", "C_major_A_minor"):
            raise ValueError("The author baseline requires the released harmonic/MIDI policy.")
        files = segment_files(directory)
        if [path.name for path in files] != [part["file"] for part in metadata["segments"]]:
            raise ValueError(f"Segment cache metadata/order mismatch: {directory}")
        vectors = []
        for path in files:
            array = np.load(io.BytesIO(arrays[path.name]), allow_pickle=False)
            if array.shape != (1, 12, 768) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Expected finite float32 (1, 12, 768): {path}")
            # Follow saved-array indices [5, 6], not ambiguous paper layer names.
            vectors.append(np.concatenate((array[:, 5, :], array[:, 6, :]), axis=1).squeeze(0))
        key = harmony["key.lab"].decode().strip()
        tokens = encode_harmony(harmony["normalized_chords.lab"].decode().splitlines(), key, root_map, quality_map)
        label_set = set(row["tags"])
        if not label_set or label_set - set(names):
            raise ValueError(f"Unknown or empty mood/theme labels: {row['track_id']}")
        target = np.asarray([name in label_set for name in names], dtype=np.float32)
        splits[row["split"]].append(ReproductionTrack(row["track_id"], np.stack(vectors), *tokens, target))
        segment_count += len(vectors)
        missing_keys += int(key == "None")
        repaired_views += int("cqt_repair" in harmonic_metadata["extraction"])
        digest.update(fingerprint({"id": row["track_id"], "mert": marker, "harmony": harmony_marker}).encode())
    for split, records in splits.items():
        if not records:
            raise ValueError(f"Empty split: {split}")
        sums = np.stack([r.labels for r in records]).sum(0)
        if not ((sums > 0) & (sums < len(records))).all():
            raise ValueError(f"Every tag needs positive and negative examples for macro AUC: {split}")
    empirical = np.stack([r.labels for r in splits["train"]]).mean(0, dtype=np.float64)
    if not np.allclose(prevalence, empirical, rtol=0, atol=1e-12):
        raise ValueError("Author class weights do not match this manifest's training labels/tag order.")
    identity = {"manifest": file_checksum(manifest), "tag_list": file_checksum(tag_list), "features": digest.hexdigest(),
                "metadata": {name: file_checksum(author_metadata / name) for name in ("chord_root.json", "chord_attr.json", "probas_train.npy")}}
    audit = {"split_counts": {key: len(value) for key, value in splits.items()}, "segment_count": segment_count,
             "unavailable_keys_mapped_to_major_as_author": missing_keys, "cqt_repaired_mixtures": repaired_views,
             "saved_layer_indices": [5, 6], "audio_dimension": 1536, "harmonic_dimension": 8,
             "key_mode_dimension": 1, "prevalence_matches_training_labels": True, "input_hashes": identity}
    return ReproductionData(splits, names, prevalence.copy(), fingerprint(identity), audit)


class JamendoReproductionDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[ReproductionTrack], *, training: bool):
        self.records, self.training = records, training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        start, end = 0, len(record.segments)
        if self.training:
            start = int(np.random.randint(0, end))
            end = int(np.random.randint(start + 1, end + 1))
        return {"x_mert": torch.from_numpy(record.segments[start:end].mean(axis=0)),
                "x_chord_root": torch.from_numpy(record.roots), "x_chord_attr": torch.from_numpy(record.qualities),
                "x_key": torch.tensor([record.mode], dtype=torch.long),
                "labels": torch.from_numpy(record.labels), "track_id": record.track_id}
