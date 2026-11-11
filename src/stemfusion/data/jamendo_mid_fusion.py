"""Verified source-aligned Jamendo caches, without reading source waveforms."""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from tqdm.auto import tqdm

from stemfusion.data.jamendo_reproduction import (
    ReproductionData, ReproductionTrack, _verified_folder, encode_harmony, prepare_reproduction_data,
)
from stemfusion.data.segments import segment_files
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.musicfm_segments import read_segment_manifest
from stemfusion.training.reporting import fingerprint

SOURCES = ("mixture", "guitar", "drums", "piano", "bass", "vocals")


@dataclass
class MultiViewTrack:
    track_id: str
    views: dict[str, ReproductionTrack]
    labels: np.ndarray


@dataclass
class MidFusionData:
    mixture: ReproductionData
    splits: dict[str, list[MultiViewTrack]]
    source_fingerprints: dict[str, str]
    cohort_fingerprint: str
    audit: dict

    def condition(self, sources):
        return ReproductionData(self.splits, self.mixture.tag_names, self.mixture.prevalence,
                                fingerprint({source: self.source_fingerprints[source] for source in sources}),
                                {**self.audit, "sources": list(sources), "cohort_fingerprint": self.cohort_fingerprint})


def _boundaries(metadata):
    return [(part["start_sample"], part["end_sample"]) for part in metadata["segments"]]


def prepare_mid_fusion_data(manifest, mert_dir, harmony_dir, tag_list, author_metadata, sources=SOURCES):
    if not sources or sources[0] != "mixture" or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("Select distinct sources with the original mixture first.")
    mixture = prepare_reproduction_data(manifest, mert_dir, harmony_dir, tag_list, author_metadata)
    records = {record.track_id: MultiViewTrack(record.track_id, {"mixture": record}, record.labels)
               for split in mixture.splits.values() for record in split}
    root_map = json.loads((author_metadata / "chord_root.json").read_text())
    quality_map = json.loads((author_metadata / "chord_attr.json").read_text())
    rows = read_segment_manifest(manifest)
    digests = {source: hashlib.sha256() for source in sources[1:]}
    missing_keys = {"mixture": mixture.audit["unavailable_keys_mapped_to_major_as_author"]}
    segment_counts = {"mixture": mixture.audit["segment_count"]}
    with tqdm(total=len(rows) * (len(sources) - 1), desc="Verify aligned stem caches", unit="view", dynamic_ncols=True) as progress:
        for row in rows:
            relative = str(PurePosixPath(row["relative_path"]).with_suffix(""))
            if len(sources) == 1:
                continue
            reference = json.loads((mert_dir / "mert_30s" / relative / "segments.json").read_text())
            reference_harmony = json.loads((harmony_dir / "mixture" / relative / "metadata.json").read_text())
            reference_bounds = _boundaries(reference)
            for source in sources[1:]:
                directory = mert_dir / f"mert_30s_{source}" / relative
                arrays, marker = _verified_folder(directory)
                metadata = json.loads(arrays["segments.json"])
                harmony, hmarker = _verified_folder(harmony_dir / source / relative)
                hmeta = json.loads(harmony["metadata.json"])
                for entry in (metadata, hmeta):
                    if any(entry.get(key) != value for key, value in (
                            ("track_id", row["track_id"]), ("split", row["split"]), ("source", source))):
                        raise ValueError(f"Source/ID/split mismatch: {row['track_id']} / {source}")
                    if entry.get("audio", {}).get("path") != row["paths"].get(source):
                        raise ValueError(f"Cached audio path differs from manifest: {row['track_id']} / {source}")
                if metadata.get("reference_mixture") != reference.get("audio"):
                    raise ValueError(f"Stem uses a different mixture reference: {row['track_id']} / {source}")
                if metadata["settings"] != reference["settings"] or _boundaries(metadata) != reference_bounds:
                    raise ValueError(f"MERT settings or segment boundaries differ: {row['track_id']} / {source}")
                for key in ("author_commit", "midi_policy", "normalization"):
                    expected = {"author_commit": AUTHOR_COMMIT, "midi_policy": "music2emotion_released",
                                "normalization": "C_major_A_minor"}[key]
                    if hmeta["settings"].get(key) != expected:
                        raise ValueError("All sources require the retained BTC/MIDI/key policy.")
                for key in ("checkpoint_sha256", "btc_config_sha256", "btc_parameters", "author_files",
                            "package_versions", "package_sources", "midi_pitch_base", "midi_program", "midi_velocity"):
                    if hmeta["settings"].get(key) != reference_harmony["settings"].get(key):
                        raise ValueError(f"Harmonic model/settings differ across views: {key}")
                files = segment_files(directory)
                if [path.name for path in files] != [part["file"] for part in metadata["segments"]]:
                    raise ValueError(f"Segment cache metadata/order mismatch: {directory}")
                vectors = []
                for path in files:
                    array = np.load(io.BytesIO(arrays[path.name]), allow_pickle=False)
                    if array.shape != (1, 12, 768) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError(f"Expected finite float32 (1, 12, 768): {path}")
                    vectors.append(np.concatenate((array[0, 5], array[0, 6])))
                key = harmony["key.lab"].decode().strip()
                tokens = encode_harmony(harmony["normalized_chords.lab"].decode().splitlines(), key, root_map, quality_map)
                track = records[row["track_id"]]
                track.views[source] = ReproductionTrack(row["track_id"], np.stack(vectors), *tokens, track.labels)
                digests[source].update(fingerprint({"id": row["track_id"], "mert": marker, "harmony": hmarker}).encode())
                missing_keys[source] = missing_keys.get(source, 0) + int(key == "None")
                segment_counts[source] = segment_counts.get(source, 0) + len(vectors)
                progress.update()
    splits = {name: [records[record.track_id] for record in split] for name, split in mixture.splits.items()}
    cohort = fingerprint({name: [{"id": record.track_id, "labels": record.labels.tolist()} for record in split]
                          for name, split in mixture.splits.items()})
    identities = {"mixture": mixture.data_fingerprint,
                  **{source: fingerprint({"mixture": mixture.data_fingerprint, "stem": digest.hexdigest()})
                     for source, digest in digests.items()}}
    audit = {"split_counts": mixture.audit["split_counts"], "sources": list(sources),
             "segment_counts": segment_counts, "unavailable_keys_mapped_to_major_as_author": missing_keys,
             "source_specific_harmony": True, "training_segment_range": "shared_across_views",
             "saved_layer_indices": [5, 6], "audio_dimension_per_source": 1536,
             "encoded_dimension_per_source": 1545, "cohort_fingerprint": cohort}
    return MidFusionData(mixture, splits, identities, cohort, audit)


class JamendoMidFusionDataset(torch.utils.data.Dataset):
    def __init__(self, records, *, sources, training):
        self.records, self.sources, self.training = records, tuple(sources), training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        views = [record.views[source] for source in self.sources]
        counts = {len(view.segments) for view in views}
        if len(counts) != 1:
            raise ValueError("Source segment counts must match before temporal sampling.")
        start, end = 0, len(views[0].segments)
        if self.training:
            start = int(np.random.randint(0, end))
            end = int(np.random.randint(start + 1, end + 1))
        return {"x_mert": torch.from_numpy(np.stack([view.segments[start:end].mean(0) for view in views])),
                "x_chord_root": torch.from_numpy(np.stack([view.roots for view in views])),
                "x_chord_attr": torch.from_numpy(np.stack([view.qualities for view in views])),
                "x_key": torch.tensor([[view.mode] for view in views], dtype=torch.long),
                "labels": torch.from_numpy(record.labels), "track_id": record.track_id}
