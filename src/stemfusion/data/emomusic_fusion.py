"""EmoMusic cache adapter, parity-tested against the frozen DEAM/PMEmo loader.

Supplied score columns are retained verbatim. The target/cohort implementation
is shared with DEAM; its cache loader stays frozen while DEAM training runs.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import PurePosixPath

import numpy as np
from tqdm.auto import tqdm

from stemfusion.data.jamendo_mid_fusion import SOURCES, MultiViewTrack, _boundaries
from stemfusion.data.jamendo_late_fusion import _trusted_folder
from stemfusion.data.jamendo_reproduction import ReproductionTrack, _verified_folder, encode_harmony
from stemfusion.data.emotion_fusion import EmotionFeatureData
from stemfusion.features.emotion_dataset import EmotionDatasetAdapter
from stemfusion.data.segments import segment_files
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.training.reporting import file_checksum, fingerprint


def label_policy_for(dataset: str) -> dict:
    """Binarize supplied columns at their native midpoint; do not swap targets."""
    if dataset != "emomusic":
        raise ValueError("This adapter supports EmoMusic only.")
    return {"annotation_scale": "supplied_1_9", "threshold": 5.0,
            "positive_comparison": ">=", "negative_class": "low", "positive_class": "high"}


def prepare_emotion_features(manifest, mert_dir, harmony_dir, chord_root, chord_attr,
                             sources=SOURCES, *, adapter: EmotionDatasetAdapter, verify_checksums=True) -> EmotionFeatureData:
    """Load reviewed dataset views with exactly the PMEmo cache/encoding policy."""
    label_policy = label_policy_for(adapter.dataset)
    if type(verify_checksums) is not bool:
        raise ValueError("verify_checksums must be a boolean.")
    if not sources or sources[0] != "mixture" or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("Select distinct sources with the original mixture first.")
    rows = adapter.read_manifest(manifest)
    root_map, quality_map = [json.loads(path.read_text()) for path in (chord_root, chord_attr)]
    for mapping in (root_map, quality_map):
        if (not isinstance(mapping, dict) or mapping.get("N") != 0
                or any(type(value) is not int or not 0 <= value < 14 for value in mapping.values())):
            raise ValueError("Use released root/quality vocabularies with N=0 and indices below 14.")
    splits = {split: [] for split in ("train", "validation", "test")}
    digests = {source: hashlib.sha256() for source in sources}
    counts, missing_keys = dict.fromkeys(sources, 0), dict.fromkeys(sources, 0)
    encoder_identity = None
    desc = "Verify aligned emotion caches" if verify_checksums else "Load aligned emotion caches (no payload hashes)"
    with tqdm(total=len(rows) * len(sources), desc=desc, unit="view", dynamic_ncols=True) as progress:
        for row in rows:
            relative = str(PurePosixPath(row["relative_path"]).with_suffix(""))
            views = {}
            for source in sources:
                directory = mert_dir / ("mert_30s" if source == "mixture" else f"mert_30s_{source}") / relative
                hdir = harmony_dir / source / relative
                arrays, marker = _verified_folder(directory) if verify_checksums else _trusted_folder(directory)
                harmony, hmarker = _verified_folder(hdir) if verify_checksums else _trusted_folder(hdir, harmony=True)
                if "segments.json" not in arrays or not {"metadata.json", "key.lab", "normalized_chords.lab"} <= harmony.keys():
                    raise ValueError(f"Required cache metadata is missing: {row['track_id']} / {source}")
                metadata, hmeta = json.loads(arrays["segments.json"]), json.loads(harmony["metadata.json"])
                for entry in (metadata, hmeta):
                    if any(entry.get(key) != value for key, value in (
                            ("track_id", row["track_id"]), ("split", row["split"]), ("source", source))):
                        raise ValueError(f"Source/ID/split mismatch: {row['track_id']} / {source}")
                    if entry.get("audio", {}).get("path") != row["paths"].get(source):
                        raise ValueError(f"Cached audio path differs from manifest: {row['track_id']} / {source}")
                settings, hsettings = metadata["settings"], hmeta["settings"]
                if source == "mixture":
                    reference, reference_harmony = metadata, hmeta
                    expected = {"model_name": MODEL_NAME, "revision": MODEL_REVISION, "shape": [1, 12, 768],
                                "saved_hidden_states": list(range(1, 13)), "sample_rate": 24000,
                                "segment_seconds": 30.0, "temporal_pooling": "mean", "tail_policy": "retain",
                                "processor_normalize": True, "minimum_input_samples": 400,
                                "short_segment_policy": "pad_to_minimum", "stem_alignment": "mixture_duration"}
                    if any(settings.get(key) != value for key, value in expected.items()):
                        raise ValueError("Expected the shared pinned MERT-v1-95M retained-tail/aligned extraction policy.")
                    identity = fingerprint({key: settings.get(key) for key in (
                        *expected, "model_assets_sha256", "max_alignment_seconds")})
                    if encoder_identity is not None and encoder_identity != identity:
                        raise ValueError("Mixed encoder checkpoints or preprocessing policies.")
                    encoder_identity = identity
                elif (metadata.get("reference_mixture") != reference.get("audio")
                      or settings != reference["settings"] or _boundaries(metadata) != _boundaries(reference)):
                    raise ValueError(f"MERT reference, settings or segment boundaries differ: {row['track_id']} / {source}")
                expected_harmony = {"author_commit": AUTHOR_COMMIT, "midi_policy": "music2emotion_released",
                                    "normalization": "C_major_A_minor"}
                if any(hsettings.get(key) != value for key, value in expected_harmony.items()):
                    raise ValueError("Use the shared BTC/MIDI/music21 harmonic policy.")
                for key in ("checkpoint_sha256", "btc_config_sha256", "btc_parameters", "author_files",
                            "package_versions", "package_sources", "midi_pitch_base", "midi_program", "midi_velocity"):
                    if hsettings.get(key) != reference_harmony["settings"].get(key):
                        raise ValueError(f"Harmonic model/settings differ across sources: {key}")
                files = segment_files(directory)
                if [path.name for path in files] != [part["file"] for part in metadata["segments"]]:
                    raise ValueError(f"Segment metadata/order mismatch: {directory}")
                boundaries = _boundaries(metadata)
                if not boundaries or boundaries[0][0] != 0 or any(
                        end <= start or end - start > 720000 or (index and start != boundaries[index - 1][1])
                        for index, (start, end) in enumerate(boundaries)):
                    raise ValueError(f"Invalid retained-segment boundaries: {directory}")
                vectors = []
                for path in files:
                    if path.name not in arrays:
                        raise ValueError(f"Unrecorded feature cache payload: {path}")
                    array = np.load(io.BytesIO(arrays[path.name]), allow_pickle=False)
                    if array.shape != (1, 12, 768) or array.dtype != np.float32 or not np.isfinite(array).all():
                        raise ValueError(f"Expected finite float32 (1, 12, 768): {path}")
                    vectors.append(np.concatenate((array[0, 5], array[0, 6])))
                key = harmony["key.lab"].decode().strip()
                tokens = encode_harmony(harmony["normalized_chords.lab"].decode().splitlines(), key, root_map, quality_map)
                views[source] = ReproductionTrack(row["track_id"], np.stack(vectors), *tokens, np.empty(0, dtype=np.float32))
                counts[source] += len(vectors)
                missing_keys[source] += int(key == "None")
                digests[source].update(fingerprint({"id": row["track_id"], "mert": marker, "harmony": hmarker}).encode())
                progress.update()
            splits[row["split"]].append(MultiViewTrack(row["track_id"], views, np.empty(0, dtype=np.float32)))
    metadata_identity = {"adapter": adapter.identity(), "manifest": file_checksum(manifest), "chord_root": file_checksum(chord_root),
                         "chord_attr": file_checksum(chord_attr)}
    identities = {source: fingerprint({**metadata_identity, "features": digest.hexdigest()})
                  for source, digest in digests.items()}
    audit = {"dataset": adapter.dataset, "split_counts": {split: len(records) for split, records in splits.items()},
             "sources": list(sources), "segment_counts": counts, "source_specific_harmony": True,
             "unavailable_keys_mapped_to_major_as_author": missing_keys,
             "training_segment_range": "shared_across_views", "saved_layer_indices": [5, 6],
             "audio_dimension_per_source": 1536, "encoded_dimension_per_source": 1545,
             "cache_verification": {"feature_payload_checksums_verified": verify_checksums,
                                    "feature_identity_basis": "declared_completion_markers", "structural_checks": "mandatory"}}
    return EmotionFeatureData(splits, {row["track_id"]: row["annotations"] for row in rows}, identities, audit, label_policy)
