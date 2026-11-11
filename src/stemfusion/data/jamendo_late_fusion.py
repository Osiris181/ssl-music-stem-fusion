"""Late-fusion cache loading with optional payload checksum verification.

Keep the retained reproduction/mid-fusion loaders unchanged while existing
experiments run. The trusted-cache path below preserves their representations,
declared cache identities, and structural checks; parity tests cover both paths.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath

import numpy as np
from tqdm.auto import tqdm

from stemfusion.data.jamendo_mid_fusion import (
    SOURCES, MidFusionData, MultiViewTrack, _boundaries, prepare_mid_fusion_data,
)
from stemfusion.data.jamendo_reproduction import ReproductionData, ReproductionTrack, encode_harmony
from stemfusion.data.segments import segment_files
from stemfusion.features.chords_keys import AUTHOR_COMMIT
from stemfusion.features.mert_segments import MODEL_NAME, MODEL_REVISION
from stemfusion.features.musicfm_segments import read_segment_manifest
from stemfusion.training.reporting import file_checksum, fingerprint


def _trusted_folder(directory, *, harmony=False):
    """Require complete local caches, but trust their declared payload hashes."""
    marker = json.loads((directory / "run_manifest.json").read_text())
    files = marker.get("files", {})
    if not files or not marker.get("signature"):
        raise ValueError(f"Missing completion signature/checksums: {directory}")
    required = {"metadata.json", "normalized_chords.lab", "key.lab"} if harmony else {"segments.json"}
    if not required.issubset(files):
        raise ValueError(f"Missing required feature cache entries: {directory}")
    contents = {}
    for name in files:
        if Path(name).name != name:
            raise ValueError(f"Unsafe cached filename: {name}")
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing or incomplete feature cache: {path}")
        if name in required or (not harmony and name.endswith(".npy")):
            contents[name] = path.read_bytes()
    return contents, marker


def _trusted_data(manifest, mert_dir, harmony_dir, tag_list, author_metadata, sources):
    """Match the retained loaders, omitting only per-payload SHA256 checks."""
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
    digests = {source: hashlib.sha256() for source in sources}
    counts = dict.fromkeys(sources, 0)
    missing_keys = dict.fromkeys(sources, 0)
    repaired_views, encoder_identity = 0, None
    with tqdm(total=len(rows) * len(sources), desc="Load aligned MERT + harmony (no payload hashes)",
              unit="view", dynamic_ncols=True) as progress:
        for row in rows:
            relative = str(PurePosixPath(row["relative_path"]).with_suffix(""))
            label_set = set(row["tags"])
            if not label_set or label_set - set(names):
                raise ValueError(f"Unknown or empty mood/theme labels: {row['track_id']}")
            target = np.asarray([name in label_set for name in names], dtype=np.float32)
            views = {}
            for source in sources:
                directory = mert_dir / ("mert_30s" if source == "mixture" else f"mert_30s_{source}") / relative
                arrays, marker = _trusted_folder(directory)
                metadata = json.loads(arrays["segments.json"])
                harmony, hmarker = _trusted_folder(harmony_dir / source / relative, harmony=True)
                hmeta = json.loads(harmony["metadata.json"])
                for entry in (metadata, hmeta):
                    if any(entry.get(key) != value for key, value in (
                            ("track_id", row["track_id"]), ("split", row["split"]), ("source", source))):
                        raise ValueError(f"Source/ID/split mismatch: {row['track_id']} / {source}")
                    if source != "mixture" and entry.get("audio", {}).get("path") != row["paths"].get(source):
                        raise ValueError(f"Cached audio path differs from manifest: {row['track_id']} / {source}")
                settings, hsettings = metadata["settings"], hmeta["settings"]
                if source == "mixture":
                    reference, reference_harmony = metadata, hmeta
                    if (settings.get("model_name"), settings.get("revision"), settings.get("shape"),
                            settings.get("saved_hidden_states"), settings.get("sample_rate"), settings.get("segment_seconds"),
                            settings.get("temporal_pooling")) != (
                            MODEL_NAME, MODEL_REVISION, [1, 12, 768], list(range(1, 13)), 24000, 30.0, "mean"):
                        raise ValueError("Expected pinned MERT-v1-95M all-layer 30-second features.")
                    identity = fingerprint({key: settings.get(key) for key in (
                        "model_name", "revision", "model_assets_sha256", "processor_normalize", "tail_policy",
                        "short_segment_policy", "minimum_input_samples")})
                    if encoder_identity is not None and identity != encoder_identity:
                        raise ValueError("Mixed encoder checkpoints or preprocessing rules in the baseline features.")
                    encoder_identity = identity
                    repaired_views += int("cqt_repair" in hmeta["extraction"])
                else:
                    if metadata.get("reference_mixture") != reference.get("audio"):
                        raise ValueError(f"Stem uses a different mixture reference: {row['track_id']} / {source}")
                    if settings != reference["settings"] or _boundaries(metadata) != _boundaries(reference):
                        raise ValueError(f"MERT settings or segment boundaries differ: {row['track_id']} / {source}")
                    for key in ("checkpoint_sha256", "btc_config_sha256", "btc_parameters", "author_files",
                                "package_versions", "package_sources", "midi_pitch_base", "midi_program", "midi_velocity"):
                        if hsettings.get(key) != reference_harmony["settings"].get(key):
                            raise ValueError(f"Harmonic model/settings differ across views: {key}")
                if (hsettings.get("author_commit"), hsettings.get("midi_policy"), hsettings.get("normalization")) != (
                        AUTHOR_COMMIT, "music2emotion_released", "C_major_A_minor"):
                    raise ValueError("All sources require the retained BTC/MIDI/key policy.")
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
                views[source] = ReproductionTrack(row["track_id"], np.stack(vectors), *tokens, target)
                digests[source].update(fingerprint({"id": row["track_id"], "mert": marker, "harmony": hmarker}).encode())
                counts[source] += len(vectors)
                missing_keys[source] += int(key == "None")
                progress.update()
            splits[row["split"]].append(MultiViewTrack(row["track_id"], views, target))
    for split, records in splits.items():
        if not records:
            raise ValueError(f"Empty split: {split}")
        sums = np.stack([record.labels for record in records]).sum(0)
        if not ((sums > 0) & (sums < len(records))).all():
            raise ValueError(f"Every tag needs positive and negative examples for macro AUC: {split}")
    empirical = np.stack([record.labels for record in splits["train"]]).mean(0, dtype=np.float64)
    if not np.allclose(prevalence, empirical, rtol=0, atol=1e-12):
        raise ValueError("Author class weights do not match this manifest's training labels/tag order.")
    identity = {"manifest": file_checksum(manifest), "tag_list": file_checksum(tag_list),
                "features": digests["mixture"].hexdigest(),
                "metadata": {name: file_checksum(author_metadata / name) for name in (
                    "chord_root.json", "chord_attr.json", "probas_train.npy")}}
    mixture_audit = {"split_counts": {name: len(records) for name, records in splits.items()},
                     "segment_count": counts["mixture"], "unavailable_keys_mapped_to_major_as_author": missing_keys["mixture"],
                     "cqt_repaired_mixtures": repaired_views, "saved_layer_indices": [5, 6],
                     "audio_dimension": 1536, "harmonic_dimension": 8, "key_mode_dimension": 1,
                     "prevalence_matches_training_labels": True, "input_hashes": identity}
    mixture = ReproductionData({name: [record.views["mixture"] for record in records] for name, records in splits.items()},
                               names, prevalence.copy(), fingerprint(identity), mixture_audit)
    cohort = fingerprint({name: [{"id": record.track_id, "labels": record.labels.tolist()} for record in records]
                          for name, records in splits.items()})
    identities = {"mixture": mixture.data_fingerprint,
                  **{source: fingerprint({"mixture": mixture.data_fingerprint, "stem": digests[source].hexdigest()})
                     for source in sources[1:]}}
    audit = {"split_counts": mixture_audit["split_counts"], "sources": list(sources), "segment_counts": counts,
             "unavailable_keys_mapped_to_major_as_author": missing_keys, "source_specific_harmony": True,
             "training_segment_range": "shared_across_views", "saved_layer_indices": [5, 6],
             "audio_dimension_per_source": 1536, "encoded_dimension_per_source": 1545, "cohort_fingerprint": cohort}
    return MidFusionData(mixture, splits, identities, cohort, audit)


def prepare_late_fusion_data(manifest, mert_dir, harmony_dir, tag_list, author_metadata, sources=SOURCES,
                             *, verify_checksums=True):
    if type(verify_checksums) is not bool:
        raise ValueError("verify_checksums must be a boolean.")
    if not sources or sources[0] != "mixture" or len(set(sources)) != len(sources) or set(sources) - set(SOURCES):
        raise ValueError("Select distinct sources with the original mixture first.")
    loader = prepare_mid_fusion_data if verify_checksums else _trusted_data
    data = loader(manifest, mert_dir, harmony_dir, tag_list, author_metadata, sources)
    verification = {"feature_payload_checksums_verified": verify_checksums,
                    "feature_identity_basis": "declared_completion_markers",
                    "structural_checks": "mandatory",
                    "small_metadata_checksums_computed": True}
    data.audit["cache_verification"] = verification
    data.mixture.audit["cache_verification"] = verification.copy()
    return data
