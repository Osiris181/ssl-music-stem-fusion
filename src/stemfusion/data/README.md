# Dataset manifests

## Purpose

This package converts dataset metadata and audio locations into explicit,
validated manifests shared by feature extraction and classification.

## Code

Current implementation:

| File | Purpose |
| --- | --- |
| `fma.py` | Parse FMA-Medium metadata and align mixture and HT-Demucs paths by six-digit track ID. |
| `fma_features.py` | Verify extraction reports, cache contents, exact split IDs, labels, and dimensions; load the common training track set. |
| `features.py` | Dataset-independent aligned feature containers passed to training. |
| `gtzan.py` | Read provided stratified splits, check common audio availability, assign ten-class labels, and generate audio manifests. |
| `gtzan_features.py` | Load NumPy or PyTorch GTZAN caches with exact ID alignment and extraction/import provenance validation. |
| `ismir04.py` | Preserve CSV partitions and six-class labels, exclude confirmed duplicates, and assign collision-safe IDs. |
| `ismir04_features.py` | Align NumPy/PyTorch ISMIR04 caches to manifest IDs and verify shared layer-7 extraction reports and the full-duration chunk policy. |
| `audio_duplicates.py` | Read-only file/decoded-PCM hashes, saved audit progress, and reuse of unchanged local files. |
| `mtg_jamendo.py` | Preserve provided mood/theme split records and tags, prepare separation destinations, and bind complete feature manifests with validated audio overrides. |
| `segments.py` | Load segment filenames in numeric time order and pool aligned source views using a shared random training window or all segments for evaluation. |
| `jamendo_mid_fusion.py` | Verify source-specific MERT/harmony caches and common IDs/boundaries, preserve baseline identity, and synchronize temporal sampling across views. |
| `jamendo_late_fusion.py` | Optional feature payload checksums shared by Jamendo mid/late fusion; retain mandatory structural/alignment checks and declared cache identities. The historical filename is retained for completed-run compatibility. |
| `pmemo.py` | Preserve supplied split IDs and original 0-1 continuous annotations, validate PMEmo identities, and bind chorus/stem paths for extraction without creating binary labels. |
| `pmemo_separation.py` | Prepare label-independent jobs from the same fixed PMEmo split IDs, with exact existing stem destinations and optional explicit track selection; leaves the fingerprinted feature reader unchanged. |
| `emotion_mid_fusion.py` | Verify PMEmo source-aligned MERT/harmony caches once, create separate binary targets/training-only prevalences and reuse Jamendo feature containers, encoding and sampling. |
| `emotion_fusion.py` | Explicit native-score dataset adapters with the same PMEmo cache-loading/encoding algorithm, separate raw-score target policies and training-only prevalence; used by DEAM without score normalization. |
| `deam.py` | Preserve supplied split IDs and 1-9 song-level valence/arousal means, validate numeric DEAM identities and bind existing audio for extraction without re-averaging or binarizing labels. |
| `deam_separation.py` | Label-independent separation jobs from the same DEAM split IDs, following the PMEmo preparation policy without changing the feature reader/cache identity. |
| `emomusic.py` | Preserve supplied EmoMusic split IDs and annotation columns; bind EmoMusic mixtures to same-ID DEAM stems without changing scores or generating binary labels. |

The FMA builder writes JSON Lines records containing `track_id`, `split`,
`genre`, `label`, and a `paths` mapping. It supports mixture-only manifests and
complete `htdemucs_6s` tracks distributed across multiple storage roots. All six
HT-Demucs stems for a track must come from the same root.

Important functions in `fma.py` are:

- `read_fma_medium_metadata()`: parse the official multi-row FMA CSV header;
- `build_fma_manifest()`: align labels, splits, mixtures, and stems;
- `read_manifest()`: validate the JSONL records consumed downstream.

## How to use it

Inspect all arguments with:

    stemfusion data fma-manifest --help

Alternatively, from `src/stemfusion/experiments/fma_medium/`, run
`python prepare_manifest.py --help`. This direct entry point uses the same
builder and YAML preset `configs/experiments/fma_medium/manifest.yaml` as
`stemfusion data fma-manifest`. The preset points to the local official
`data/manifests/fma_medium/metadata/tracks.csv` and configured audio roots.
Both interfaces default their output to
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl` in the repository. The
extraction and training YAML files reference that path. Explicit CLI options
override YAML. Existing manifests are protected unless `--overwrite` is used.

Multi-disk HT-Demucs examples are in `docs/fma_medium_extraction.md`.
The builder inventories all six separator outputs; revised training selects
the five named stems and excludes `other`.

Manifest builders must preserve official splits, detect duplicate IDs, and
avoid silently using different track sets for directly comparable methods.

`load_fma_features()` reads local cached embeddings, not audio files. All six
views must contain exactly the manifest IDs for each split, with finite
float32 vectors of the configured dimension. Zero vectors are retained and
counted. Input file checksums and extraction reports are recorded in a data
audit; no track intersection or additional normalization is applied.

GTZAN uses the same feature containers and shared training functions. Its
manifest is generated by `experiments/gtzan/prepare_manifest.py`, configured
through `configs/experiments/gtzan/manifest.yaml`. Labels come from the fixed
mapping and assignments from the provided split text files; the copied
metadata pickle is not read by the code. The `mix` cache alias maps to the
internal `mixture` source. See [the GTZAN guide](../experiments/gtzan/README.md).

## Music2Emo mixture baseline

`jamendo_reproduction.py` verifies cached mixture MERT/harmony artifacts and
fixed split/label alignment without opening audio. It selects saved layer
indices 5/6, preserves the author's root/quality token encoding and None-key
mode fallback, verifies training-only class frequencies, and supplies seeded
consecutive-range training pooling or all-segment evaluation means. See the
[reproduction guide](../../../docs/music2emotion_reproduction.md).
