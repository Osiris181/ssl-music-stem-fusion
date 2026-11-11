# Local experiment data

Store extracted embeddings in `features/` and dataset split/label metadata in
`manifests/`. These data files are excluded from Git; this README is tracked.

Jamendo harmonic descriptors are saved separately under
`features/mtg_jamendo/harmony/<source>/<bucket>/<track_id>/`.
Each folder contains raw and normalized chord labels, global key, generated
MIDI (unless disabled), provenance, and verified completion checksums. These
are not MusicFM features and do not replace the separate MERT embedding stage.
The shared dataset-level `extract_chords_keys.py` uses
`configs/experiments/mtg_jamendo/chords_keys.yaml` for both representations.
Descriptor contents are not MERT-specific. The shared harmony directory is
independent of both encoder-specific feature directories. Moving it preserves
the existing files and relative artifact checksums; no re-extraction is needed.

Jamendo MERT arrays are saved by `extract_mert.py` under
`features/mtg_jamendo/mert_chord_key/mert_v1_95m/mert_30s{_<source>}/<bucket>/<track_id>/`.
Each `segment_N.npy` contains float32 `(1, 12, 768)` all-layer features, with
sample-boundary/padding metadata and verified completion checksums beside it.
The `mert_30s` directory is the original mixture. These arrays are not combined
with harmony files during extraction.

```text
data/
├── features/
│   ├── fma_medium/musicfm_msd_layer7/
│   ├── ismir04/musicfm_msd_layer7/
│   └── gtzan/musicfm_msd_layer7/
└── manifests/
    ├── mtg_jamendo/
    │   ├── tag_list.npy              # Original 183-tag order; mood/theme uses indices 127 onward
    │   └── split-0/                  # Copied Music2Emotion mood/theme train/validation/test TSVs
    ├── ismir04/                       # Fixed partitions, duplicate audit, and exclusions
    ├── fma_medium/
    │   ├── metadata/                 # Local official tracks.csv and reference genres.csv
    │   ├── fma_medium_htdemucs.jsonl
    │   └── fma_medium_htdemucs.report.json
    └── gtzan/
        ├── gtzan_htdemucs.jsonl
        ├── gtzan_htdemucs.metadata.json
        ├── gtzan_htdemucs.report.json
        ├── gtzan_data_teacher_setting.pkl
        ├── train_stratified.txt
        ├── valid_stratified.txt
        └── test_stratified.txt
```

## MTG-Jamendo split metadata

`manifests/mtg_jamendo/split-0/` contains exact copies of
`autotagging_moodtheme-{train,validation,test}.tsv` from
`/home/osiris181/Music2Emotion/dataset/jamendo/splits/split-0/`.
These define 9,949/3,802/4,231 tracks and 56 mood/theme tags. Preserve their
assignments and ordering; the broader `autotagging_moodtheme.tsv` is not the
source of this experiment's track selection.

`tag_list.npy` is copied from `Music2Emotion/dataset/jamendo/meta/`.
It contains the original full 183-tag order; its final 56 entries, beginning
at index 127, define the mood/theme label order used by the original loaders.
All four copied files were verified byte-for-byte against their sources.
Audio and extracted features remain at their existing locations. Generated
audio manifests will bind these metadata records to configured local paths.

`repair_tracks.txt` selects canonical IDs for separation through
`experiments/mtg_jamendo/prepare_manifest.py --purpose separation`. Its generated
`mtg_jamendo_separation.jsonl` preserves provided tags and split assignments
and binds selected original mixtures to separate stem destinations. This
selected-audio manifest is not the complete training manifest. It and its
report remain local; the separator reads only the explicitly selected paths.

## FMA-Medium manifest

Run `python prepare_manifest.py` from
`src/stemfusion/experiments/fma_medium/`. It loads CSV/audio paths from
`configs/experiments/fma_medium/manifest.yaml`. Explicit `--tracks-csv`,
`--mixture-root`, and `--htdemucs-root` options override those defaults.
The default output is
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl` relative to the repository
root, regardless of the working directory. Use `--output` to choose another
location. The builder also writes metadata and report sidecars beside the
manifest. Extraction and both training YAML profiles reference this default
manifest. Older `fma_medium_legacy.*` files are retained in the same folder
for reference and are not selected by the active YAML profiles.

`metadata/tracks.csv` and `metadata/genres.csv` are verified copies of the
official files in `/home/osiris181/fma_metadata/`. Only `tracks.csv` is read by
the builder; `genres.csv` is retained as a reference. Metadata copies and
generated manifests remain local under the existing `data/` Git-ignore rule.
Other users must supply their official CSV copy or override its configured path.

Generated manifests contain absolute audio paths and remain excluded from Git.
Other users should generate their own with `prepare_manifest.py`, supplying
their local metadata and audio locations. The builder preserves the splits
and labels from the supplied FMA metadata and records file-availability
diagnostics in its report. Comparable experiments must use the same retained
track set.

## GTZAN imported features

`features/gtzan/musicfm_msd_layer7/` contains 18 existing feature caches copied
from `/home/osiris181/GTZAN_experiment/gtzan_features_without_energy_teacher_setting/`:

```text
gtzan_musicfm_{source}_{split}.pkl
source: mix, guitar, drums, piano, bass, vocals
split:  train, validation, test
```

`mix` is the original mixture. Original filenames and serialized contents are
preserved. Each cache maps a track ID such as `blues.00005` to a finite NumPy
`float32` vector of shape `(1024,)`. No energy features or `other` stem caches
are included. No new extraction or feature normalization was performed.

The source extraction script is
`/home/osiris181/GTZAN_experiment/teacher_setting_get_musicfm_without_energy_GTZAN.py`.
Its main extraction path explicitly uses MusicFM-MSD, layer index 7, 24 kHz mono
audio, full-duration input, and temporal mean pooling. The pickle files do not
embed checkpoint or layer metadata; these settings are inferred from that script.

### Splits and labels

`manifests/gtzan/gtzan_data_teacher_setting.pkl` preserves the original list of
training pairs, validation pairs, test pairs, and genre-to-index mapping. Each
pair contains a track ID and its class label. It is retained as a reference
copy. Use the provided `train_stratified.txt`, `valid_stratified.txt`, and
`test_stratified.txt` files as the source of split assignments when building
the GTZAN manifest, with the fixed label mapping below.

| Split | Metadata tracks | Tracks available in every selected source |
| --- | ---: | ---: |
| Training | 500 | 499 |
| Validation | 250 | 250 |
| Test | 250 | 250 |

The imported metadata remains unchanged. Use the provided stratified split
assignments and the same source-aligned IDs across every condition and all
five seeds. The split track IDs are disjoint.

The labels are `blues=0`, `classical=1`, `country=2`, `disco=3`, `hiphop=4`,
`jazz=5`, `metal=6`, `pop=7`, `reggae=8`, and `rock=9`.

### Import audit and use

`features/gtzan/musicfm_msd_layer7/import_audit.json` records source and destination
paths, SHA-256 checksums, vector validation, and retained IDs.
The copied caches and metadata were verified against their sources; source files
were preserved.

The GTZAN loader handles string track IDs, the `mix` filename alias, ten
classes, and either NumPy or PyTorch vectors. It validates all 18 caches
against the manifest and their import audit or shared-extractor reports.
The original metadata pickle is not used by the code. Run the dataset scripts
as described in the [GTZAN guide](../src/stemfusion/experiments/gtzan/README.md).

## ISMIR04 fixed partitions

`manifests/ismir04/ismir04_htdemucs.jsonl` preserves the provided CSV partition
assignments and uses their six-class labels. A complete exact file/decoded-audio
hash audit of all 2,187 records excluded eight duplicate records, leaving
729 train, 726 validation, and 724 test tracks. No tracks were reassigned.
The manifest report records exclusions, three verified CSV filename
corrections, and one required separate stem set for the Edward Martin
`6-fantasia_6` performance. The other recording with that filename retains its
verified existing stems. See the
[ISMIR04 preparation guide](../src/stemfusion/experiments/ismir04/README.md).

Fresh MSD/layer-7 caches are written by ISMIR04's `extraction.py` into
`features/ismir04/musicfm_msd_layer7/`. They use unique manifest IDs and the
canonical train/validation/test names, with `mix` as the mixture filename
alias. Long-track chunking is explicit in the task YAML and saved reports.
Legacy feature caches remain outside this folder.

## MTG-Jamendo MusicFM segments

`manifests/mtg_jamendo/mtg_jamendo_features.jsonl` binds the copied mood/theme
split TSVs to existing mixtures and stems, with optional validated audio
overrides. Its builder reads metadata without scanning audio and preserves
the provided assignments and multi-label tags.

Build it with `python prepare_manifest.py` at the Jamendo experiment root.
Both MusicFM and MERT+chord/key audio preprocessing use this shared mapping.

`features/mtg_jamendo/musicfm_msd_layer7/` is the default destination for fresh
1,024-dimensional float32 MusicFM-MSD layer-7 arrays. Stems match the decoded
mixture's duration, and fixed-width 30-second segmentation retains partial
tails. Only inputs shorter than 1,025 samples are padded to that minimum;
supported tails are unchanged. Freshly extracted source/track folders include
actual boundaries, encoder input lengths, end adjustments, and checksums.
Previously imported arrays can remain in `features/mtg_jamendo/musicfm/`.
Imported legacy arrays retain their existing representations; their source
locations, splits, and per-segment checksums can be recorded in the consolidated
`import_audit.json` and `import_audit.records.jsonl` files. See the
[Jamendo MusicFM guide](../src/stemfusion/experiments/mtg_jamendo/musicfm/README.md)
for preparation, selected-track extraction, full extraction, and reuse rules.
