# Local experiment data

Store extracted embeddings in `features/` and dataset split/label metadata in
`manifests/`. These data files are excluded from Git; this README is tracked.

```text
data/
├── features/
│   ├── fma_medium/musicfm_msd_layer7/
│   └── gtzan/musicfm_msd_layer7/
└── manifests/
    ├── fma_medium/
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

## FMA-Medium manifest

Run `python prepare_manifest.py` from
`src/stemfusion/experiments/fma_medium/`, providing `--tracks-csv`,
`--mixture-root`, and `--htdemucs-root`. The default output is
`data/manifests/fma_medium/fma_medium_htdemucs.jsonl` relative to the repository
root, regardless of the working directory. Use `--output` to choose another
location. The builder also writes metadata and report sidecars beside the
manifest. Extraction and both training YAML profiles reference this default
manifest. Older `fma_medium_legacy.*` files are retained in the same folder
for reference and are not selected by the active YAML profiles.

Generated manifests contain absolute audio paths and remain excluded from Git.
Other users should generate their own with `prepare_manifest.py`, supplying
their local metadata and audio locations. The builder preserves the splits
and labels from the supplied FMA metadata; any exclusions of incomplete tracks
are recorded in its report. Comparable experiments must use the same retained
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

`jazz.00054` is absent from all six training caches. The imported metadata remains
unchanged; downstream loading must use the same 999 available tracks for every
condition and record this exclusion. The split track IDs are disjoint. Retain
these provided stratified splits across every condition and all five seeds.

The labels are `blues=0`, `classical=1`, `country=2`, `disco=3`, `hiphop=4`,
`jazz=5`, `metal=6`, `pop=7`, `reggae=8`, and `rock=9`.

### Import audit and use

`features/gtzan/musicfm_msd_layer7/import_audit.json` records source and destination
paths, SHA-256 checksums, vector validation, retained IDs, and the missing track.
The copied caches and metadata were verified against their sources; source files
were preserved.

The GTZAN loader handles string track IDs, the `mix` filename alias, ten
classes, and either NumPy or PyTorch vectors. It validates all 18 caches
against the manifest and their import audit or shared-extractor reports.
The original metadata pickle is not used by the code. Run the dataset scripts
as described in the [GTZAN guide](../src/stemfusion/experiments/gtzan/README.md).
